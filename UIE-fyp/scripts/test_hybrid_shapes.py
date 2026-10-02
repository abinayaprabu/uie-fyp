#!/usr/bin/env python3
"""Shape / unit tests for the feature-guided enhancer (required pre-flight test).

Every tensor that enters or leaves a major block is printed, so the
architecture can be checked against the locked diagram before any training:

    python scripts/test_hybrid_shapes.py

Checks performed (all must pass; the script exits 1 otherwise):
  1. encoder -> 3 skips + 256x14x14 bottleneck, correct shapes
  2. decoder -> 3x224x224 in [0, 1]
  3. FiLM identity at initialisation: feature-guided output == image-only
     output (same shared weights) -> the ablation starts from a level field
  4. after two optimizer steps the output DEPENDS on the features (nonzero
     conditioning gradients; changing the features changes the output)
  5. image-only variant accepts a feature argument and ignores it
  6. parameter counts, printed for the report
  7. one forward+backward timing measurement (for the compute plan)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cnn.hybrid.conditioning import FeatureConditioning  # noqa: E402
from cnn.hybrid.model import FeatureGuidedEnhancer, count_params  # noqa: E402

torch.manual_seed(42)
K = 10                     # a stand-in for "whatever Stage A selected"
B = 2
IMG = torch.rand(B, 3, 224, 224)
FEAT = torch.randn(B, K)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    print("=" * 78)
    print("SHAPE / UNIT TESTS -- feature-guided enhancer")
    print("=" * 78)

    guided = FeatureGuidedEnhancer(n_features=K, use_features=True)
    plain = FeatureGuidedEnhancer(n_features=K, use_features=False)
    print(f"\nparameters: feature-guided {count_params(guided):,} | "
          f"image-only {count_params(plain):,} | "
          f"conditioning branch {count_params(guided.conditioning):,}")

    print("\n[1] encoder shapes")
    bottleneck, (s1, s2, s3) = guided.encoder(IMG)
    print(f"    input          {tuple(IMG.shape)}")
    print(f"    skip s1        {tuple(s1.shape)}")
    print(f"    skip s2        {tuple(s2.shape)}")
    print(f"    skip s3        {tuple(s3.shape)}")
    print(f"    bottleneck     {tuple(bottleneck.shape)}")
    check("bottleneck is 256x14x14", tuple(bottleneck.shape) == (B, 256, 14, 14),
          str(tuple(bottleneck.shape)))
    check("skips are 32/64/128 at 112/56/28",
          tuple(s1.shape) == (B, 32, 112, 112)
          and tuple(s2.shape) == (B, 64, 56, 56)
          and tuple(s3.shape) == (B, 128, 28, 28))

    print("\n[2] conditioning + decoder shapes")
    gamma, beta = guided.conditioning(FEAT)
    mod = guided.conditioning.modulate(bottleneck, FEAT)
    out = guided(IMG, FEAT)
    print(f"    gamma/beta     {tuple(gamma.shape)} / {tuple(beta.shape)}")
    print(f"    conditioned    {tuple(mod.shape)}")
    print(f"    output         {tuple(out.shape)}  range [{out.min():.3f}, {out.max():.3f}]")
    check("gamma/beta are (B, 256)", tuple(gamma.shape) == (B, 256)
          and tuple(beta.shape) == (B, 256))
    check("conditioned keeps 256x14x14", tuple(mod.shape) == (B, 256, 14, 14))
    check("output is 3x224x224 in [0,1]",
          tuple(out.shape) == (B, 3, 224, 224)
          and float(out.min()) >= 0.0 and float(out.max()) <= 1.0)

    print("\n[3] FiLM identity at initialisation (clean ablation start)")
    plain.load_state_dict(guided.state_dict(), strict=False)
    check("all conditioning params initialised to zero",
          all(float(p.detach().abs().max()) == 0.0 for p in guided.conditioning.to_film.parameters()))
    with torch.no_grad():
        out_guided0 = guided(IMG, FEAT)
        out_plain = plain(IMG)
    delta0 = float((out_guided0 - out_plain).abs().max())
    print(f"    max |feature-guided - image-only| at init = {delta0:.3e}")
    check("identical outputs at initialisation", delta0 < 1e-6, f"delta={delta0:.2e}")

    print("\n[4] gradients + feature dependence after 2 optimizer steps")
    opt = torch.optim.Adam(guided.parameters(), lr=1e-3)
    target = torch.rand_like(out)
    for _ in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(guided(IMG, FEAT), target)
        loss.backward()
        opt.step()
    gnorm = float(guided.conditioning.to_film.weight.grad.norm())
    n_nonzero = sum(1 for p in guided.parameters()
                    if p.grad is not None and float(p.grad.abs().sum()) > 0)
    print(f"    conditioning grad norm = {gnorm:.3e}; "
          f"parameters with nonzero grads: {n_nonzero}")
    check("conditioning gradients are nonzero after steps", gnorm > 0)
    with torch.no_grad():
        out_a = guided(IMG, FEAT)
        out_b = guided(IMG, torch.zeros_like(FEAT))
        out_c = guided(IMG, FEAT.roll(1, dims=0))       # shuffled across batch
    d_feat = float((out_a - out_b).abs().max())
    d_shuf = float((out_a - out_c).abs().max())
    print(f"    max |normal - zeroed features|   = {d_feat:.3e}")
    print(f"    max |normal - shuffled features| = {d_shuf:.3e}")
    check("features affect the output after training steps",
          d_feat > 1e-4 and d_shuf > 1e-4)

    print("\n[5] image-only variant ignores a feature argument")
    with torch.no_grad():
        p_none = plain(IMG, None)
        p_feat = plain(IMG, FEAT)
    check("same output with/without features",
          float((p_none - p_feat).abs().max()) == 0.0)

    print("\n[6] timing (forward+backward, batch 16, 2 CPU threads)")
    torch.set_num_threads(2)
    del out, out_guided0, out_plain, out_a, out_b, out_c, mod, gamma, beta
    import gc; gc.collect()
    for label, model, feats in (("image_only", plain, None),
                                ("feature_guided", guided, torch.randn(8, K))):
        x = torch.rand(8, 3, 224, 224)
        t0 = time.time()
        for _ in range(3):
            model.zero_grad()
            loss = torch.nn.functional.l1_loss(model(x, feats), torch.rand(8, 3, 224, 224))
            loss.backward()
            model.zero_grad()
        dt = (time.time() - t0) / 3
        print(f"    {label:16s}: {dt:.2f} s per 8-image step "
              f"-> ~{dt * 78 / 60:.1f} min per 623-image epoch (78 steps, batch 8)")
    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED CHECKS: {failures}")
        return 1
    print("ALL SHAPE / UNIT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
