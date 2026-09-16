# Your project in plain language (zero assumed background)

Written to be readable with no prior knowledge of machine learning or image
processing. Every technical word is defined the first time it appears.

---

## 1. What the project is actually about

Underwater photographs look bad. Water absorbs red light within a few metres, so
everything turns blue or green. Floating particles scatter light, so the picture
looks hazy. Contrast is low, so it looks flat.

There are **two separate jobs** you could do about this:

- **Job A — make the picture better.** This is called *image enhancement*.
- **Job B — say how good the result is.** This is called *quality assessment*.

**Your project does Job A with ordinary mathematics (no AI), and Job B with AI.**

This is the single most important thing to understand, because it is the thing
most likely to be misunderstood in a viva:

> Your neural network does **not** repair photographs. It is a **judge**. It looks
> at an already-cleaned-up photograph and predicts a score for how good that
> cleanup was.

If someone asks "does your CNN enhance underwater images?", the correct answer is
**no** — the classical mathematics does the enhancing, and the CNN predicts the
score. Say that clearly and you are safe. Say "yes" and you will be asked to show
the enhanced output, which your CNN cannot produce.

---

## 2. The dataset (UIEB)

You use a public dataset called **UIEB**. It contains **890 pairs** of pictures.

Each pair is:

- a **raw** image — the bad, blue, murky original photograph
- a **reference** image — a nicer-looking version of the same scene

There are also **60 extra images** with no reference. You cannot use those for
training, because there is nothing to compare against, so you have no score to
teach the program. You correctly exclude them. (Your documentation should say
"950 total, 60 excluded, 890 used" — otherwise the numbers look inconsistent.)

### An important honesty point about the reference images

The reference images are **not** the true original scene. There is no such thing
here — nobody has a perfect photo of what that reef actually looked like.

Instead, the dataset's creators ran several existing enhancement programs on each
raw image, showed the results to human volunteers, and **kept whichever version
the volunteers preferred**.

So the reference is "what humans liked best", not "what is physically true".

**Why this matters:** your scores measure *agreement with one particular look*.
A result could be technically excellent but score poorly because it doesn't match
the style the volunteers happened to prefer. You must state this as a limitation.
It is a real limitation of the dataset, not a mistake you made — every project
using UIEB has it.

---

## 3. The two scores you predict

### SSIM (Structural Similarity Index)

- A number from **0 to 1**. 1 means the two pictures are identical.
- It compares **structure** — shapes, edges, patterns — in small windows across
  the image, which is closer to how human eyes judge than raw pixel differences.
- In your data the actual values run from **0.24 to 0.97**.

### PSNR (Peak Signal-to-Noise Ratio)

- Measured in **decibels (dB)**. Higher is better. There is no fixed maximum.
- It is based on raw pixel-by-pixel differences.
- Rough intuition: below ~20 dB is poor, ~30 dB is decent, ~40 dB is very close.
- In your data the actual values run from **9.7 to 29.6 dB**.

Your program predicts **both** of these, at the same time, from one photograph.

---

## 4. What a "feature" is

Instead of handing a computer the whole photograph (which is hundreds of
thousands of numbers), you first measure **25 summary numbers** that describe it.

Think of describing a person to a friend. You could show a photograph, or you
could say "175 cm tall, 70 kg, brown hair, brown eyes". The second version is a
list of **features** — a few numbers that capture what matters.

Your 25 numbers fall into four groups:

**Brightness and contrast (6)** — average brightness, how spread out the
brightness values are, how much the darkest and brightest differ, how much
variety there is in the tones.

**Colour (7)** — average amount of red, green and blue; how colourful the image
is overall; the **ratio of red to green+blue** (this one matters a lot
underwater, because lost red light is the main problem); how saturated and how
bright the colours are.

**Texture (8)** — measured using a **GLCM**. That stands for *Gray-Level
Co-occurrence Matrix*, which is just a table answering "when a pixel of
brightness X appears, how often is its neighbour brightness Y?" From that table
you get numbers describing whether the surface looks rough, smooth, regular or
random.

**Edges and sharpness (4)** — what fraction of the picture is edges; how strong
the brightness changes are; how sharp the image is; how many distinctive
"interest points" a standard detector finds.

Because a human being decided what to measure, these are called **handcrafted
features**. The alternative is to let the computer invent its own descriptions —
that is what a neural network does (section 7).

---

## 5. Shrinking 25 numbers down to fewer

Twenty-five numbers is a lot, and many of them say the same thing. Your project's
main research contribution is a careful four-step process for choosing well.

### Step 1 — Rank them: which numbers actually matter?

**Tool: Random Forest.** Imagine 200 separate decision trees. A decision tree is
a chain of yes/no questions ("is brightness above 120? then is red-ratio below
0.4? then..."). Each tree is built from a different random sample of the data, so
each makes different mistakes. When you ask all 200 and average their answers,
you get something much more reliable than any single tree. This is a standard,
well-respected method.

**How to rank: permutation importance.** This is the clever part, and worth
explaining properly in a viva.

You take your trained Random Forest and measure how accurate it is. Then you take
**one** column of numbers — say "average blue" — and **shuffle it into random
order**, so those values no longer belong to the right photographs. Then you
measure accuracy again.

- If accuracy **crashes**, that column was doing real work → it is important.
- If accuracy **doesn't change**, that column was contributing nothing.

It is like finding out which ingredient matters in a recipe by leaving it out and
seeing whether the cake collapses.

**A detail you should mention:** shuffling sometimes makes the score *better*.
That happens when a column was pure noise, and removing the noise helps. In your
data this happened to **10 of the 25** columns for SSIM and **5 of 25** for PSNR.
Since your ranking method rescales everything so the worst becomes 0.000 and the
best becomes 1.000, a feature showing **0.000 means "actively harmful"**, not
"neutral". Don't let anyone read 0.000 as "mildly unimportant".

### Step 2 — Remove duplicates

Some of the 25 numbers measure essentially the same thing. **Correlation** is the
tool that detects this: it answers "when number A goes up, does number B go up
too?" It is written as **r**, running from −1 to +1.

- r = +1 → they move perfectly together (knowing one tells you the other)
- r = 0 → no relationship
- r = −1 → they move perfectly opposite

Your rule: **if two features have r of 0.90 or more (in either direction), keep
whichever one ranked better in Step 1 and throw the other away.**

This is genuinely worth knowing about your own data. Four of your 25 features are
not merely similar — they are **mathematically the same thing written
differently**, which I verified to the last decimal place:

- `std` and `rms_contrast` are **exactly identical** (difference: 0.0)
- `variance` is just `std` multiplied by itself
- `ASM` is just `energy` multiplied by itself
- `ASM` and `glcm_variance` are the same number scaled by a constant

So your "25 features" really contain only about **21 genuinely different pieces
of information**. That is not a mistake — your filtering step correctly removes
the duplicates — but if you claim "25 features" in a viva, someone may ask, and
you should be able to answer that four of them are provably redundant and your
filter removes them.

### Step 3 — Try different amounts

You test keeping the top 6, top 8, top 10, top 12, and top 14, and see which
group predicts best.

### Step 4 — Pick the winner

**Here is the honest result, and it is not what your old notes claimed.** The
scores were:

| how many kept | score |
|---|---|
| 14 | 0.4533 |
| 12 | 0.4398 |
| 10 | 0.4380 |
| 8 | 0.4400 |
| 6 | 0.4186 |

These are **all essentially the same**. The winner (14) beats third place (10) by
0.015, which is far too small to be meaningful.

So the correct conclusion is:

> **Throwing away the duplicate features (25 → 14) costs nothing — that is a
> genuine win, because a simpler model is better. But choosing exactly *how many*
> to keep makes no real difference. There is no magic "best k".**

Do not claim "k = 10 was found to be optimal". It wasn't. Claiming it invites an
easy attack.

---

## 6. Splitting the data — and why the order matters

You divide the 890 pairs into three groups:

| group | count | what it is for |
|---|---|---|
| **train** | 623 | the textbook — the program studies this |
| **validation** | 134 | practice exams — used to make decisions during development |
| **test** | 133 | the sealed final exam — opened exactly once, at the very end |

**Why this matters so much:** if you use the final exam to *decide which features
to keep*, you are cheating. You would be choosing the features that happen to
work on those exact 133 photos, and your final score would look better than the
program's real ability. This cheating has a name: **data leakage**.

Your project handles this **correctly**, and it is one of its real strengths:
ranking uses train, duplicate-removal uses train, the "how many to keep" decision
uses validation, and the test set is only opened at the very end. Many published
undergraduate projects get this wrong. You should say so explicitly in your
report.

**One extra subtlety you handle correctly:** seven pairs of images in UIEB are
byte-for-byte identical files. If one copy landed in train and the other in test,
your "final exam" would contain a question the program had literally already
seen. Your code groups identical files together so they always go to the same
side. I checked all seven groups — all correct, zero leaks.

---

## 7. What the CNN is

A **Convolutional Neural Network (CNN)** is a program that looks at raw pixels
and **learns its own descriptions**, instead of being told what to measure.

Where your 25 handcrafted features are like giving someone a fixed checklist
("measure brightness, measure redness..."), a CNN works out for itself what to
look for — perhaps it discovers that a certain pattern of greenish haze matters,
which no human thought to measure.

Yours is deliberately **small**: about 430,000 adjustable numbers. That is tiny
for this kind of model, and it is the right choice — you only have 623 training
images, and a big model would memorise them instead of learning general rules.

Two design choices worth being able to defend:

**Global average pooling.** Near the end, instead of flattening a large grid of
numbers into an enormous list (which would add millions of adjustable numbers),
each channel is reduced to a single average. This keeps the model small. Small is
good when you have 623 images.

**No pre-trained weights, on purpose.** Many projects start from a network
already trained on millions of ordinary photographs. You deliberately don't, so
that when you compare "CNN alone" against "CNN + your 25 features", any
difference must come from *your features* and not from someone else's million
photos. That is careful experimental design and you should claim credit for it.

### "Fusion"

Your final model, called **HybridCNN**, feeds in **both**:

```
   cleaned-up photo ──► CNN ──► its own learned description (256 numbers) ┐
                                                                          ├─► combine ─► predicts SSIM and PSNR
   the 14 handcrafted features ──► small network (32 numbers) ────────────┘
```

Combining two sources of information like this is called **fusion**. Your
research question is simply: **does adding the handcrafted features to the CNN
actually help, or is the CNN fine on its own?**

That is a real, answerable question, and your project is set up to answer it
honestly either way.

---

## 8. Understanding your scores — what R² means

**R²** ("R squared") answers one question: *how much of the variation in the real
scores did your program explain?*

- **R² = 1.0** → perfect, every prediction exactly right
- **R² = 0.0** → no better than just guessing the average score every time
- **negative** → *worse* than guessing the average

### Your actual results

| what you predict | R² | plain reading |
|---|---|---|
| SSIM | **0.36** | explains about a third of the variation |
| PSNR | **0.22** | explains about a fifth |

**Is that good?** Be straight about it: it is **weak to moderate**. It is much
better than guessing, but it is not a strong predictor. Do not describe it as
"high accuracy" — an examiner will look at 0.36 and lose confidence in
everything else you say.

The honest framing is much stronger:

> "Predicting a full-reference quality score from an image alone, without access
> to the reference, is inherently hard — you are being asked to guess how close
> something is to a picture you have never seen. An R² of 0.36 for SSIM shows the
> handcrafted descriptors carry real but partial signal."

### Also report the error in dB

For PSNR, R² hides something important. Your average error is about **2.8 dB**.
Given that the whole dataset spans only about 20 dB, being wrong by 2.8 dB is a
big deal. Say this yourself before someone else does.

### Why small differences between models mean nothing

You only have **133 test images**. That is a small exam. If you re-ran the whole
thing with a different random 133 images, your R² would land somewhere between
roughly **0.09 and 0.49** for SSIM. That range is called a **confidence
interval**, and I have added code to compute and report it.

**Consequence:** if model A scores 0.36 and model B scores 0.38, that difference
is **noise**. You cannot claim one is better. Any claim of improvement needs to
be bigger than the confidence interval, or backed by running the experiment
several times with different random seeds and testing properly.

---

## 9. THE BUG — what was actually wrong

This is the most important section. Your project had a genuine bug that silently
corrupted your main result.

### The analogy

A teacher has two pieces of paper.

- **Paper 1** is the class list in admission-number order: *Alice, Bob, Charlie,
  Diana...*
- **Paper 2** is the marks scoreboard, sorted alphabetically: *Alice, Bob,
  Charlie, Diana...* — but with **different marks attached**, because it was
  sorted a different way.

The teacher tapes Paper 1's *names* column next to Paper 2's *marks* column and
announces the top scorer. But the two lists were in different orders, so the
names line up with the **wrong marks**. The teacher confidently announces that
Diana got 100%, when actually Alice did.

Everything after that — the prize giving, the report card, the ranking — is
internally consistent and looks completely fine. **Nothing would ever flag it.**

### What happened in your code

Your program computed two rankings (one for SSIM, one for PSNR), each sorted by
its own scores. It then built a table combining them — and attached the **feature
names** in the wrong order.

The names were in "canonical" order (the order they are defined in your config).
The scores were matched up alphabetically. So every feature got the score
belonging to a *different* feature.

The shift is completely predictable, and it explains the exact symptom:

- `red_ratio` was genuinely the best feature for **both** SSIM and PSNR
- alphabetically, `red_ratio` sits in position **21**
- position 21 in your canonical list is `edge_density`
- so `edge_density` was printed as the #1 feature with a perfect score of 1.000

**Your old notes and the review you were given both said `edge_density` was your
most important feature. It was not. It was tenth.**

### How I caught it

Not by reading the numbers — by adding a **self-check** to the code. The check
says: "the feature my combined table calls #1 must be the same feature my SSIM
table calls #1." That is a rule that must always be true. When I ran it, the check
**failed**, which proved the bug was live in the code and not just a leftover old
file.

I then proved the diagnosis by mathematically un-shuffling the old file — after
which it matched the two correct ranking files **exactly, to the last decimal**.

### Why it was invisible

Because the corruption happened at the *first* step, every step after it consumed
the corrupted output and behaved perfectly sensibly. Duplicate-removal removed
the wrong features, but removed them correctly. Subset evaluation scored the
wrong list, but scored it correctly. There was no error anywhere to see — only a
wrong answer that was internally consistent.

**This is the most dangerous kind of bug, and the lesson generalises: add checks
that must always be true, and let them crash the program when they aren't.**

### What changed as a result

Your final feature list is **completely different**:

```
OLD (wrong, 10):  edge_density, rms_contrast, mean_blue, contrast,
                  mean_saturation, mean_value, homogeneity, mean_red,
                  glcm_variance, dynamic_range

NEW (correct, 14): red_ratio, dynamic_range, entropy, correlation,
                  keypoint_density, homogeneity, mean_blue, mean_value,
                  mean, dissimilarity, laplacian_variance, glcm_variance,
                  mean_saturation, variance
```

Only **6 names** appear in both lists. Your corrected results are slightly
*better* for PSNR (0.2249 instead of 0.2029) and about the same for SSIM.

There is also a nice side-effect. Before the fix, the ranking looked
**meaningless** — the order of importance had essentially no relationship to how
well each feature actually performed on its own. After the fix, that relationship
is **real and statistically significant**. The bug had been hiding the fact that
your ranking method works.

---

## 10. Everything I fixed

| problem | plain description | status |
|---|---|---|
| The name/score mismatch | section 9 above | **fixed**, plus a permanent self-check |
| Your README was cut off | the file literally ended mid-word with the text `...[truncated 4045 chars]`, losing 9 sections including your Limitations section | **fixed** — all sections rebuilt |
| No `.gitignore` | the 1.6 GB of images would have been committed into your repository, which has a size limit | **fixed** |
| No augmentation | the program saw the same 623 photos the same way every time | **fixed** — see below |
| No error ranges | you reported single numbers with no indication of how uncertain they are | **fixed** |
| Seven small documentation errors | wrong split numbers in comments, references to "8 features" when it is 14, and similar | **fixed** |

### About "augmentation"

With only 623 training photos, a useful trick is to create extra variety by
flipping images — a mirror image of a reef is still a reef.

But your situation is special: your **answer key is a score comparing two
pictures**. If you alter the picture, the correct score might change, and you
would be teaching the program with wrong answers.

So I **measured** which flips are safe rather than guessing:

- **Flipping left-right: safe.** All 25 features stay exactly the same (I checked
  to 15 decimal places) and the scores don't change.
- **Flipping upside-down: also safe.** Same result.
- **Rotating 90°: NOT safe.** It changes 8 of the texture measurements, because
  those measurements specifically look at *horizontal* neighbours, and rotating
  turns horizontal into vertical.
- **Changing brightness or colour: definitely NOT safe.** That genuinely changes
  the correct answer.

Result: the program now trains on 4× the variety (original, flipped sideways,
flipped vertically, both), using only transformations I proved are harmless. This
matters because it reduces memorisation without corrupting the answers.

---

## 11. What this means for your thesis — concretely

1. **Change "10 features" to "14 features" everywhere**, and use the new names.
2. **Change your results table** to: SSIM R² = 0.3576, PSNR R² = 0.2249,
   PSNR error = 2.77 dB.
3. **Drop the claim that k = 10 is optimal.** Say instead that the sweep was flat
   and that the value of the selection stage is removing duplicates, not picking
   a specific count.
4. **Drop any claim that `edge_density` is your most important feature.** The most
   important feature is `red_ratio` — which is also intuitively satisfying, since
   red loss *is* the defining problem of underwater imaging. That is a nice thing
   to be able to say in a viva: your method independently rediscovered the
   physics.
5. **Add the Limitations section** (references are human preferences, not truth;
   the model predicts scores rather than enhancing; modest R²; non-standard
   dataset split).
6. **Do not say your CNN enhances images.** Say it predicts the quality that your
   classical pipeline achieves.

---

## 12. What is running right now

The AI models are training in the background. So far:

| model | what it is | SSIM R² | PSNR R² |
|---|---|---|---|
| Random Forest | the 200-decision-trees method, using your 14 features | **0.3576** | **0.2249** |
| MLP | a small simple neural network using the same 14 features | 0.3255 | 0.1407 |
| Image-only CNN | looks at pixels, no handcrafted features | training | |
| Hybrid CNN | pixels **+** your 14 features — your proposed model | queued | |
| Hybrid CNN + all 25 | same but with every feature — to test whether selection helps | queued | |

**One result is already interesting and slightly awkward:** the simple Random
Forest is currently **beating** the neural network on the same features. That is
not unusual with only 623 training images — tree methods often win on small
tabular data — but it is a fact you will need to report.

If the Hybrid CNN ends up beating everything, you have a clean positive result.
If it does not, you still have a valid project: your own code contains the rule
*"if the hybrid does not beat the baselines, that is the finding — do not re-tune
on test to force a win."* Holding to that rule is worth more marks than a
suspiciously good number.
