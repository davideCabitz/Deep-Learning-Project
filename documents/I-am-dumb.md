# How to Actually Do This Project (A Methodology Guide)

> Answers to three starting questions: how the work should be done, when to use
> the notebook, and what to study. Read this before writing code.

---

## 1. The methodology: how research-style DL work actually gets done

The mental model to internalize: **this is an experimental science, not a
build-a-feature task.** You are not "implementing an app." You are running a loop
of *hypothesis → measure → interpret → improve*, and every claim in your final
report must be backed by a number from a measuring stick you trust.

That gives you a strict ordering. Most beginners get it backwards and start
building the exciting model (the fusion module Φ) first. Don't. The correct order
is:

**Step 1 — Build the measuring stick before anything else (the eval harness).**
The single most important artifact in the whole project is the code that computes
Recall@K / Precision@K from `celeba_evaluation.json`. If this is wrong, *every*
number you report is wrong and you won't know it. Build it first, and stress-test
it with the sanity checks the roadmap already lists (the
`celeba.filename[13] == "182651.jpg"` assertion is not optional — that indexing
gotcha will silently corrupt your scores). A good rule: a result you can't measure
doesn't exist.

**Step 2 — Get the data + features pipeline rock-solid and frozen.**
Extract all CLIP image features once, cache them to disk (`.pt` tensors indexed by
dataset index). After this, every experiment is a fast matrix-vector cosine. This
is the "boring engineering" that makes the next two weeks of iteration feasible.
Do it carefully once.

**Step 3 — Establish the floor: the dumbest possible baseline (Tier 0).**
`q = normalize(v_ref + Σα·t⁺ − Σβ·t⁻)`, score by cosine. This will probably
perform *poorly*. **That is a feature, not a failure** — it gives you (a) a number
to beat, (b) proof your whole pipeline runs end-to-end, and (c) a concrete
motivation for everything fancier. In your report, a weak baseline that you then
beat is worth more than a strong model with nothing to compare against.

**Step 4 — Climb one rung at a time, always measuring against the rung below.**
CLAY reproduction → training-free improvement → trained Φ. The discipline: *change
one thing, measure, write down the number and your interpretation of why it
moved.* If you change three things at once and the score goes up, you've learned
nothing about which change helped — and "thoroughness" is an explicit grading
criterion.

The thought process at each rung is always the same four questions:
- What do I *expect* to happen, and why? (Write it down first — this is your hypothesis.)
- What actually happened? (The number.)
- Does the gap between expectation and reality teach me something? (This is where the report-worthy insight lives.)
- What's the smallest next experiment that tests my new understanding?

**Practical habits that separate a good submission from a frustrating two weeks:**
- **Sanity-check obsessively.** Before trusting a model, can you retrieve the
  reference image itself with near-perfect score? Can a tiny model *overfit* 20
  training examples to ~100%? If not, there's a bug — find it before scaling up.
- **Fix seeds** everywhere (`torch`, `numpy`, `random`) so results are reproducible.
- **Keep an experiment log** — a running text/markdown file: date, what you
  changed, the resulting numbers, your one-line interpretation. This file
  *becomes* your report's Results & Discussion section. Future-you in week two
  will not remember why run #14 was better than #11.
- **Commit often** with meaningful messages. With 3 people, git is your
  coordination layer.

## 2. The notebook question

Here's the key distinction the assignment hides: **the notebook is the
deliverable, but it is a terrible place to *develop*.** Treat them as two
different things.

**During development (weeks 1–2):** write your real code as `.py` modules —
exactly the `data.py`, `features.py`, `eval.py`, `methods.py`, `fusion.py`,
`viz.py` split the roadmap proposes. Why?
- Python files **diff and merge in git**; notebooks don't (a `.ipynb` is a giant
  JSON blob — three people editing one notebook = constant merge hell).
- You can `import` them, write quick tests, and reuse code across experiments
  without copy-paste.
- They force modular, readable code — which is itself a grading criterion.

For *exploration and experiments*, yes — each person can keep their own **scratch
notebook** (or scratch script) that imports the shared modules and tries things.
These are disposable. They are *not* the deliverable and don't need to be clean.
This is the "keep a separate one and merge later" instinct — and it's correct,
with one refinement: you don't literally merge scratch notebooks, you **distill**
them.

**The deliverable notebook is assembled at the end**, in Phase C, from the
stabilized modules. You curate it: import (or paste, since it must be
self-contained) the final clean code into logical cells, and write the markdown
report *around* it — methodology + math, experimental setup, results tables,
qualitative retrieval grids, discussion. The spec says it should "read like a
detailed report interwoven with executable code." That is an editorial act done
once, at the end, on top of code that already works — not something you grow
organically while debugging.

So the workflow is: **develop in `.py` → experiment in scratch notebooks → at the
end, curate one clean report notebook.** Don't try to make your working notebook
gradually become the final one; that path leads to a 3000-line mess you're
untangling at 2am on day 14.

## 3. What to study, how deeply, and where

You don't need to become a vision-language researcher. You need a **working
understanding of about seven concepts** — deep enough to implement and explain,
no deeper. Ranked by importance, with depth target and where to learn each.

**Tier A — you cannot do the project without these (study first, ~2 days):**

1. **Embeddings + cosine similarity + the unit hypersphere.** CLIP maps images and
   text to vectors; "similar" = high cosine = small angle. L2-normalization puts
   everything on a unit sphere. *Depth: solid intuition + be able to code it.*
   Source: any "intro to embeddings/cosine similarity" explainer.

2. **CLIP itself** (Radford et al., 2021, "Learning Transferable Visual Models From
   Natural Language Supervision"). How it's trained (contrastive image-text), what
   the shared embedding space *is*, and how zero-shot classification/retrieval
   works on top of it. *Depth: read intro + method, understand the contrastive
   objective and the shared space; skim the rest.* The single most important
   external paper for you. Source: paper + OpenAI's CLIP blog post + a good video
   walkthrough (e.g. Yannic Kilcher's CLIP video).

3. **Retrieval metrics: Recall@K and Precision@K.** What they mean, why Recall@K is
   the "did I get at least one right" hit rate. *Depth: fully — you're
   implementing these.* Source: the spec defines them precisely; that's enough.

4. **The modality gap.** A known quirk: image embeddings and text embeddings in
   CLIP live in *separate cones* of the space, so naively adding a text vector to
   an image vector is geometrically crude. This single fact *motivates your entire
   project*. *Depth: understand the intuition well enough to explain it in the
   report.* Source: "Mind the Gap" (Liang et al., 2022) — abstract + figures.

**Tier B — needed for your specific methods (study as you reach each tier, ~2–3 days):**

5. **The two assigned papers, GDE (Berasi 2025) and CLAY (Lim 2026)**, already in
   your `documents/` folder. *Depth: GDE — understand the compositionality idea
   conceptually. CLAY — understand it mechanically: the log-map to tangent space,
   SVD to get a textual subspace, the projection `P = VₖVₖᵀ`, the rotation H for
   the modality gap, and crucially the "naïve stacking before SVD" that you are
   attacking.* Read these slowly, twice. They are the heart of the assignment.

6. **Contrastive learning / InfoNCE / triplet loss.** Only matters once you train
   Φ. The idea: pull the query toward valid targets, push it away from violating
   images. *Depth: understand the loss and be able to use it.* Source: any
   "InfoNCE explained" / SimCLR explainer.

7. **Cross-attention / FiLM gating** (the candidate Φ architectures). *Depth:
   enough to implement a small one in PyTorch.* Source: the attention mechanism
   from "Attention is All You Need" (just the attention part) + the FiLM abstract.

**Tier C — context, optional, skim only:** SVD as a linear-algebra refresher (you
mostly call `torch.linalg.svd`); SAEs/interpretability (explicitly a stretch
goal — ignore unless you finish early).

**How to study, concretely:**
- **Don't read passively.** For CLIP and CLAY, read with a pen: write the
  equations in your own notation, and after each paper write a 5-sentence summary
  "what problem, what trick, what's the weakness." That summary is a draft of your
  report's background section.
- **Learn just-in-time.** Don't study contrastive learning in week one — you won't
  need it until Tier 2b. Study CLIP and the modality gap now because *everything*
  depends on them.
- **Prefer one good video + one careful paper read** over ten blog posts.
- **The fastest way to understand an embedding space is to poke it.** Once you have
  features cached, spend an hour just retrieving: pick an image, find its nearest
  neighbors, add a text vector, see what changes. You'll learn more about CLIP's
  behavior in that hour than in a day of reading — and those observations are
  report gold.

---

**Realistic first-week target:** by end of day ~3 the *boring* stuff is
bulletproof (eval harness + cached features + Tier-0 baseline producing real
numbers), and you personally have CLIP + the modality gap understood. Everything
creative comes after, and rests on that foundation.
