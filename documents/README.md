# Documents

The final method, code and results are in [`notebooks/deliverable_report.ipynb`](../notebooks/deliverable_report.ipynb) and the project [README](../README.md). This folder records **how we got there**: the plans, theory briefs, design reports, dead ends and progress logs written during the project. They are kept as written. Each one starts with a banner that says who wrote it, when, and what happened next.

[DECISIONS.md](DECISIONS.md) is the one-page version of the story: 15 decisions, each linked to the document where it was made.

## Where to start

| Time | Read |
|---|---|
| **5 min** | The project [README](../README.md), then the timeline at the top of [DECISIONS.md](DECISIONS.md) |
| **30 min** | All of [DECISIONS.md](DECISIONS.md), then [13_FinalDestination](process/13_FinalDestination.md), the audit that triggered the pivot to probabilistic scoring |
| **Deep dive** | The [deliverable notebook](../notebooks/deliverable_report.ipynb), then the process documents in order |

## Contents

| Path | What it is |
|---|---|
| [project_specification.md](project_specification.md) | The assignment specification, **provided by the course**. All design decisions trace back to it. |
| [links.md](links.md) | Links collected from the course material (**course-provided**). |
| [DECISIONS.md](DECISIONS.md) | The key decisions and their measured effect |
| [literature/](literature/) | [SOTA review](literature/SOTA.md) and the [reading list](literature/README.md) of papers studied in depth |
| [process/](process/) | 16 design documents, numbered in the order they were written |
| [logs/](logs/) | Per-person progress logs: [Alfonso](logs/Alfonso.md), [Davide](logs/Davide.md) |

The exploratory notebooks behind these documents (data preparation, feature extraction, early fusion training) are in [`notebooks/progressions/`](../notebooks/progressions/).

## Process documents

| # | Document | Author | Date | Outcome |
|---|---|---|---|---|
| 01 | [Roadmap](process/01_ROADMAP.md) | Davide | Jun 18 | Initial 10-day plan |
| 02 | [Data contract](process/02_CONTRACT.md) | Davide | Jun 26 | Team agreement, in force throughout |
| 03 | [Training-free split](process/03_TRAINING_FREE_SPLIT.md) | Davide | Jun 27 | Two parallel tracks (V and S) |
| 04 | [Track V theory](process/04_TrackV.md) | Alfonso | Jun 28 | Became the GDE reproduction and GDE-VP |
| 05 | [Track S paper notes](process/05_S_notions.md) | Davide | Jun 28 | Reading notes for 06 |
| 06 | [Track S plan](process/06_S_plan.md) | Davide | Jun 29 | Became Asymmetric Subspaces |
| 07 | [Original training-free approaches](process/07_OriginalTrainingFreeApproaches.md) | Alfonso | Jun 29 | None shipped; CKS anticipated BAP |
| 08 | [CLAY adaptive-k investigation](process/08_CLAY_TNEG_ADAPTIVEK_INVESTIGATION.md) | Alfonso | Jun 29 | Negative result, retired |
| 09 | [Subspace-union plan](process/09_planSV.md) | Davide | Jun 29 | Became Subspace Union |
| 10 | [Fusion model guide](process/10_FusionModelGuide.md) | Alfonso | Jun 29 | Became the cross-attention Φ |
| 11 | [Tier-3 variations](process/11_Tier3_Variations_Report.md) | Alfonso | Jul 1 | Composer was the first to break the negation floor |
| 12 | [Negation-aware fusion](process/12_Negation_Fusion_Report.md) | Alfonso | Jul 1 | Dual Score kept |
| 13 | [FinalDestination](process/13_FinalDestination.md) | Davide | Jul 2 | **The pivot to Bayesian scoring (BAP)** |
| 14 | [NextStep](process/14_NextStep.md) | Davide | Jul 2 | Became BAP+ and the final A-BAP |
| 15 | [AwA2 transfer analysis](process/15_awa2_negative_transfer_analysis.md) | Alfonso | Jul 3 | Superseded by a benchmark correction |
| 16 | [OVAD transfer analysis](process/16_ovad_negative_transfer_analysis.md) | Alfonso | Jul 5 | Negative transfer, explained |

## Glossary

The process documents use the working names from the time ("Tier-N", "Track V/S", "Bayes-Φ", "CD-Bayes"). The README and the deliverable use the final names.

| Working name | Final name | Code |
|---|---|---|
| Tier-0 | CLIP (vanilla arithmetic baseline) | [`src/Baselines/clip_baseline.py`](../src/Baselines/clip_baseline.py) |
| Tier-0 enhanced | CLIP Enhanced | [`src/Training-Free/clip_enhanced.py`](../src/Training-Free/clip_enhanced.py) |
| Tier-1 CLAY | CLAY reproduction | [`src/Baselines/clay.py`](../src/Baselines/clay.py) |
| Tier-1 GDE, originally Track V | GDE reproduction | [`src/Baselines/gde.py`](../src/Baselines/gde.py) |
| Tier-2a, Track S | Asymmetric Subspaces | [`src/Training-Free/asymmetric_subspaces.py`](../src/Training-Free/asymmetric_subspaces.py) |
| Tier-2b, Track V (visual extension) | GDE-VP | [`src/Training-Free/gde_vp.py`](../src/Training-Free/gde_vp.py) |
| Tier-2c, Track SV-union | Subspace Union | [`src/Training-Free/subspace_union.py`](../src/Training-Free/subspace_union.py) |
| Tier-2d | DGP (Dynamic Gated Projection) | [`src/Training-Free/dgp.py`](../src/Training-Free/dgp.py) |
| Tier-3 DGP Φ, Fusion-DGP | Φ-Gate | [`src/Trained/phi_gate.py`](../src/Trained/phi_gate.py) |
| Tier-3c | Cross-attention Φ | [`src/Trained/cross_attention_phi.py`](../src/Trained/cross_attention_phi.py) |
| Tier-3 Combined | Φ-Combined | [`src/Trained/phi_combined.py`](../src/Trained/phi_combined.py) |
| Tier-3 DualScore (P2) | Dual Score | [`src/Trained/dual_score.py`](../src/Trained/dual_score.py) |
| Tier-3/4 hybrid | Attribute probes + probe-penalty negation | [`src/Trained/attribute_probes.py`](../src/Trained/attribute_probes.py) |
| Tier-4 Bayes-Φ, Stage A | BAP (closed form) | [`src/Trained/bap.py`](../src/Trained/bap.py) |
| Tier-4 Bayes-Φ, Stage B | First version of A-BAP | [`src/Trained/bap.py`](../src/Trained/bap.py) |
| Tier-5 CD-Bayes, Stage A′ | BAP+ | [`src/Trained/cdbayes.py`](../src/Trained/cdbayes.py) |
| Tier-5 CD-Bayes, Stage B′ (ListNet) | **A-BAP** (final model) | [`src/Trained/cdbayes.py`](../src/Trained/cdbayes.py) |

## Old file names

On July 5 the code was reorganized (commit `73aa2b0`). File names in the process documents and logs refer to the earlier layout:

| Before | After |
|---|---|
| `src/tier0.py` | `src/Baselines/clip_baseline.py` |
| `src/tier0_enhanced.py` | `src/Training-Free/clip_enhanced.py` |
| `src/tier1_CLAY.py` (earlier `tier1.py`) | `src/Baselines/clay.py` |
| `src/tier1_GDE.py` (earlier `tier2a_visual.py`, Track V) | `src/Baselines/gde.py` |
| `src/tier2a.py` (earlier `tier2a_S.py`; planned as `tier2a_subspace.py`) | `src/Training-Free/asymmetric_subspaces.py` |
| `src/tier2b.py` (earlier `tier2a_visual_extension.py`) | `src/Training-Free/gde_vp.py` |
| `src/tier2c.py` (planned as `tier2a_SVunion.py`) | `src/Training-Free/subspace_union.py` |
| `src/tier2d_dgp.py` | `src/Training-Free/dgp.py` |
| `src/tier3_dgp.py` (earlier `fusion_dgp.py`) | `src/Trained/phi_gate.py` |
| `src/tier3c_xattn.py` | `src/Trained/cross_attention_phi.py` |
| `src/tier3_combined.py` | `src/Trained/phi_combined.py` |
| `src/tier3_dualscore.py` | `src/Trained/dual_score.py` |
| `src/tier3_hybrid.py` | `src/Trained/attribute_probes.py` |
| `src/tier3_train_utils.py` | `src/Trained/fusion_train_utils.py` |
| `src/tier4_bayes.py` | `src/Trained/bap.py` |
| `src/tier5_cdbayes.py` | `src/Trained/cdbayes.py` |
| `tier3_composer`, `tier3_contrastive`, `tier3_negation`, `tier3_neggate`, `tier3_polaritycomposer` | Removed (did not ship); still in git history |
| `documents/<name>.md` | `documents/process/NN_<name>.md` or `documents/logs/` |
| `documents/papers/*.md` | Not published; see the [reading list](literature/README.md) |
