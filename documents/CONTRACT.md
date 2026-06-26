# Data Contract — read this before writing any code

This file is the **agreement between the two of us** about the exact shape and
format of every piece of data we pass between our two halves of the project.
It is boring on purpose. Boring = nothing crashes when we plug our code together.

> Rule of thumb: if your code produces or consumes any of the things below,
> it MUST match this file exactly. If you need to change something here,
> tell the other person and edit this file — never change it silently.

---

## 0. The golden rule: images are referred to by INDEX, never by filename

The ground-truth file (`Evaluation/celeba_evaluation.json`) refers to every image
by a **number**, e.g. `13`. That number means *"the image at position 13 in the
PyTorch CelebA dataset"* — i.e. you get it with `celeba[13]`.

**It does NOT mean the file `000013.jpg`.** Those are two different images.

- ✅ Always: `img, attrs = celeba[13]`
- ❌ Never: `Image.open(".../000013.jpg")`

Every image, everywhere in our code — in rankings, in ground truth, in the
feature table — is identified by this **dataset index** (a plain integer).
This is the single most common way this kind of project silently breaks.

---

## 1. The 40 attribute names, in order (the master list)

CelebA labels each face with 40 yes/no attributes. The **order below is fixed**
(it is the order in `list_attr_celeba.txt`). Column `i` of the attribute tensor
means the attribute at position `i` in this list. We both use this exact order.

```
 0  5_o_Clock_Shadow      10  Blurry             20  Male                  30  Sideburns
 1  Arched_Eyebrows       11  Brown_Hair         21  Mouth_Slightly_Open   31  Smiling
 2  Attractive            12  Bushy_Eyebrows     22  Mustache              32  Straight_Hair
 3  Bags_Under_Eyes       13  Chubby             23  Narrow_Eyes           33  Wavy_Hair
 4  Bald                  14  Double_Chin        24  No_Beard              34  Wearing_Earrings
 5  Bangs                 15  Eyeglasses         25  Oval_Face             35  Wearing_Hat
 6  Big_Lips              16  Goatee             26  Pale_Skin             36  Wearing_Lipstick
 7  Big_Nose              17  Gray_Hair          27  Pointy_Nose           37  Wearing_Necklace
 8  Black_Hair            18  Heavy_Makeup       28  Receding_Hairline     38  Wearing_Necktie
 9  Blond_Hair            19  High_Cheekbones    29  Rosy_Cheeks           39  Young
```

In code, build this once and share it:

```python
ATTRIBUTE_NAMES = [
    "5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald",
    "Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair",
    "Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair",
    "Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache",
    "Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose",
    "Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair",
    "Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick",
    "Wearing_Necklace","Wearing_Necktie","Young",
]
ATTR_TO_IDX = {name: i for i, name in enumerate(ATTRIBUTE_NAMES)}  # "Smiling" -> 31
```

---

## 2. The attribute tensor (produced by the data half)

A table of the yes/no labels for every image in the test split.

| Property | Agreed value |
|---|---|
| Variable name | `attributes` |
| Type | `torch.Tensor` |
| Shape | `[N, 40]` where `N` = number of test images (~19,962) |
| dtype | `torch.float32` |
| Values | `+1.0` = attribute present, `0.0` = attribute absent |
| Row meaning | row `i` is the attributes of `celeba[i]` (same index as everything else) |
| Column meaning | column `j` is the attribute `ATTRIBUTE_NAMES[j]` |

> Note: the raw CelebA file uses `-1` for "absent". We convert `-1 → 0` so the
> tensor is a clean 0/1 mask. (If you'd rather keep `-1`, that's fine — just pick
> one here and write it down. We picked **0/1**.)

Saved to disk as: `attributes_test.pt`

---

## 3. How to read a query string (the query parser)

Queries look like these (from the JSON):

```
"+Smiling"
"+Black_Hair, -Wavy_Hair"
"+Wearing_Lipstick, -Heavy_Makeup, +Smiling"
```

Rules:
- Split the string on commas.
- Each piece starts with `+` (must HAVE this attribute) or `-` (must NOT have it).
- The rest of the piece is an attribute name from the master list (section 1).

The parser turns a query string into two lists of attribute names:

```python
def parse_query(query_str):
    pos, neg = [], []
    for piece in query_str.split(","):
        piece = piece.strip()
        sign, name = piece[0], piece[1:]
        if sign == "+": pos.append(name)
        elif sign == "-": neg.append(name)
    return pos, neg   # e.g. (["Black_Hair"], ["Wavy_Hair"])
```

We call these two lists `T_pos` and `T_neg` throughout the code.

---

## 4. The ground-truth file (`Evaluation/celeba_evaluation.json`)

This file is **authoritative** — we do not modify it. Its structure:

- It is a **list of 14 query objects**. (Note: 14, not 12 — and `-Young` appears
  twice. We still need to confirm against the spec which are the "mandatory 12"
  for the final results table. TODO before final report.)
- Each object has:
  - `"query"`: the query string (e.g. `"+Smiling"`).
  - `"ground_truth"`: a dictionary mapping **source image** → **list of valid target images**.

Important detail about the keys:

```python
ground_truth = {
    "13": [325, 456, 579, ...],   # keys are STRINGS, values are ints
    "27": [...],
    ...
}
```

- The keys are **strings** like `"13"` — cast them to `int` immediately: `int("13")`.
- Both keys (sources) and values (targets) are **dataset indices** (section 0).
- A "valid target" already satisfies the rule: matches the query AND is within
  Hamming distance ≤ 2 of the source on the other attributes. We do NOT recompute
  this — we trust the JSON.

---

## 5. What a "ranking" is (the output of every retrieval method)

Every method we build (Tier-0 baseline, CLAY, our fusion module) answers the same
question: *given a query, which images match best?* The answer always has the
same format so the evaluation code can score any method identically.

| Property | Agreed value |
|---|---|
| Type | `list[int]` (or 1-D tensor/array of ints) |
| Contents | **dataset indices**, ordered **best match first** |
| Length | the full ranked corpus (we slice top-K inside the evaluator) |
| Source exclusion | the **source image's own index must be removed** from the ranking |

---

## 6. The CLIP image-feature table (produced by the model half)

Every test image turned into a vector of numbers by CLIP.

| Property | Agreed value |
|---|---|
| Variable name | `image_features` |
| Type | `torch.Tensor` |
| Shape | `[N, 512]` (CLIP ViT-B/32 gives 512 numbers per image) |
| dtype | `torch.float32` |
| Row meaning | row `i` is the features of `celeba[i]` (same index as everything else) |
| Normalized? | **Yes** — each row is L2-normalized (length 1) at save time |

Saved to disk as: `clip_image_features_test.pt`
(Plus `clip_image_features_train.pt` later, for training the fusion module.)

---

## 7. The one shared function signature

This is the seam where our two halves meet. Every method matches this shape:

```python
def score(T_pos, T_neg, v_ref_idx, image_features, attributes, **kwargs):
    """
    Inputs:
      T_pos, T_neg   : lists of attribute names (from parse_query)
      v_ref_idx      : int, the source image's dataset index
      image_features : [N, 512] tensor (section 6)
      attributes     : [N, 40] tensor (section 2)
    Returns:
      ranking : list[int] of dataset indices, best-first, source excluded (section 5)
    """
```

The evaluation code only ever calls something with this signature, so any method
we write — baseline or fancy — drops straight into the same scoring pipeline.

---

## Who owns what (so we don't step on each other)

- **Data half:** sections 1, 2, 4 (load CelebA, build `attributes`, parse the GT JSON, sanity-check indices).
- **Model/eval half:** sections 3, 5, 6, 7 (query parser, CLIP features, the `score` signature, and the evaluation metrics that consume rankings).

Both halves only truly meet at sections 0, 5, and 7 — keep those exact.
