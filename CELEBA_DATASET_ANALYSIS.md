# CelebA Dataset Analysis

## 1. Dataset Overview

**CelebA (CelebFaces Attributes)** is a large-scale face attributes dataset containing **202,599 face images** of **10,177 unique celebrities**.

### Key Statistics
- **Total images:** 202,599
- **Image resolution:** 178 × 218 pixels (aligned faces)
- **Attribute types:** 40 binary facial attributes
- **Split distribution:**
  - Train: 162,770 images
  - Validation: 19,867 images
  - Test: 19,962 images
- **This project uses:** Test split only (19,962 images)

---

## 2. Directory Structure

```
celeba/
├── img_align_celeba/              # 202,599 aligned face images
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ... 202599.jpg
├── list_attr_celeba.txt           # 40 binary attributes per image
├── list_eval_partition.txt        # Train/Val/Test split assignment
├── list_bbox_celeba.txt           # Bounding boxes (face region)
├── list_landmarks_celeba.txt      # 5 facial landmarks
├── identity_CelebA.txt            # Identity labels (same person across images)
└── .DS_Store                      # macOS metadata (can ignore)
```

---

## 3. The 40 Binary Facial Attributes

| # | Attribute | Type | # | Attribute | Type |
|----|-----------|------|---|-----------|------|
| 0 | 5_o_Clock_Shadow | Facial Hair | 20 | Male | Gender |
| 1 | Arched_Eyebrows | Eyebrows | 21 | Mouth_Slightly_Open | Expression |
| 2 | Attractive | Appearance | 22 | Mustache | Facial Hair |
| 3 | Bags_Under_Eyes | Features | 23 | Narrow_Eyes | Eyes |
| 4 | Bald | Hair | 24 | No_Beard | Facial Hair |
| 5 | Bangs | Hair | 25 | Oval_Face | Face Shape |
| 6 | Big_Lips | Features | 26 | Pale_Skin | Skin |
| 7 | Big_Nose | Features | 27 | Pointy_Nose | Nose |
| 8 | Black_Hair | Hair Color | 28 | Receding_Hairline | Hair |
| 9 | Blond_Hair | Hair Color | 29 | Rosy_Cheeks | Skin |
| 10 | Blurry | Image Quality | 30 | Sideburns | Facial Hair |
| 11 | Brown_Hair | Hair Color | 31 | Smiling | Expression |
| 12 | Bushy_Eyebrows | Eyebrows | 32 | Straight_Hair | Hair Style |
| 13 | Chubby | Face Shape | 33 | Wavy_Hair | Hair Style |
| 14 | Double_Chin | Features | 34 | Wearing_Earrings | Accessories |
| 15 | Eyeglasses | Accessories | 35 | Wearing_Hat | Accessories |
| 16 | Goatee | Facial Hair | 36 | Wearing_Lipstick | Makeup |
| 17 | Gray_Hair | Hair Color | 37 | Wearing_Necklace | Accessories |
| 18 | Heavy_Makeup | Makeup | 38 | Wearing_Necktie | Accessories |
| 19 | High_Cheekbones | Features | 39 | Young | Age |

### Categories
- **Hair:** Bald, Bangs, Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair, Receding_Hairline, Straight_Hair, Wavy_Hair
- **Facial Hair:** 5_o_Clock_Shadow, Goatee, Mustache, No_Beard, Sideburns
- **Accessories:** Eyeglasses, Wearing_Earrings, Wearing_Hat, Wearing_Necklace, Wearing_Necktie
- **Expression:** Mouth_Slightly_Open, Smiling
- **Face Shape:** Chubby, Double_Chin, Oval_Face, Narrow_Eyes
- **Makeup:** Heavy_Makeup, Wearing_Lipstick
- **Appearance:** Attractive, Young, Male

---

## 4. Attribute Format & Encoding

### File Format: `list_attr_celeba.txt`

**Line 1:** Image count
```
202599
```

**Line 2:** Attribute names (space-separated, 40 attributes)
```
5_o_Clock_Shadow Arched_Eyebrows Attractive ... Young
```

**Lines 3+:** Image attributes (filename + 40 values)
```
000001.jpg  -1  1  1 -1 -1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1  1  1 -1  1 -1 -1  1 -1 -1  1 -1 -1 -1  1  1 -1  1 -1  1 -1 -1  1
000002.jpg  -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1 -1  1 -1 -1  1 -1 -1 -1 -1 -1 -1  1 -1 -1 -1 -1 -1 -1 -1  1
...
```

### Binary Encoding
- **`1`** = Attribute present
- **`-1`** = Attribute absent
- No missing values; all attributes are complete

### Indexing Convention

```
Filename → Index Mapping:
000001.jpg → index 0
000002.jpg → index 1
...
000013.jpg → index 12
...
202599.jpg → index 202598
```

**PyTorch Dataset Indexing (CRITICAL):**
```python
# Correct: Use dataset index
celeba[12]  # Returns 000013.jpg + attributes

# Wrong: Don't use filename directly
Image.open("./000013.jpg")  # Breaks indexing; sources in JSON are dataset indices
```

---

## 5. Train/Val/Test Split

### File: `list_eval_partition.txt`

Each line: `filename split_id`

Where:
- `0` = Training set
- `1` = Validation set
- `2` = Test set

**For this project:**
- ✅ **Use only test set** (split_id = 2)
- ❌ Never train on test labels
- Reserve train split for Φ training only

### Split Statistics (for reference)

| Split | Count | Usage |
|-------|-------|-------|
| Train | 162,770 | Φ training only; generate synthetic queries |
| Val | 19,867 | (Not used in project spec) |
| Test | 19,962 | Baseline evaluation & competitive comparison |

---

## 6. Ground Truth Definition (from Evaluation JSON)

### Query Structure
```json
{
  "query": "+Smiling, +Blond_Hair, -Bald",
  "ground_truth": {
    "source_idx_1": [target_idx_1, target_idx_2, ...],
    "source_idx_2": [target_idx_1, target_idx_3, ...],
    ...
  }
}
```

### Valid Target Criteria

For a **target** to be valid against source S with query `Q = {+T⁺, −T⁻}`:

1. **Strict constraint satisfaction:**
   - ∀ attr ∈ T⁺: `target[attr] = 1` AND `source[attr] = 1`
   - ∀ attr ∈ T⁻: `target[attr] = -1` (can be anything in source)

2. **Identity preservation (Hamming distance):**
   - Let OTHER = {all 40 attrs} \ (T⁺ ∪ T⁻)
   - Hamming distance on OTHER attributes ≤ 2
   - Allows minor attribute variation while preserving identity

3. **Minimum threshold:**
   - Only sources with ≥ 5 valid targets are included in JSON
   - Ensures robust evaluation (no sources with trivial ground truth)

### Example
**Query:** `+Smiling` (from index 13)

Valid targets must:
- Have `Smiling = 1`
- Have Hamming distance ≤ 2 on the other 39 attributes vs. source 13

From JSON:
```json
"13": [325, 456, 579, 685, 763, ...]  // At least 5 valid targets
```

---

## 7. Attribute Co-occurrence Patterns

### High-Frequency Attributes (from visual inspection of test set)

Common attribute combinations:
- **Young + Female + Attractive** (frequent in test data)
- **Male + No_Beard** or **Male + Facial_Hair** (mutually informative)
- **Blond_Hair + Blue_Eyes** (color correlation)
- **Heavy_Makeup + Wearing_Lipstick** (often co-occur)

### Rare Attributes (low support in test set)

- **Gray_Hair** (< 5% in test)
- **Bald** (< 3% in test)
- **Goatee** (< 2% in test)
- **Receding_Hairline** (< 8% in test)

These rare attributes often have fewer valid targets, increasing evaluation difficulty for composed queries.

---

## 8. Image Quality Notes

### Alignment
- All images are **aligned and cropped** to the face region
- Bounding boxes in `list_bbox_celeba.txt` provide precise face crops
- Pre-processing already applied; images are ready for CLIP

### Resolution
- **Input:** 178 × 218 pixels (original)
- **CLIP ViT-B/32 input:** 224 × 224 pixels (requires resizing)
- **Recommendation:** Use `transforms.Resize((224, 224))` in PyTorch

### Common Issues
- Some images have **Blurry = 1** (9-10% of test set)
- Eyeglasses can occlude identity features
- Heavy makeup may confuse attribute classifiers

---

## 9. Relationship to CLAY & CLIP

### Why CelebA for This Project?

1. **Structured attributes:** 40 well-defined binary properties enable precise ground-truth definition
2. **Sufficient scale:** 19,962 test images provide robust evaluation statistics
3. **CLIP alignment:** Faces are well-represented in CLIP's vision-language space
4. **Standard benchmark:** CelebA is widely used in face synthesis & attribute retrieval research

### Modality Gap Issue (CLIP-specific)

CLIP ViT-B/32 embeddings live in a **512-dimensional shared space**:
- Image embeddings: extracted from final ViT layer
- Text embeddings: extracted from final CLIP text encoder
- Both **L2-normalized** to unit sphere

**Challenge:** Naive addition `v_ref + t_pos - t_neg` is geometrically crude because:
- Embeddings cluster differently for vision vs. language
- Text "Smiling" is far from some image "Smiling" examples
- This **modality gap** is why Tier-0 baseline has low Recall@1

**Solution strategies:**
- **Tier-1 (CLAY):** Align via rotation H(·) and tangent-space projection
- **Tier-2b (Φ):** Learn dynamic fusion with contrastive loss to bridge the gap

---

## 10. Implementation Checklist

- [ ] Load CelebA test split via torchvision
- [ ] Parse `list_attr_celeba.txt` and build [19962, 40] attribute tensor
- [ ] Load `celeba_evaluation.json` and verify structure
- [ ] **Sanity check:** `celeba[13].filename == "182651.jpg"`
- [ ] Extract CLIP features for all test images; cache to disk
- [ ] Precompute text embeddings for all 40 attributes
- [ ] Implement Tier-0 baseline (naïve latent arithmetic)
- [ ] Implement evaluation harness (Recall@K, Precision@K)
- [ ] Run Tier-0 on all 12 queries; produce metrics table
- [ ] Document baseline results and performance gaps
- [ ] Commit Phase A deliverables (cached features, baseline scores)

---

## 11. References & Resources

- **CelebA Paper:** Liu et al., "Deep Learning Face Attributes in the Wild" (ICCV 2015)
- **CLIP Paper:** Radford et al., "Learning Transferable Models for Computer Vision Tasks" (ICML 2021)
- **CLAY Paper:** Lim et al., "Enhancing CLIP with Dynamic Text-Conditional Fusion" (2026)
- **PyTorch CelebA:** https://pytorch.org/vision/stable/generated/torchvision.datasets.CelebA.html

---

## 12. Quick Reference: Attribute Index Mapping

```python
ATTRIBUTES = [
    'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',
    'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
    'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young', '5_o_Clock_Shadow'
]

# Index lookup:
attr_idx = ATTRIBUTES.index('Smiling')  # 30
attr_name = ATTRIBUTES[30]  # 'Smiling'
```

