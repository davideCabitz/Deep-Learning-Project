# Baseline Establishment Guide

## Overview
This guide explains how to establish the baseline for the Deep Learning Project using CLIP ViT-B/32 on CelebA dataset, following the project roadmap's **Milestone M1**.

---

## 1. CelebA Dataset Structure

The CelebA dataset is organized as follows:

### Dataset Files (location: `c:\Users\alfon\Downloads\celeba\celeba\`)

| File | Content | Size |
|------|---------|------|
| **img_align_celeba/** | 202,599 aligned face images (000001.jpg → 202599.jpg) | ~1.3 GB |
| **list_attr_celeba.txt** | 40 binary facial attributes per image | CSV-like format |
| **list_eval_partition.txt** | Split assignment: 0=train, 1=val, 2=test | 202,599 rows |
| **list_bbox_celeba.txt** | Bounding boxes (x, y, width, height) for face crops | |
| **list_landmarks_celeba.txt** | 5 landmark points per face (left/right eye, nose, mouth corners) | |
| **identity_CelebA.txt** | Identity labels for grouping same person across images | |

### 40 Attributes
```
5_o_Clock_Shadow, Arched_Eyebrows, Attractive, Bags_Under_Eyes, Bald, Bangs, 
Big_Lips, Big_Nose, Black_Hair, Blond_Hair, Blurry, Brown_Hair, Bushy_Eyebrows, 
Chubby, Double_Chin, Eyeglasses, Goatee, Gray_Hair, Heavy_Makeup, High_Cheekbones, 
Male, Mouth_Slightly_Open, Mustache, Narrow_Eyes, No_Beard, Oval_Face, Pale_Skin, 
Pointy_Nose, Receding_Hairline, Rosy_Cheeks, Sideburns, Smiling, Straight_Hair, 
Wavy_Hair, Wearing_Earrings, Wearing_Hat, Wearing_Lipstick, Wearing_Necklace, 
Wearing_Necktie, Young
```

### Attribute Format
- **Binary values:** `-1` (absent) or `1` (present)
- **Indexing:** Row 1 = filename header, rows 3-202601 = images (000001.jpg to 202599.jpg)
- **Critical:** Dataset indices (0-202598) ≠ filenames. Always use `celeba[idx]` for PyTorch indexing.

---

## 2. Evaluation Setup

### celeba_evaluation.json Structure
```json
[
  {
    "query": "+Smiling",  // Simple positive condition
    "ground_truth": {
      "13": [325, 456, 579, ...],  // source_idx: [target_idx1, target_idx2, ...]
      "14": [235, 1787, 2806, ...],
      ...
    }
  },
  {
    "query": "+Smiling, +Blond_Hair, -Bald",  // Composed query
    "ground_truth": { ... }
  }
]
```

### Ground Truth Validation Rule
A target image is valid if:
1. **Strict constraint satisfaction:** target has ALL `+` attributes and NONE of the `−` attributes
2. **Identity preservation:** Hamming distance on OTHER attributes ≤ 2 (allows minor variation)
3. **Minimum threshold:** source must have ≥ 5 valid targets (filtered in JSON)

### 12 Mandatory Queries
- **7 simple queries:** Single `+` attribute (e.g., `+Smiling`, `+Male`)
- **5 composed queries:** Multiple `+` and `−` constraints (e.g., `+Smiling, +Blond_Hair, -Male`)

---

## 3. Step-by-Step Baseline Workflow

### Phase A: Data Preparation (Days 1–3, M1 Milestone)

#### Step 1: Load CelebA Dataset
```python
import torch
from torchvision import datasets, transforms
import json

# Load CelebA test split only
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # CLIP ViT-B/32 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711])
])

celeba = datasets.CelebA(
    root='c:\\Users\\alfon\\Downloads\\celeba\\',
    split='test',  # Only test split as per spec
    download=False,
    transform=transform
)

print(f"CelebA test set size: {len(celeba)}")  # Should be ~19,962 images
```

#### Step 2: Build Attribute Tensor
```python
# Read list_attr_celeba.txt
attributes = []
with open('c:\\Users\\alfon\\Downloads\\celeba\\celeba\\list_attr_celeba.txt', 'r') as f:
    # Skip header lines (line 0: count, line 1: attribute names)
    header_attrs = f.readline().strip().split()  # 40 attribute names
    f.readline()  # Skip blank line
    
    for line in f:
        parts = line.strip().split()
        filename = parts[0]
        attrs = [int(x) for x in parts[1:]]  # Convert to int (-1 or 1)
        attributes.append(attrs)

# Create mapping for test split only
celeba_attrs = torch.tensor(attributes, dtype=torch.int8)
print(f"Attribute tensor shape: {celeba_attrs.shape}")  # [19962, 40]
```

#### Step 3: Load Evaluation JSON
```python
with open('c:\\Users\\alfon\\Desktop\\Deep-Learning-Project\\Evaluation\\celeba_evaluation.json', 'r') as f:
    queries = json.load(f)

print(f"Total queries: {len(queries)}")
print(f"Query 0: {queries[0]['query']}")
print(f"Sources in Query 0: {len(queries[0]['ground_truth'])} (each with ≥5 targets)")
```

#### Step 4: Sanity Check (CRITICAL)
```python
# Verify dataset-index ↔ filename consistency
test_idx = 13  # From ROADMAP
target_filename = celeba.filename[test_idx]
print(f"celeba[{test_idx}].filename = {target_filename}")  # Should be "182651.jpg"

# Verify attribute tensor indexing
attr_row = celeba_attrs[test_idx]
print(f"Attributes for index {test_idx}: {attr_row}")

# Confirm JSON ground truth structure
if str(test_idx) in queries[0]['ground_truth']:
    targets = queries[0]['ground_truth'][str(test_idx)]
    print(f"Sample source {test_idx} has {len(targets)} valid targets for '{queries[0]['query']}'")
```

---

### Phase B: CLIP Feature Extraction (Days 2–3)

#### Step 5: Load CLIP Model
```python
from transformers import CLIPProcessor, CLIPModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)
model.eval()

print(f"CLIP loaded on {device}")
print(f"Image embedding dim: {model.config.projection_dim}")  # 512
```

#### Step 6: Extract & Cache Image Features
```python
batch_size = 64
image_features_list = []

celeba_loader = torch.utils.data.DataLoader(celeba, batch_size=batch_size, shuffle=False, num_workers=4)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(celeba_loader):
        images = images.to(device)
        # Get image embeddings (normalized)
        outputs = model.get_image_features(images)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)  # L2 normalize
        image_features_list.append(outputs.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size} / {len(celeba)} images")

# Concatenate all batches
image_features = torch.cat(image_features_list, dim=0)
print(f"Image features shape: {image_features.shape}")  # [19962, 512]

# Cache to disk
torch.save(image_features, 'clip_image_features_test.pt')
print("Image features cached to clip_image_features_test.pt")
```

#### Step 7: Build Text Prompt Templates
```python
# Simple attribute-to-text prompt mapping
ATTRIBUTE_NAMES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

# Simple prompt template (can be enhanced)
def attribute_to_prompt(attr_name):
    return f"A face with {attr_name.replace('_', ' ').lower()}"

# Alternative: more robust templates
PROMPT_TEMPLATES = {
    'Smiling': "A smiling face",
    'Male': "A male face",
    'Blond_Hair': "A person with blond hair",
    'Black_Hair': "A person with black hair",
    # ... (expand for all 40 attributes)
}
```

#### Step 8: Precompute Text Features
```python
# Compute text embeddings for all attributes
text_features_dict = {}

with torch.no_grad():
    for attr_name in ATTRIBUTE_NAMES:
        prompt = PROMPT_TEMPLATES.get(attr_name, attribute_to_prompt(attr_name))
        inputs = processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        text_output = model.get_text_features(**inputs)
        text_output = text_output / text_output.norm(dim=-1, keepdim=True)  # L2 normalize
        text_features_dict[attr_name] = text_output.cpu()

print(f"Text features computed for {len(text_features_dict)} attributes")
```

---

### Phase C: Tier-0 Baseline Implementation (Days 2–3)

#### Step 9: Implement Tier-0 Baseline
```python
def tier0_baseline(v_ref_idx, positive_attrs, negative_attrs, image_features, 
                   text_features_dict, attribute_names):
    """
    Vanilla zero-shot baseline: q = normalize(v_ref + α·Σt+ − β·Σt−)
    
    Args:
        v_ref_idx: Reference image index
        positive_attrs: List of attribute names (must present)
        negative_attrs: List of attribute names (must absent)
        image_features: [N, 512] cached image embeddings
        text_features_dict: {attr_name: [1, 512]} text embeddings
        attribute_names: List of all 40 attribute names
    
    Returns:
        rankings: [N] scores (higher = more similar)
    """
    device = image_features.device
    
    # Get reference image embedding
    v_ref = image_features[v_ref_idx]  # [512]
    
    # Aggregate positive and negative text embeddings
    alpha = 1.0  # Weight for positive conditions
    beta = 1.0   # Weight for negative conditions
    
    q = v_ref.clone()  # Start with reference
    
    for attr in positive_attrs:
        if attr in text_features_dict:
            t = text_features_dict[attr].squeeze()  # [512]
            q = q + alpha * t
    
    for attr in negative_attrs:
        if attr in text_features_dict:
            t = text_features_dict[attr].squeeze()  # [512]
            q = q - beta * t
    
    # L2 normalize
    q = q / q.norm(dim=-1, keepdim=True)
    
    # Score all images via cosine similarity
    scores = torch.nn.functional.cosine_similarity(q.unsqueeze(0), image_features)
    
    return scores

# Example query execution
query_text = "+Smiling"
positive_attrs = ["Smiling"]
negative_attrs = []

# Pick a source from ground_truth
source_idx = 13
scores = tier0_baseline(source_idx, positive_attrs, negative_attrs, 
                       image_features, text_features_dict, ATTRIBUTE_NAMES)

# Get top-K retrievals
top_k_indices = torch.argsort(scores, descending=True)[:10]
print(f"Top-10 retrievals for source {source_idx} with query '{query_text}':")
for rank, idx in enumerate(top_k_indices, 1):
    print(f"  {rank}. Index {idx}, score {scores[idx]:.4f}")
```

---

### Phase D: Evaluation Harness (Days 2–3)

#### Step 10: Implement Recall@K and Precision@K
```python
def evaluate_query(query_dict, method_fn, image_features, text_features_dict, 
                   attributes_tensor, attribute_names, k_values=[1, 5, 10]):
    """
    Evaluate one query across all valid sources.
    
    Args:
        query_dict: {"query": "+Smiling", "ground_truth": {source_idx: [target_indices]}}
        method_fn: Callable(source_idx, pos_attrs, neg_attrs) → scores
        k_values: List of K for Recall@K and Precision@K
    
    Returns:
        metrics: {"recall@1": ..., "precision@1": ..., "recall@5": ..., ...}
    """
    query_str = query_dict['query']
    ground_truth = query_dict['ground_truth']
    
    # Parse query string: "+Attr1, +Attr2, -Attr3, -Attr4"
    positive_attrs = []
    negative_attrs = []
    
    for part in query_str.split(','):
        part = part.strip()
        if part.startswith('+'):
            positive_attrs.append(part[1:])
        elif part.startswith('-'):
            negative_attrs.append(part[1:])
    
    # Collect metrics across valid sources
    recalls_by_k = {k: [] for k in k_values}
    precisions_by_k = {k: [] for k in k_values}
    
    for source_idx_str, valid_targets in ground_truth.items():
        source_idx = int(source_idx_str)
        valid_targets_set = set(valid_targets)
        
        # Get retrieval scores
        scores = method_fn(source_idx, positive_attrs, negative_attrs)
        
        # Exclude self
        scores_copy = scores.clone()
        scores_copy[source_idx] = -float('inf')
        
        # Top-K rankings
        for k in k_values:
            top_k_indices = torch.argsort(scores_copy, descending=True)[:k].tolist()
            top_k_set = set(top_k_indices)
            
            # Recall@K: what fraction of ground truth did we retrieve?
            hits = len(valid_targets_set & top_k_set)
            recall_k = hits / len(valid_targets_set) if valid_targets_set else 0
            recalls_by_k[k].append(recall_k)
            
            # Precision@K: what fraction of top-K were correct?
            precision_k = hits / k
            precisions_by_k[k].append(precision_k)
    
    # Average across sources
    metrics = {}
    for k in k_values:
        metrics[f'recall@{k}'] = sum(recalls_by_k[k]) / len(recalls_by_k[k]) if recalls_by_k[k] else 0
        metrics[f'precision@{k}'] = sum(precisions_by_k[k]) / len(precisions_by_k[k]) if precisions_by_k[k] else 0
    
    return metrics

# Run evaluation
all_metrics = []
for query_dict in queries:
    metrics = evaluate_query(
        query_dict,
        lambda idx, pos, neg: tier0_baseline(idx, pos, neg, image_features, 
                                            text_features_dict, ATTRIBUTE_NAMES),
        image_features,
        text_features_dict,
        celeba_attrs,
        ATTRIBUTE_NAMES
    )
    print(f"Query '{query_dict['query']}': {metrics}")
    all_metrics.append(metrics)

# Summary table
print("\n=== Tier-0 Baseline Results ===")
print(f"{'Query':<30} {'Recall@1':<10} {'Recall@5':<10} {'Recall@10':<10}")
print("=" * 60)
for query_dict, metrics in zip(queries, all_metrics):
    print(f"{query_dict['query']:<30} {metrics['recall@1']:.4f}    {metrics['recall@5']:.4f}    {metrics['recall@10']:.4f}")
```

---

## 4. Expected Baseline Results

For **Tier-0** (vanilla latent arithmetic):
- **Simple queries (+X only):** Recall@1 typically **15–35%** (exploits CLIP alignment)
- **Composed queries (+X, +Y, -Z):** Recall@1 typically **5–20%** (naive text stacking is weak)

This establishes the **floor** that Tier-1 (CLAY) and Tier-2 (our contribution) must beat.

---

## 5. Critical Implementation Notes

| Item | Importance | Note |
|------|-----------|------|
| **Dataset indexing** | 🔴 CRITICAL | Always use `celeba[idx]`, never filename-based loading |
| **L2 normalization** | 🔴 CRITICAL | CLIP embeddings must be normalized for cosine similarity |
| **Test split only** | 🔴 CRITICAL | Never evaluate on train/val splits; reserve train for Φ training |
| **Self-exclusion** | 🟡 HIGH | Exclude source image from retrieval ranking (don't rank itself) |
| **Ground truth validation** | 🟡 HIGH | JSON is authoritative; sources are pre-filtered (≥5 valid targets) |
| **Attribute names** | 🟡 HIGH | Use exact names from `list_attr_celeba.txt` (underscores matter) |
| **Prompt engineering** | 🟢 MEDIUM | Simple templates sufficient for baseline; refinement for Tier-1/2 |

---

## 6. Files to Produce (Phase A Deliverables)

By **end of Day 3 (Milestone M1)**, generate:

1. **`clip_image_features_test.pt`** — Cached [19962, 512] image embeddings
2. **`celeba_attributes_test.pt`** — Cached [19962, 40] attribute binary tensor
3. **`tier0_baseline_results.json`** — All 12 queries × K∈{1,5,10} metrics
4. **Evaluation harness notebook cells** — Fully working Recall/Precision computation
5. **Sanity checks passing** — Index verification, shape validation, non-trivial retrieval scores

---

## Next Steps (Phase B: Tier-1 & Tier-2)

Once M1 is complete, proceed to:
- **Tier-1:** CLAY reproduction (manifold-aware tangent space, SVD, rotation H)
- **Tier-2a:** Training-free negative rejection (orthogonal complement)
- **Tier-2b:** Trained fusion module Φ with contrastive loss

