# MixRec × SYNERISE — Heterogeneous Graph Collaborative Filtering

> A PyTorch implementation of [MixRec (WSDM 2025)](https://arxiv.org/abs/2412.13825) applied to the [SYNERISE ACM RecSys 2025 Challenge dataset](https://recsys.synerise.com), demonstrating multi-behavior recommendation on a large-scale real-world e-commerce dataset.

---

## Overview

Traditional recommender systems treat all user–item interactions as the same type of signal. **MixRec** addresses this by modeling multiple behavior types (e.g. browsing, adding to cart, purchasing) as a heterogeneous graph, and learning intent-specific user preferences through a parameterized hypergraph neural network.

This project applies MixRec to the SYNERISE dataset — a large-scale, timestamped e-commerce dataset from the ACM RecSys 2025 Challenge — which was **not used in the original paper**, demonstrating the model's transferability to unseen real-world data.

---

## Results

| Metric | Score |
|--------|-------|
| HR@10 | 0.7963 |
| NDCG@10 | 0.5879 |
| Best epoch | 6 (of 21 run) |

Evaluation protocol: 100-item sampling (1 true positive + 99 random negatives), leave-one-out split on purchase behavior.

---

## Dataset

**SYNERISE** — released as part of the ACM RecSys 2025 Challenge.

| File | Used | Role |
|------|------|------|
| `add_to_cart.parquet` | ✓ | Cart behavior (auxiliary signal) |
| `product_buy.parquet` | ✓ | Buy behavior (prediction target) |
| `page_visit.parquet` | ✗ | Dropped — URL space ≠ SKU space |
| `remove_from_cart.parquet` | ✗ | Not used |
| `search_query.parquet` | ✗ | Not used |
| `product_properties.parquet` | ✗ | Reserved for Phase 3 extension |

**Dataset statistics after preprocessing:**

| | Count |
|---|---|
| Users | 1,059,121 |
| Items | 691,084 |
| Cart interactions (train) | 3,911,851 |
| Buy interactions (train) | 1,237,422 |
| Test users | 295,088 |

Download the dataset at [recsys.synerise.com](https://recsys.synerise.com).

---

## Model Architecture

MixRec consists of three main components:

```
1. Embedding Layer
   └── User embeddings  (n_users × latdim)
   └── Item embeddings  (n_items × latdim)

2. Heterogeneous Hypergraph GNN
   └── For each behavior (Cart, Buy):
       └── Nodes vote for hyperedges via learned weight matrix W
       └── Hyperedges aggregate member embeddings (intent prototypes)
       └── Nodes update from their hyperedges
       └── Stacked gnn_layer times
   └── Behavior fusion: learned weighted average across behaviors

3. Training Objectives
   └── BPR loss       — score(positive item) > score(negative item)
   └── Contrastive SSL — align same user's embeddings across behaviors (InfoNCE)
   └── L2 regularization
```

> **Note on implementation:** This is a PyTorch port of the original TensorFlow 1.x codebase. The core architecture (heterogeneous hypergraph GNN, multi-behavior modeling, relation-wise contrastive learning, BPR training) is faithfully preserved. Minor differences include a simplified hypergraph convolution formulation and omission of the global SSL objective, due to the constraints of porting from TF1.

---

## Project Structure

```
.
├── notebooks/
│   ├── 01_preprocessing.ipynb   # SYNERISE → .pkl sparse matrices
│   └── 02_mixrec_model.ipynb    # PyTorch model, training, evaluation, tuning
├── Datasets/
│   └── synerise/
│       ├── trn_Cart.pkl         # scipy CSR sparse matrix (cart, train)
│       ├── trn_Buy.pkl          # scipy CSR sparse matrix (buy, train)
│       ├── tst_Buy.pkl          # dict {user_id: [item_id]} (test)
│       ├── stats.json           # {n_users, n_items}
│       ├── user2id.pkl          # raw ID → integer mapping
│       └── item2id.pkl          # raw ID → integer mapping
└── README.md
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/mixrec-synerise.git
cd mixrec-synerise
```

### 2. Install dependencies

```bash
pip install torch numpy scipy pandas matplotlib tqdm
```

### 3. Download the SYNERISE dataset

Download the `.parquet` files from [recsys.synerise.com](https://recsys.synerise.com) and place them in a `raw/` folder.

### 4. Run preprocessing

Open and run `notebooks/01_preprocessing.ipynb`. This produces the `.pkl` files in `Datasets/synerise/`.

Update `DATA_DIR` in the notebook to point to your `raw/` folder:

```python
DATA_DIR = 'raw/'
OUT_DIR  = 'Datasets/synerise/'
```

### 5. Train the model

Open and run `notebooks/02_mixrec_model.ipynb`. Update `data_path` in Cell 2:

```python
data_path = 'Datasets/synerise'
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latdim` | 64 | Embedding size |
| `gnn_layer` | 2 | GNN propagation layers |
| `hyperNum` | 128 | Number of hyperedges (intent prototypes) |
| `lr` | 1e-3 | Learning rate |
| `reg` | 1e-4 | L2 weight decay |
| `batch` | 512 | Batch size |
| `ssl_reg` | 1e-6 | Contrastive SSL loss weight |
| `keepRate` | 0.5 | Dropout keep rate |
| `graphSampleN` | 5000 | Subgraph nodes sampled per training step |
| `epoch` | 30 | Max training epochs |
| `patience` | 5 | Early stopping patience (eval rounds) |
| `shoot` | 10 | K for top-K evaluation |

The notebook includes a full hyperparameter sweep (Section 6) over `latdim`, `lr`, and `ssl_reg`.

---

## Evaluation Metrics

**HR@10 (Hit Rate at 10)**
The fraction of test users for whom the true held-out purchase appears in the model's top-10 recommendations. Higher is better; maximum is 1.0.

**NDCG@10 (Normalized Discounted Cumulative Gain at 10)**
Like HR@10 but rewards higher rankings more. Finding the true item at rank 1 contributes more than finding it at rank 10. Computed as `1 / log₂(rank + 1)`, averaged across users.

Both metrics use **100-item sampling**: for each test user, the model ranks 1 true positive item against 99 randomly sampled negative items.

---

## Preprocessing Pipeline

```
Raw SYNERISE parquet files
        ↓
Select 2 behaviors (add_to_cart, product_buy)
        ↓
Deduplicate (user, item) pairs
        ↓
K-core filtering (min 2 interactions per user/item)
        ↓
Integer ID remapping (0 … N-1)
        ↓
Leave-one-out train/test split on purchases
        ↓
Save as scipy CSR sparse matrices (.pkl)
```

---

## Environment

| | |
|---|---|
| Platform | Kaggle (GPU T4 × 1) |
| Python | 3.12 |
| PyTorch | 2.x |
| Training time | ~1–2 hours (30 epochs) |

---

## Paper Reference

```bibtex
@inproceedings{mixrec2025,
  title     = {MixRec: Heterogeneous Graph Collaborative Filtering},
  booktitle = {Proceedings of the 18th ACM International Conference on Web Search and Data Mining (WSDM)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2412.13825}
}
```

Original codebase: [github.com/HKUDS/MixRec](https://github.com/HKUDS/MixRec)

---

## Acknowledgements

- MixRec paper and original TF1 implementation by the HKUDS lab
- SYNERISE dataset provided by Synerise for the ACM RecSys 2025 Challenge
- University of Jeddah — CCAI 422 Recommender Systems course project
