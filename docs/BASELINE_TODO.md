# Baseline Experiments TODO

## 1. MC Dropout / Deep Ensemble on WN18RR and YAGO3-10

### Current Status
- ✅ FB15k-237: MC Dropout 0.430, Deep Ensemble 0.225
- ❌ WN18RR: Not run
- ❌ YAGO3-10: Not run

### Colab Notebook Modifications

The notebook `notebooks/colab_baselines.ipynb` is hardcoded for FB15k-237. To run on other datasets:

#### Option A: Parameterize the notebook

```python
# Add at the top of the notebook:
DATASET = "WN18RR"  # or "FB15k-237" or "YAGO3-10"

DATASET_URLS = {
    "FB15k-237": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237",
    "WN18RR": "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr",
    # YAGO3-10 uses different format (OpenKE), needs special handling
}

CAGP_RESULTS = {
    "FB15k-237": {'mean': 0.960, 'std': 0.000},
    "WN18RR": {'mean': 0.871, 'std': 0.003},
    "YAGO3-10": {'mean': 0.942, 'std': 0.000},
}
```

#### Option B: Duplicate notebook for each dataset

Create:
- `notebooks/colab_baselines_wn18rr.ipynb`
- `notebooks/colab_baselines_yago.ipynb`

### YAGO3-10 Special Handling

YAGO3-10 uses OpenKE format (train2id.txt with "h t r" format):

```python
# Download YAGO3-10
base_url = "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/YAGO3-10"

def load_openke_format(url):
    """Load triples from OpenKE format."""
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    n = int(lines[0])  # First line is count
    triples = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 3:
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            triples.append((h, r, t))  # Convert to (h, r, t)
    return triples

train = load_openke_format(f"{base_url}/train2id.txt")
test = load_openke_format(f"{base_url}/test2id.txt")
```

### Expected Runtime

| Dataset | Entities | Training Time (50 epochs) |
|---------|----------|---------------------------|
| WN18RR | 40,943 | ~10 min per model |
| YAGO3-10 | 123,161 | ~30-45 min per model |

With 3 seeds × (1 MC Dropout + 5 Deep Ensemble models) = 18 training runs per dataset.

---

## 2. Exact YAGO3-10 Coverage Statistics

### Current Status
- Paper uses estimated values: p_h=0.71, p_t=0.89, P(both)=0.63, s_r=0.92
- Need exact computation

### How to Compute

Add YAGO3-10 to `scripts/verify_theorem.py`:

```python
# In the datasets dict:
datasets = {
    'WN18RR': os.path.join(project_root, 'data', 'raw', 'wn18rr'),
    'FB15k-237': os.path.join(project_root, 'data', 'raw', 'fb15k-237'),
    'YAGO3-10': os.path.join(project_root, 'data', 'raw', 'yago3-10'),
}

# Add observed AUROC:
observed_auroc = {
    'WN18RR': 0.657,
    'FB15k-237': 0.821,
    'YAGO3-10': 0.760,
}
```

**Note:** YAGO3-10 uses OpenKE format, so `load_triples()` needs modification:

```python
def load_triples_openke(path):
    """Load triples from OpenKE format (h t r per line, first line is count)."""
    triples = []
    with open(path) as f:
        n = int(f.readline().strip())
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                h, t, r = parts[0], parts[2], parts[1]  # OpenKE: h t r
                triples.append((h, r, t))
    return triples
```

### Coverage Stats to Compute

For the appendix table, compute:
- `p_h`: P(head entity covered for its relation | ID test triple)
- `p_t`: P(tail entity covered for its relation | ID test triple)
- `P(both)`: P(both head and tail covered | ID test triple)
- `s_r (avg)`: Average relation sparsity across all relations

```python
def compute_coverage_stats(train_triples, test_triples, num_entities, num_relations):
    """Compute coverage statistics for test triples."""

    # Build coverage matrix from training
    coverage = {}  # (entity, relation) -> bool
    for h, r, t in train_triples:
        coverage[(h, r)] = True
        coverage[(t, r)] = True

    # Compute p_h, p_t, P(both) on test triples
    head_covered = 0
    tail_covered = 0
    both_covered = 0

    for h, r, t in test_triples:
        h_cov = coverage.get((h, r), False)
        t_cov = coverage.get((t, r), False)

        if h_cov:
            head_covered += 1
        if t_cov:
            tail_covered += 1
        if h_cov and t_cov:
            both_covered += 1

    n = len(test_triples)
    p_h = head_covered / n
    p_t = tail_covered / n
    p_both = both_covered / n

    # Compute average relation sparsity
    entities_per_relation = {}
    for h, r, t in train_triples:
        if r not in entities_per_relation:
            entities_per_relation[r] = set()
        entities_per_relation[r].add(h)
        entities_per_relation[r].add(t)

    sparsities = []
    for r, entities in entities_per_relation.items():
        s_r = 1 - len(entities) / num_entities
        sparsities.append(s_r)

    avg_sparsity = sum(sparsities) / len(sparsities)

    return {
        'p_h': p_h,
        'p_t': p_t,
        'P(both)': p_both,
        's_r': avg_sparsity,
    }
```

---

## 3. Update Paper After Experiments

Once results are obtained:

### experiments.tex - Main Table

```latex
MC Dropout & X.XXX & 0.430 & X.XXX \\
Deep Ensemble & X.XXX & 0.225 & X.XXX \\
```

### appendix.tex - Coverage Stats

Replace estimated YAGO values:
```latex
YAGO3-10 & 0.71* & 0.89* & 0.63* & 0.92* \\
```

With exact values (remove asterisks and footnote).

---

## Priority

1. **High**: MC Dropout/Deep Ensemble on WN18RR (fast, completes the table)
2. **Medium**: YAGO coverage stats (can be done locally, quick)
3. **Low**: MC Dropout/Deep Ensemble on YAGO3-10 (slow, but baselines already shown to be poor)

If time is limited, WN18RR baselines + exact YAGO coverage would be sufficient for reviewers.
