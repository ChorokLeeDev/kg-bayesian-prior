# Research Pivot: Heterogeneous Graph OOD Detection

## New Thesis

> "Node embeddings in heterogeneous graphs are systematically blind to edge-type coverage. This causes OOD detection methods to fail when nodes appear with novel edge types - a failure mode affecting 11-25% of queries across domains."

## Generalization from KG to Heterogeneous Graphs

| Domain | Node Type | Edge Type | Novel Edge-Type Problem |
|--------|-----------|-----------|------------------------|
| Knowledge Graph | Entity | Relation | (entity, relation) unseen |
| Social Network | User | Interaction | (user, interaction) unseen |
| Citation Network | Paper | Citation type | (paper, cite_type) unseen |
| Molecular Graph | Atom | Bond type | (atom, bond) unseen |
| E-commerce | User/Item | Action | (user, action) unseen |
| Academic | Author/Paper | Collaboration | (author, collab_type) unseen |

## Experimental Results

### Synthetic Heterogeneous Graph (50 edge types)
```
Nodes: 5000, Edge types: 50
Novel edge-type percentage: 50.4%

Results (3 seeds):
  Energy AUROC:   0.625 ± 0.005  → FAILS
  Coverage AUROC: 1.000 ± 0.000  → WORKS
```
**✓ CONFIRMED**: Coverage blind spot exists when many edge types

### Real PyG DBLP (4 edge types)
```
Node types: author, paper, term, venue
Edge types: writes, has_term, published_at, cites

Results (3 seeds):
  Novel edge-type %: 0.3%  ← VERY LOW!
  Coverage AUROC: 0.511 ± 0.011  ← RANDOM
```
**✗ NOT APPLICABLE**: Too few edge types per node type

### Real PyG IMDB (4 edge types)
```
Node types: movie, actor, director
Edge types: acted_in, directed

Results (3 seeds):
  Novel edge-type %: 5.3%  ← MODERATE
  Coverage AUROC: 0.925 ± 0.003  ← WORKS!
```
**✓ PARTIAL**: Some movies don't appear with all edge types

---

## CRITICAL FINDING

The coverage blind spot requires:
1. **Many edge types (relations)**
2. **Nodes not seeing all edge types**

| Graph Type | Edge Types per Node | Novel Edge-Type % | Coverage AUROC | Blind Spot? |
|------------|---------------------|-------------------|----------------|-------------|
| KG (FB15k-237) | 237 total, ~5% per entity | 28% | 1.00 | ✓ YES |
| KG (ICEWS14) | 222 total, ~2% per entity | 24% | 0.99 | ✓ YES |
| Synthetic Hetero (50) | 50 total, ~30% per node | 50% | 1.00 | ✓ YES |
| PyG IMDB | 4 total, varied per movie | 5.3% | 0.93 | ✓ PARTIAL |
| PyG DBLP | 4 total, ~100% per node | 0.3% | 0.51 | ✗ NO |

**CONCLUSION**: The blind spot is specific to **relation-rich** graphs (KGs),
not general heterogeneous graphs.

---

## Revised Thesis Direction

The generalization to "all heterogeneous graphs" does NOT work.

The blind spot is specific to:
- Knowledge Graphs (100+ relations)
- Graphs where nodes see only a small fraction of edge types

This is STILL a significant finding because KGs are widely used:
- Drug discovery
- Recommendation systems
- Question answering (RAG)
- Enterprise knowledge management

But we should NOT claim it generalizes to DBLP-style hetero graphs.

---

## Experimental Plan

### Phase 1: Validate on Existing KG Results (DONE)
- [x] FB15k-237: Coverage blind spot confirmed
- [x] WN18RR: Coverage blind spot confirmed
- [x] ICEWS14/18: Coverage blind spot confirmed
- [x] GNNSafe anti-prediction on FB15k-237

### Phase 2: Social Network Experiments (TODO)
- [ ] Dataset: OGB-MAG (Microsoft Academic Graph)
- [ ] Dataset: Reddit hyperlink network
- [ ] Dataset: Amazon co-purchase
- [ ] Metric: Novel edge-type AUROC

### Phase 3: Citation Network Experiments (TODO)
- [ ] Dataset: Cora/Citeseer with edge types
- [ ] Dataset: OGB-Citation2
- [ ] Metric: Novel citation-type AUROC

### Phase 4: Molecular Graph Experiments (TODO)
- [ ] Dataset: OGB-MolHIV
- [ ] Dataset: QM9
- [ ] Metric: Novel bond-type AUROC

### Phase 5: Theoretical Generalization (TODO)
- [ ] Generalize impossibility theorem to heterogeneous GNNs
- [ ] Prove coverage blind spot for any edge-type-agnostic embedding
- [ ] Information-theoretic decomposition for heterogeneous graphs

## Datasets to Use

### OGB Heterogeneous Benchmarks
1. **ogbn-mag**: Microsoft Academic Graph
   - Nodes: Papers, Authors, Institutions, Fields
   - Edges: writes, affiliated, cites, has_topic
   - Task: Node classification
   - Can create OOD split by edge type

2. **ogbl-citation2**: Citation network
   - Directed citation links
   - Can add citation type metadata

### Other Heterogeneous Datasets
3. **IMDB**: Movie database
   - Nodes: Movies, Actors, Directors
   - Edges: acted_in, directed

4. **DBLP**: Academic network
   - Nodes: Authors, Papers, Venues
   - Edges: writes, published_in, cites

5. **Freebase (FB15k-237)**: Already done

6. **WordNet (WN18RR)**: Already done

## Methods to Test

1. **Baselines (from KG work)**
   - Variational embeddings (entity variance)
   - Deep Ensembles
   - MC Dropout
   - Energy scoring
   - GNNSafe

2. **Heterogeneous GNN Methods**
   - HAN (Heterogeneous Attention Network)
   - HGT (Heterogeneous Graph Transformer)
   - R-GCN (Relational GCN)
   - CompGCN

3. **Our Method**
   - Edge-type coverage tracking (generalization of relation coverage)
   - Combined semantic + structural uncertainty

## Expected Findings

1. **All embedding-based methods fail on novel edge types**
   - Same pattern as KG: high-freq nodes with unseen edge types
   - AUROC ~ 0.5 on novel edge-type detection

2. **Coverage tracking works universally**
   - Simple (node, edge_type) hash table
   - AUROC ~ 1.0 on novel edge-type detection

3. **The blind spot is prevalent**
   - 10-30% of test edges involve novel (node, edge_type) pairs
   - Hidden by aggregate metrics

## Timeline

- **Week 1-2**: Set up OGB datasets, implement edge-type coverage
- **Week 3-4**: Run experiments on ogbn-mag
- **Week 5-6**: Run experiments on citation/social networks
- **Week 7-8**: Molecular graph experiments
- **Week 9-10**: Theoretical generalization
- **Week 11-12**: Paper writing

## Success Criteria

1. **Same pattern across 3+ domains**: KG, Social, Citation
2. **Coverage blind spot prevalence**: 10-25% in each domain
3. **Baseline failure**: AUROC < 0.6 on novel edge types
4. **Coverage solution**: AUROC > 0.95

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Pattern doesn't generalize | Medium | Start with most similar (citation) |
| Datasets don't have edge types | Low | OGB-MAG has multiple edge types |
| Computation too expensive | Low | Use smaller subsets first |
| Time constraints | Medium | Prioritize 2-3 key datasets |

## Files

- Experiment scripts: `scripts/heterogeneous/`
- Results: `outputs/heterogeneous/`
- Paper draft: `paper/sections/heterogeneous_extension.tex`
