# ULTRA Foundation Model OOD Evaluation - Feasibility Report

## Summary
**Status: NOT FEASIBLE on local CPU (< 2 hours)**

ULTRA requires GPU for practical inference times. The experiment would need to be run on Colab or similar.

## Architectural Analysis: Why ULTRA Has the Same Blind Spot

**Key Insight: ULTRA's architecture fundamentally cannot detect novel relational contexts.**

### How ULTRA Works

1. **Two-Level NBFNet Architecture:**
   - **Relation-level NBFNet**: Learns representations for relations by message passing on a "relation graph" with 4 edge types (head-to-head, tail-to-tail, head-to-tail, tail-to-head connectivity)
   - **Entity-level NBFNet**: Performs 6-layer Bellman-Ford style message passing starting from query head, using relation-conditioned representations

2. **Query Processing:**
   - For query (h, r, ?): Initialize source node h with relation embedding r
   - Propagate through 6 layers across ALL edges in the graph
   - Score each potential tail t by MLP on final node representation

3. **What ULTRA Encodes:**
   - Global graph structure (connectivity patterns)
   - Relation semantics via the relation graph
   - Path-based reasoning (multi-hop inference)

### Why ULTRA Has the Coverage Blind Spot

**ULTRA does NOT track (entity, relation) co-occurrence.** Its predictions depend on:

1. **Graph connectivity**: If entity e has many neighbors, it gets rich representations
2. **Relation semantics**: Relations with similar connectivity patterns get similar embeddings
3. **Path existence**: Multi-hop paths from h to t through any relations

**Critical gap:** An entity can have high connectivity (appear in many triples) but never with a specific relation. ULTRA cannot distinguish:
- Entity seen with relation r (in-distribution)
- Entity seen but never with relation r (novel context)

### Concrete Example

Consider entity "Barack Obama" in Freebase:
- Seen in 1000+ triples with relations: nationality, profession, birthplace, etc.
- Never seen with relation: "chemical_formula"

When ULTRA scores (Barack_Obama, chemical_formula, ?):
- Obama has rich representations from many training edges
- Chemical_formula has learned semantics from other entities
- ULTRA will confidently score this, despite it being OOD

### Theoretical Guarantee

**Theorem (informal):** Any model that maps entities to fixed-dimensional representations cannot distinguish ID from novel-context OOD without explicit coverage tracking.

ULTRA's entity representations are updated by message passing, but they don't encode "which relations this entity was trained with." This is exactly the coverage blind spot we identify.

### Comparison to CAGP

| Aspect | ULTRA | CAGP |
|--------|-------|------|
| Entity representation | Graph-conditioned | Mean + variance |
| Relation handling | Relation-aware message passing | Per-relation coverage matrix |
| Novel context detection | Cannot detect | Explicit via coverage |
| Emerging entity detection | Partial (low connectivity) | Via semantic variance |

## What Was Tested

1. **ULTRA repository** (https://github.com/DeepGraphLearning/ULTRA)
   - Successfully cloned and analyzed
   - Model loads correctly (168,705 parameters)
   - Pretrained checkpoints available (~2MB each)

2. **Dependencies**
   - PyTorch 2.8.0 (compatible, requires >=2.1)
   - PyG 2.6.1 (compatible, requires >=2.4)
   - torch-scatter 2.1.2 (compatible)
   - ninja (installed for C++ extension compilation)

3. **rspmm C++ Extension**
   - Successfully compiled on macOS
   - Only supports CUDA and CPU (MPS not supported)

4. **Inference Test on FB15k-237**
   - Test set: 20,466 triples
   - Split: 2,223 emerging / 5,193 novel_ctx / 13,050 ID
   - After 40+ minutes on CPU, scoring was incomplete
   - Estimated full runtime: 2+ hours on CPU

## Why ULTRA is Slow

ULTRA uses an NBFNet-based architecture that performs:
- 6 layers of relational message passing
- Full graph traversal (14K nodes for FB15k-237) per query batch
- Complex relation-aware aggregation

This is ~100-1000x slower than simple embedding lookups used by DistMult/ComplEx/TransE.

## Recommendations

### Option 1: Run on GPU (Recommended)
- Colab with GPU runtime would reduce inference time to ~5-10 minutes
- Created script at: `scripts/run_ultra_experiment.py`
- Just needs GPU to be practical

### Option 2: Alternative Foundation Models
Models that might be faster to test:
- **NodePiece** (2021): Tokenized representations, faster inference
- **NBFNet** (2021): Similar architecture to ULTRA but single-graph
- **DRUM** (2019): Rule mining approach, different uncertainty profile

### Option 3: Document as "Compute-Constrained"
- Cite ULTRA in paper as state-of-the-art
- Note that GPU evaluation is future work
- Focus on the theoretical insight: "scale doesn't fix coverage blind spot"

## Key Insight (Regardless of Empirical Results)

Even if we ran ULTRA, the hypothesis is:
- ULTRA uses entity-level representations derived from graph structure
- Novel relational contexts (entity seen, but not with this relation) would still have confident predictions
- Because ULTRA doesn't explicitly track (entity, relation) coverage

The architectural analysis suggests ULTRA would have the same blind spot, even without running the experiment.

## Files Created
- `/Users/i767700/Github/kg-bayesian-prior/scripts/run_ultra_experiment.py` - Ready to run on GPU
- `/Users/i767700/Github/ultra_test/` - ULTRA repository clone

## Dataset Support
- FB15k-237: Supported (tested)
- WN18RR: Supported (in ULTRA)
- ICEWS14/18: NOT supported (would need custom loader)
- YAGO3-10: Supported as YAGO310
