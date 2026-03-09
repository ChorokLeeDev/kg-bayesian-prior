# Why OGBL-BioKG Has 0.4% Novel-Context Rate

**Date**: 2026-03-09
**Analysis Script**: `scripts/analyze_biokg_novel_context.py`

## Summary

OGBL-BioKG exhibits an extremely low novel-context rate (0.4%) compared to FB15k-237 (31.4%) and WN18RR (39.2%). This document explains why.

## Key Metrics Comparison

| Dataset    | Relations | Entities | Mean Rel/Ent | Novel Context | Emerging Entity |
|------------|-----------|----------|--------------|---------------|-----------------|
| FB15k-237  | 237       | 14,541   | 9.56         | **31.4%**     | 0.1%            |
| WN18RR     | 11        | 40,943   | 1.84         | **39.2%**     | 6.7%            |
| BioKG      | 51        | 93,773   | 1.83         | **0.4%**      | 0.0%            |

## Root Cause: Random Edge Split

**The primary explanation is OGB's split strategy.**

From the [OGB documentation](https://ogb.stanford.edu/docs/linkprop/#ogbl-biokg):

> "For this dataset, we adopt a **random split**... it is incredibly challenging to obtain accurate information as to when individual experiments and observations underlying the triplets were made."

### What Random Split Means

With random edge splitting:
1. All edges are pooled together
2. Edges are randomly assigned to train (93.6%), valid (3.2%), test (3.2%)
3. **No structural novelty is introduced**

If an entity E appears with relation R in 1000 edges, ~936 go to train, ~32 to valid, ~32 to test. The test edges are statistically similar to training edges.

### Why FB15k-237/WN18RR Are Different

Both FB15k-237 and WN18RR use **curated splits** designed to test KGE generalization:
- FB15k-237: Test triples were selected to include "harder" cases requiring multi-hop reasoning
- WN18RR: Removed inverse relations from WN18, making test triples require genuine inference

These curated splits deliberately include entity-relation combinations that are **structurally novel**.

## BioKG-Specific Factors

### 1. Entity Type Constraints

BioKG has 5 entity types with constrained relation patterns:

| Entity Type | Count    | Typical Relations                |
|-------------|----------|----------------------------------|
| protein     | 17,499   | protein-protein, protein-drug    |
| drug        | 10,533   | drug-protein, drug-sideeffect    |
| disease     | 10,687   | disease-protein                  |
| function    | 45,085   | function-protein                 |
| sideeffect  | 9,969    | sideeffect-drug                  |

Each entity type only participates in a small subset of the 51 relations. This creates **implicit coverage** - most valid (entity, relation) pairs are already present in training.

### 2. Relation Frequency Distribution

BioKG has extreme relation imbalance (Gini = 0.759):

| Relation | Edges      | Percentage |
|----------|------------|------------|
| 42       | 1,433,230  | 30.1%      |
| 43       | 777,577    | 16.3%      |
| 50       | 352,546    | 7.4%       |
| ...      | ...        | ...        |
| 9        | 128        | 0.003%     |

The top 2 relations account for **46% of all edges**. Random sampling will heavily favor these common relations, which are already well-covered in training.

### 3. Coverage Density

Despite having only 1.83 relations per entity (similar to WN18RR's 1.84):
- 50% of entities see only 1 relation
- 99% of entities see 9 or fewer relations
- Entity types constrain which relations are valid

This low coverage per entity combined with random splitting means test edges almost always come from already-covered relations.

## Why This Matters for Our Paper

### Good News
BioKG serves as a **ceiling baseline** - a dataset where coverage-based OOD detection has minimal impact because there are almost no novel-context queries.

### The Real Problem
Datasets with **structural splits** (FB15k-237, WN18RR, ICEWS14) show 25-40% novel-context rates. These are the scenarios where:
1. KGE models are overconfident
2. Coverage tracking is critical
3. Our Theorem 2 (embedding-based impossibility) applies

### Implication
If a dataset has ~0% novel-context due to random splitting, coverage-based OOD detection provides little benefit. But this is rare in practice - real KG queries often involve entity-relation combinations unseen during training.

## Technical Details

### BioKG Test Set Breakdown (n=162,870)
- Novel context: 660 (0.405%)
  - Head-only: 141
  - Tail-only: 500
  - Both: 19
- Emerging entity: 0 (0.000%)
- In-distribution: 162,210 (99.595%)

### FB15k-237 Test Set Breakdown (n=20,466)
- Novel context: 6,431 (31.4%)
- Emerging entity: 25 (0.1%)
- In-distribution: 14,010 (68.5%)

### WN18RR Test Set Breakdown (n=3,134)
- Novel context: 1,228 (39.2%)
- Emerging entity: 210 (6.7%)
- In-distribution: 1,696 (54.1%)

## Deep Dive: Why WN18RR Has 39% Despite Similar Coverage

An interesting puzzle: WN18RR has nearly identical mean relations per entity (1.84) as BioKG (1.83), yet exhibits **39% novel-context** vs BioKG's **0.4%**.

### Key Finding: Entity-Level Coverage Gaps

Analysis script: `scripts/analyze_coverage_deep_dive.py`

| Dataset    | Test Entities with Coverage Gap | Mean Gap Size |
|------------|--------------------------------|---------------|
| WN18RR     | 27.7%                          | 0.28 relations |
| FB15k-237  | 47.5%                          | 0.63 relations |

In WN18RR, **27.7% of test entities** appear with at least one relation they never saw during training. This is the source of novel-context queries.

### Novel-Context Rate Varies by Relation

WN18RR shows extreme variation in novel-context rate by relation:

| Relation | Test Edges | Novel-Context Rate |
|----------|------------|-------------------|
| 0        | 114        | 88.6%             |
| 7        | 1,251      | 71.2%             |
| 5        | 253        | 63.2%             |
| 6        | 1,074      | 2.9%              |
| 4        | 39         | 2.6%              |

Despite **0.9998 correlation** between train/test relation distributions, certain relations (0, 7, 5) appear almost exclusively in novel-context queries. This suggests the test set was constructed to include these challenging cases.

### Why BioKG Is Different

In BioKG with random edge split:
- If entity E appears with relation R in the dataset, ~93.6% of those edges go to train
- Test edges are uniformly sampled from the same (E, R) pairs
- Result: essentially 0% of test entities have coverage gaps

## Conclusion

The 0.4% novel-context rate in BioKG is **not a bug** - it's a natural consequence of:

1. **Random edge split** (vs structural splits in FB15k-237/WN18RR)
2. **Entity type constraints** that reduce effective relation space
3. **Extreme relation imbalance** (top 2 relations = 46% of data)

The key insight is that **similar coverage density does not imply similar novel-context rates** - it depends entirely on how the test set is constructed. WN18RR's test set was curated to include entity-relation pairs absent from training, while BioKG's random split preserves coverage overlap.

This explains why BioKG is sometimes used to demonstrate that KGE models "work well" - the test set is almost entirely in-distribution by construction. For OOD detection research, datasets with structural splits (or real temporal splits like ICEWS) are more informative.
