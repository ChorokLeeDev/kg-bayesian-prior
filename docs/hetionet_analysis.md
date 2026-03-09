# Hetionet Coverage Blind Spot Analysis

## Dataset Summary

**Hetionet v1.0** is a biomedical knowledge graph for drug repurposing research:
- 45,158 entities across 11 types
- 2,250,197 edges across 24 relation types
- Entity types: Gene, Disease, Compound, Biological Process, Side Effect, etc.

## Key Finding 1: Relation-Specific Blind Spots (Random Split)

While the **overall novel-context rate is low (1.7%)**, specific relation types show dramatically higher rates:

| Relation | Novel Context Rate | Sample Size | Description |
|----------|-------------------|-------------|-------------|
| DdG | **61.7%** | 798 | Disease-downregulates-Gene |
| DuG | **58.6%** | 778 | Disease-upregulates-Gene |
| DaG | **30.4%** | 1,225 | Disease-associates-Gene |
| CbG | 8.5% | 1,112 | Compound-binds-Gene |
| CuG | 8.2% | 1,922 | Compound-upregulates-Gene |

## Key Finding 2: Inductive Split (Drug Repurposing Scenario)

When simulating a drug repurposing scenario (80% diseases for training, 20% held-out):

| Category | Count | Percentage |
|----------|-------|------------|
| Novel context | 5,983 | **64.9%** |
| Emerging entity | 3,138 | **34.0%** |
| In-distribution | 101 | 1.1% |
| **TOTAL OOD** | **9,121** | **98.9%** |

**This means 99% of drug repurposing predictions for new diseases would involve coverage blind spots!**

## Interpretation

The low overall rate in random split is due to **hub genes** - highly connected entities that participate in many relation types. These genes dominate the test set by frequency.

However:
1. **Disease-gene interactions** (DdG, DuG, DaG) show 30-62% novel-context rates in random split
2. **Inductive settings** (new diseases) show 99% OOD rate

This is critical because:
- These relations are **drug discovery targets** - identifying disease-gene associations is essential for finding therapeutic targets
- Models will be **overconfident** - a disease entity may have many pathway/function edges, making its embedding appear confident, but have ZERO evidence for gene regulation
- **Safety implications** - confidently predicting drug-disease-gene pathways without evidence could mislead therapeutic development

## Comparison to Other Datasets

| Dataset | Overall Novel Context | Critical Relations |
|---------|----------------------|-------------------|
| FB15k-237 | 25% | - |
| OGBL-BioKG | 15% | - |
| WN18RR | 11% | - |
| **Hetionet (random)** | **1.7%** | **30-62% for Disease-Gene** |
| **Hetionet (inductive)** | **98.9%** | Drug repurposing scenario |
| OGBL-DDI | 100% | All (by design) |

## Recommendations for Biomedical KG Models

1. **Report stratified metrics**: Overall metrics hide relation-specific blind spots
2. **Track coverage per relation type**: Flag predictions on rare relation types
3. **Prioritize Disease-Gene validation**: These high-stakes predictions need extra scrutiny
4. **Use coverage tracking**: Simple (entity, relation) hash table catches blind spots
5. **Test on inductive splits**: Random splits underestimate real-world OOD prevalence

## Script Location

Analysis script: `scripts/run_hetionet_experiment.py`

## Raw Output

```
Test set breakdown (n=225,021):
  Novel context: 3,866 (1.7%)
  Emerging entity: 405 (0.2%)
  In-distribution: 220,750 (98.1%)

Inductive test breakdown (held-out diseases):
  Novel context: 5,983 (64.9%)
  Emerging entity: 3,138 (34.0%)
  In-distribution: 101 (1.1%)
  TOTAL OOD: 98.9%

Coverage distribution:
  Entities with <= 1 relations: 23,944 (53.7%)
  Entities with <= 2 relations: 24,817 (55.7%)
  Avg relations per entity: 4.08 / 24
  Median: 1.0 (highly skewed)
```
