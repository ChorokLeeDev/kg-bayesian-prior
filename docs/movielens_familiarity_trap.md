# MovieLens Familiarity Trap Verification

## Summary

**Key Finding**: MovieLens does NOT exhibit the same Familiarity Trap as Knowledge Graphs, but shows a related phenomenon through **Rating Diversity**.

| Domain | "Familiar" Entity Accuracy | "Novel" Entity Accuracy | Pattern |
|--------|---------------------------|-------------------------|---------|
| KG (FB15k-237) | 32.3% | 59.5% | Familiarity = WORSE |
| MovieLens | MAE 0.71 | MAE 0.95 | Familiarity = BETTER |

## Background: KG Familiarity Trap

In Knowledge Graphs, we discovered:
- **Full Coverage** (entity seen with many relation types): 32.3% accuracy
- **Partial Zero** (entity seen with few relation types): 59.5% accuracy
- **Cause**: Embedding dilution - one embedding vector must encode ALL relation semantics

## MovieLens Results

### User Activity Effect
| Category | N Predictions | MAE | RMSE |
|----------|--------------|-----|------|
| Light (<50 ratings) | 2,339 | 0.7887 | 0.9928 |
| Medium (50-150) | 6,061 | 0.7345 | 0.9408 |
| Heavy (>150) | 11,600 | 0.7288 | 0.9201 |

**Finding**: Heavy users have LOWER error (opposite of KG)

### Item Popularity Effect
| Category | N Predictions | MAE | RMSE |
|----------|--------------|-----|------|
| Unpopular (<30 ratings) | 1,610 | 0.8248 | 1.0306 |
| Medium (30-100) | 5,389 | 0.7575 | 0.9532 |
| Popular (>100) | 13,001 | 0.7184 | 0.9150 |

**Finding**: Popular items have LOWER error (opposite of KG)

### KG-Style Coverage Analysis
| Coverage Type | N Predictions | MAE | RMSE |
|---------------|--------------|-----|------|
| Full Coverage (Heavy + Popular) | 6,934 | 0.7090 | 0.8985 |
| Partial Zero (Asymmetric) | 10,733 | 0.7418 | 0.9413 |
| Zero Coverage (Light + Unpopular) | 124 | 0.9516 | 1.1671 |

**Finding**: Full Coverage performs BEST (opposite of KG)

## The Real Dilution Effect: Rating Diversity

While frequency doesn't cause dilution in MovieLens, **rating diversity** does:

### User Rating Diversity Analysis
| Diversity | N Predictions | MAE | Avg Std |
|-----------|--------------|-----|---------|
| Low (std < 0.8) | 1,889 | **0.5357** | 0.73 |
| Medium (0.8-1.2) | 14,977 | 0.7129 | 0.99 |
| High (std > 1.2) | 3,134 | **0.9767** | 1.32 |

**Critical Finding**: Users with diverse rating patterns (high std) have 82% higher MAE than consistent raters!

### Controlled Test: Heavy Users by Rating Diversity
| Diversity | MAE | N Predictions |
|-----------|-----|--------------|
| Low | 0.5393 | 781 |
| Medium | 0.7032 | 9,067 |
| High | 0.9455 | 1,752 |

**This IS the MovieLens version of embedding dilution!**
- Heavy user + consistent ratings = accurate embedding = low MAE
- Heavy user + diverse ratings = diluted embedding = high MAE

## Genre Diversity Analysis (Multi-Relation Analogy)

### All Items
| Genre Diversity | N Predictions | MAE | Avg Genres |
|-----------------|--------------|-----|------------|
| Single genre | 5,961 | 0.7602 | 1.0 |
| Multi-genre (2-3) | 12,256 | 0.7260 | 2.4 |
| Many genres (4+) | 1,783 | 0.7404 | 4.3 |

### Popular Items Only (>=100 ratings)
| Genre Diversity | MAE | N Predictions |
|-----------------|-----|--------------|
| Single genre | 0.7418 | 2,750 |
| Multi-genre | 0.7096 | 8,806 |
| Many genres | 0.7276 | 1,445 |

**Finding**: Weak or no genre-based dilution effect

## Why MovieLens Differs from KG

### 1. Single vs Multi-Relation Structure
| Aspect | Knowledge Graph | MovieLens |
|--------|-----------------|-----------|
| Relation types | Many (237 in FB15k) | One ("rates") |
| Embedding task | Encode all relations | Encode preference |
| More data means | More constraints | Better estimation |

### 2. Nature of Dilution
| Factor | KG | MovieLens |
|--------|-----|-----------|
| Frequency | Causes dilution | Improves accuracy |
| Diversity | (same as frequency) | **Causes dilution** |
| Mechanism | Multi-relation constraint | Multi-preference variance |

### 3. The Key Difference
- **KG**: One entity embedding must satisfy constraints from DIFFERENT relation types
  - "Paris" must work with: `capitalOf`, `locatedIn`, `hasPopulation`, `foundedIn`, ...
  - Each relation pulls the embedding in different directions
  
- **MovieLens**: One user embedding captures ONE preference distribution
  - More ratings = better estimate of the SAME preference
  - Unless user has diverse tastes (high rating variance)

## Conclusion

The Familiarity Trap in KG is caused by **relation type diversity**, not mere frequency. MovieLens shows that:

1. **Frequency alone is beneficial** - more data = better embeddings
2. **Diversity causes dilution** - users with varied ratings have worse predictions
3. **The KG effect is specific to multi-relational structures**

### Implication for KG Research
The coverage blind spot in KG is fundamentally about **structural diversity**, not observation count. Solutions should focus on:
- Relation-specific embeddings (different embedding per relation type)
- Coverage tracking at the (entity, relation) pair level, not entity level
- Explicit uncertainty quantification for multi-relation entities

## Experimental Setup

- **Dataset**: MovieLens 100K (100,000 ratings, 943 users, 1,682 items)
- **Model**: SVD with 100 factors, 20 epochs
- **Split**: 80/20 train/test
- **Overall Performance**: RMSE 0.9352, MAE 0.7375

## Files

- Script: `scripts/movielens_familiarity_trap.py`
- Results: `outputs/movielens/familiarity_trap_results.json`
- Visualization: `outputs/movielens/familiarity_trap_visualization.png`
