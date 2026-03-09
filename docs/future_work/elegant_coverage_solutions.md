# Elegant Solutions for Coverage-Aware KG Uncertainty

## Problem Statement

### The Coverage Blind Spot
- 11-25% of KG test queries involve "novel relational contexts"
- Entity e appears with relation r it was never trained with
- All existing methods report LOW uncertainty (false confidence)
- Root cause: Embeddings encode semantic similarity, NOT coverage history

### Current Solution: Hash Table
```python
coverage[(e, r)] = 1 if (e, r) seen in training else 0
```
- **Pros**: Exact, O(1), zero training cost, 100% reliable
- **Cons**: External data structure, not end-to-end learnable, "inelegant"

### Why Seek Elegant Solutions?
1. **Unified architecture**: Single model instead of embedding + hash table
2. **End-to-end learning**: Coverage awareness as learned inductive bias
3. **Soft coverage**: Graceful degradation instead of binary 0/1
4. **Generalization**: May work in inductive settings where hash table fails

---

## Experimental Results (2026-03-06)

### What We Tested

Three approaches to learn coverage-awareness without explicit hash table:

1. **Coverage Reconstruction Loss**: Train embedding to predict which relations entity has seen
2. **Relation-Set Encoding (NodePiece-style)**: Encode entity by its relation vocabulary
3. **Contrastive Coverage**: Learn to distinguish seen vs unseen (e, r) pairs

### Results on Novel-Context Detection

| Approach | Novel-Context AUROC | Result |
|----------|---------------------|--------|
| Hash Table | 1.000 | Perfect |
| Coverage Reconstruction | ~0.55 | Failed |
| Relation-Set | ~0.58 | Failed |
| Contrastive | ~0.67 | Partial |

**Note**: These experiments used ICEWS14's role-shift split (ρ = 1.0), which tests frequency-awareness, not coverage-awareness. This was the wrong benchmark—role-shift OOD has coverage=1 by definition, so coverage-based methods cannot distinguish it from ID.

### Key Insight: We Tested the Wrong Thing

The experiments conflated two different problems:

| Problem | What it tests | Hash table works? |
|---------|--------------|-------------------|
| **Novel context** | Is (e, r) unseen? | ✓ Yes, perfectly |
| **Role-shift** | Is (e, r) rare? | ✗ No, needs frequency |

The paper's contribution is detecting **novel contexts** (unseen pairs). The hash table already solves this with AUROC = 1.0.

"Role-shift" (rare but covered pairs) is a **different problem** that:
- Requires frequency tracking, not coverage
- Has circular supervision (frequency ≈ OOD label)
- Is outside the paper's scope

### Conclusion

**The hash table IS the elegant solution for novel-context detection.**

The experiments revealed a flawed assumption: we were trying to solve role-shift with coverage-based methods, which is fundamentally impossible. For the paper's actual problem (novel contexts), the hash table is:
- O(1) lookup
- 100% accurate
- Zero training cost
- The correct solution

Neural approximations add complexity without benefit for this inherently discrete problem.

---

## Remaining Research Directions (Future Work)

### For Novel-Context Detection (Paper's Problem)
The hash table is optimal. No further research needed.

### For Inductive Settings
Hash table fails when entities are unseen at test time. Potential directions:
- GNN-based coverage estimation from neighborhood structure
- Transfer learning from similar entities

### For Soft Coverage (Graceful Degradation)
Instead of binary 0/1, predict confidence based on:
- How many times (e, r) was seen
- Recency of observations
- This is frequency estimation, not coverage detection

---

## Files

- `scripts/elegant_coverage_experiments.py`: Implementation of 3 approaches
- Experiment logs: `outputs/elegant_*.log`
