# Coverage Paradox: Broader Impact Exploration

**Date**: 2026-04-09  
**Status**: Literature review + experimental design

---

## 1. The Original Finding

On FB15k-237 Knowledge Graph:

| Coverage Type | Definition | Hits@10 |
|---------------|------------|---------|
| Full Coverage | Both h and t seen with relation r | 32.3% |
| Partial Zero | Only one entity seen with relation r | **59.5%** |
| Full Zero | Neither entity seen with relation r | 14.8% |

### Identified Mechanisms

1. **Information Dilution**: High-coverage entities have embeddings averaged over many contexts
2. **Overconfidence**: "Seen before" creates false certainty
3. **Anchor Effect**: Partial coverage provides clear signal (one anchor, one target)
4. **Diverse Training**: Partial coverage entities trained on more diverse relations

We call this the **"Familiarity Trap"** or **"Coverage Trap"**.

---

## 2. Literature Connection by Domain

### 2.1 Recommender Systems

#### Related Phenomena

**Popularity Bias** (Abdollahpouri et al., 2019; Chen et al., 2020)
- Popular items are over-recommended
- Users with many interactions get worse personalization
- **Connection**: High-frequency items = "Full Coverage" equivalent

**Cold Start Paradox** (Schein et al., 2002)
- New users sometimes get *better* recommendations than active users
- Fresh preferences vs. diluted historical signal
- **Connection**: "Partial coverage" = cleaner signal

**Filter Bubble / Over-specialization** (Pariser, 2011)
- Heavy users get narrow, overfit recommendations
- Diversity-accuracy tradeoff
- **Connection**: More data can hurt via embedding dilution

#### Key Papers to Examine

1. **"Unbiased Learning-to-Rank with Biased Feedback"** (Joachims et al., 2017)
   - Position bias creates similar "familiarity trap"
   - Frequently shown items appear better regardless of quality

2. **"Debiasing Item-to-Item Recommendations With Small Annotated Datasets"** (Schnabel et al., 2020)
   - Popularity confounds quality estimation
   - Active users = diluted preferences

3. **"User Fatigue and Recommendation Effectiveness"** (Zhang et al., 2022)
   - Long-term users show decreased satisfaction
   - Possible: embeddings become too generalized

#### Experimental Design for RecSys

**Dataset**: MovieLens-20M or Amazon Reviews

**Coverage Definition**:
- Full Coverage: User rated >50 movies in genre G, item has >100 ratings in G
- Partial: User rated <10 OR item has <50 ratings
- Zero: User never rated genre G, item new

**Metrics**:
- Prediction accuracy (RMSE, MAE) by coverage group
- Calibration (predicted confidence vs actual accuracy)

**Hypothesis**: Partial coverage will show better calibration than Full coverage

---

### 2.2 NLP: Entity Linking / NER

#### Related Phenomena

**Type Confusion in Frequent Entities** (Onoe & Durrett, 2020)
- Frequent entities like "Washington" have multiple types (person, location, team)
- Embeddings average over contexts = diluted representation
- **Direct parallel**: Full coverage = high frequency = diluted

**Rare Word Problem** (Luong et al., 2013)
- Rare words paradoxically have cleaner semantics (less polysemy)
- Frequent words = many senses = averaged embedding
- **Connection**: Partial coverage (one anchor) = cleaner signal

**Knowledge Base Population Difficulty** (Ji et al., 2017)
- Linking frequent entities is *harder* than rare ones
- Surprising finding aligns with our paradox

#### Key Papers

1. **"Language Models as Knowledge Bases?"** (Petroni et al., 2019)
   - BERT stores factual knowledge in embeddings
   - Performance varies non-monotonically with fact frequency

2. **"Frequency vs Semantics: The Role of Context in Word Embeddings"** (Schakel & Wilson, 2015)
   - High-frequency words have more diffuse embeddings
   - Low-frequency = specific, focused vectors

3. **"Calibration of Pre-trained Transformers"** (Desai & Durrett, 2020)
   - BERT is overconfident on frequent patterns
   - Calibration error increases with pattern frequency

#### Experimental Design for NLP

**Dataset**: CoNLL-2003 NER or TAC-KBP Entity Linking

**Coverage Definition**:
- Full Coverage: Entity appears >100 times in training
- Partial: 10-50 appearances
- Zero: <5 appearances

**Analysis**:
1. F1 by entity frequency bucket
2. Confidence calibration by frequency
3. Embedding variance analysis (high-freq = lower variance?)

**Prediction**: Medium-frequency entities will have best F1/confidence ratio

---

### 2.3 Computer Vision

#### Related Phenomena

**Long-Tail Recognition** (Liu et al., 2019)
- Head classes (frequent) often confused with each other
- Tail classes (rare) have cleaner features
- **Connection**: "Full coverage" objects = confused representations

**Scene Context Effects** (Divvala et al., 2009)
- Objects in typical contexts = overconfident predictions
- Objects in atypical contexts = actually more accurate when correct
- **Connection**: Partial context = better calibrated

**Dataset Bias** (Torralba & Efros, 2011)
- Frequently co-occurring objects bias predictions
- Model learns spurious correlations from "full coverage" patterns

#### Key Papers

1. **"Learning to Segment Everything"** (Hu et al., 2018)
   - Rare object categories have cleaner masks than common ones
   - Common categories = blurry boundaries (diluted)

2. **"The Visual Task Adaptation Benchmark"** (Zhai et al., 2019)
   - Pre-trained features work better on rare classes
   - Frequent classes = overfit to ImageNet distribution

3. **"Calibrating Deep Neural Networks"** (Guo et al., 2017)
   - Modern DNNs are overconfident
   - Overconfidence worse for frequent patterns

#### Experimental Design for CV

**Dataset**: COCO or ImageNet-LT (long-tail variant)

**Coverage Definition**:
- Full Coverage: Object class in >10K training images, typical scene
- Partial: 1K-5K images OR atypical scene
- Zero: <500 images, novel scene composition

**Analysis**:
1. mAP by frequency band
2. Confidence vs accuracy scatter plot by coverage
3. Feature space analysis (t-SNE of frequent vs rare class centroids)

**Prediction**: Per-class calibration error inversely correlated with frequency

---

### 2.4 Theoretical Framing: "Familiarity Trap"

#### Psychology Connection

**Illusion of Knowledge** (Rozenblit & Keil, 2002)
- Humans overestimate understanding of frequently encountered things
- "I've seen it before" != "I understand it"
- **Direct parallel**: Model confidence from frequency, not comprehension

**Mere Exposure Effect** (Zajonc, 1968)
- Familiarity breeds liking, not necessarily accuracy
- Applies to ML: frequency breeds confidence, not correctness

**Fluency Heuristic** (Hertwig et al., 2008)
- Easily recalled = perceived as true/likely
- ML: easily retrieved embedding = high confidence

#### ML Theory Connection

**Calibration under Distribution Shift** (Ovadia et al., 2019)
- Models are well-calibrated on frequent patterns
- BUT: this calibration fails under shift
- **Connection**: "Full coverage" = in-distribution overconfidence

**Epistemic vs Aleatoric Uncertainty** (Kendall & Gal, 2017)
- Epistemic (reducible): should decrease with more data
- BUT: averaging effect can *increase* uncertainty needs
- **Connection**: More coverage = more aleatoric, mistaken for epistemic

**Information Bottleneck** (Tishby et al., 2015)
- Optimal representations compress irrelevant info
- High-frequency patterns = more irrelevant variation to compress
- **Connection**: Dilution is compression failure

---

## 3. Most Promising Directions

### Tier 1: Immediate Validation (Existing Datasets)

#### 3.1 MovieLens Popularity Paradox
**Why promising**:
- Dataset readily available (MovieLens-20M)
- Clear coverage definition (user-item interaction patterns)
- Existing popularity bias literature to compare

**Quick validation**:
```python
# Pseudo-code
for user, item, rating in test_set:
    user_coverage = count(user_item_pairs_in_train[user])
    item_coverage = count(user_item_pairs_in_train[:, item])
    
    coverage_type = classify(user_coverage, item_coverage)
    # Full, Partial, Zero
    
    rmse_by_type[coverage_type].append(abs(pred - true))
    confidence_by_type[coverage_type].append(model_confidence)
```

**Expected outcome**: Partial coverage shows better RMSE/confidence ratio

#### 3.2 BERT Confidence on Entity Frequency
**Why promising**:
- Pre-trained models available
- Clear frequency-based stratification
- Connects to "BERT as KB" literature

**Quick validation**:
```python
# LAMA probe on BERT
for entity in test_entities:
    freq = entity_frequency_in_training[entity]
    mask_accuracy, confidence = probe_bert(entity, relation)
    
    results_by_freq[bucket(freq)].append({
        'accuracy': mask_accuracy,
        'confidence': confidence
    })
```

**Expected outcome**: Medium-frequency entities have best calibration

---

### Tier 2: Novel Experiments (Need Computation)

#### 3.3 ImageNet-LT Coverage Analysis
**Setup**:
- Use ImageNet-LT (class imbalance benchmark)
- Stratify test samples by class frequency AND scene typicality

**Novel contribution**: Scene context as second coverage dimension

#### 3.4 Cross-Domain Unified Framework
**Setup**:
- Define "Familiarity Score" = f(entity frequency, context frequency)
- Apply same metric across KG, RecSys, NLP, CV
- Test if paradox threshold is universal

---

## 4. Unified Theory Proposal

### The Familiarity Trap Framework

**Definition**: A learned representation exhibits the Familiarity Trap when:
1. Confidence monotonically increases with exposure frequency
2. Accuracy peaks at intermediate frequency, then *decreases*
3. The gap (confidence - accuracy) maximizes at high frequency

**Formal Statement**:

Let:
- $f(x)$ = exposure frequency of sample x
- $c(x)$ = model confidence on x  
- $a(x)$ = model accuracy on x

The Familiarity Trap occurs when:
$$\frac{\partial c}{\partial f} > 0 \quad \text{but} \quad \frac{\partial a}{\partial f} < 0 \text{ for } f > f^*$$

**Mechanism (Embedding Dilution)**:

For entity $e$ with embedding $\mathbf{e}$:
$$\mathbf{e} = \frac{1}{|\mathcal{C}(e)|} \sum_{c \in \mathcal{C}(e)} \mathbf{e}_c$$

where $\mathcal{C}(e)$ = contexts in which $e$ appears.

- When $|\mathcal{C}(e)| \uparrow$: embedding becomes average of many contexts
- Variance $\downarrow$ (appears confident) but informativeness $\downarrow$ (diluted)

---

## 5. Paper Implications

### If Paradox Generalizes:

**New contribution scope**:
1. Not just KG-specific finding
2. Fundamental property of embedding-based models
3. Connects uncertainty estimation to representation learning

**Potential paper framing**:
- Title: "The Familiarity Trap: Why More Data Can Hurt in Embedding-Based Models"
- Venue: ICML/NeurIPS (broader ML audience)

### Key Experiments Needed:

| Domain | Dataset | Status | Priority |
|--------|---------|--------|----------|
| KG | FB15k-237 | Done | - |
| RecSys | MovieLens-20M | TODO | High |
| NLP | CoNLL-2003 / LAMA | TODO | High |
| CV | ImageNet-LT | TODO | Medium |

---

## 6. Next Steps

1. **MovieLens validation** (1-2 days)
   - Download MovieLens-20M
   - Train simple matrix factorization
   - Stratify by user/item coverage
   - Measure accuracy and calibration

2. **BERT probe** (1 day)
   - Use LAMA benchmark
   - Stratify entities by Wikipedia frequency
   - Measure accuracy/confidence correlation

3. **Literature deep-dive** (ongoing)
   - Read Popularity Bias survey (2021)
   - Check calibration literature for frequency effects
   - Search for "embedding dilution" or "representation averaging"

4. **Theory formalization** (if experiments confirm)
   - Prove Familiarity Trap theorem
   - Derive optimal frequency threshold
   - Connect to information theory

---

## 7. References (To Read)

### Recommender Systems
- Abdollahpouri et al. (2019). "The Unfairness of Popularity Bias in Recommendation"
- Chen et al. (2020). "Bias and Debias in Recommender System: A Survey"
- Steck (2011). "Item Popularity and Recommendation Accuracy"

### NLP
- Petroni et al. (2019). "Language Models as Knowledge Bases?"
- Desai & Durrett (2020). "Calibration of Pre-trained Transformers"
- Onoe & Durrett (2020). "Fine-Grained Entity Typing for Domain Independent"

### Computer Vision
- Liu et al. (2019). "Large-Scale Long-Tailed Recognition in an Open World"
- Guo et al. (2017). "On Calibration of Modern Neural Networks"

### Theory
- Tishby et al. (2015). "Deep Learning and the Information Bottleneck"
- Ovadia et al. (2019). "Can You Trust Your Model's Uncertainty?"
- Rozenblit & Keil (2002). "The Misunderstood Limits of Folk Science"

---

## Appendix: Quick Validation Script Outline

```python
# MovieLens Coverage Paradox Test
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load MovieLens
ratings = pd.read_csv('ratings.csv')
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Build coverage
user_coverage = train.groupby('userId').size()
item_coverage = train.groupby('movieId').size()

def classify_coverage(row):
    u_cov = user_coverage.get(row['userId'], 0)
    i_cov = item_coverage.get(row['movieId'], 0)
    
    u_high = u_cov > np.percentile(user_coverage, 75)
    i_high = i_cov > np.percentile(item_coverage, 75)
    
    if u_high and i_high:
        return 'full'
    elif u_high or i_high:
        return 'partial'
    else:
        return 'zero'

test['coverage'] = test.apply(classify_coverage, axis=1)

# Train simple MF model
# ... (standard matrix factorization)

# Evaluate by coverage group
for cov_type in ['full', 'partial', 'zero']:
    subset = test[test['coverage'] == cov_type]
    rmse = compute_rmse(subset)
    calibration = compute_calibration(subset)
    print(f"{cov_type}: RMSE={rmse:.3f}, Calibration={calibration:.3f}")
```
