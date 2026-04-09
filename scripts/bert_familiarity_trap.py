#!/usr/bin/env python3
"""
BERT Familiarity Trap Analysis

Hypothesis: High-frequency entities in BERT's pretraining corpus suffer from
"embedding dilution" - they appear in too many diverse contexts, leading to
less specific representations and lower factual accuracy.

This mirrors the "Coverage Paradox" found in KG embeddings:
- Full Coverage (many contexts seen): lower accuracy
- Partial/Zero Coverage (fewer contexts): higher accuracy

Methodology:
1. Create factual probes: "[Entity] was born in [MASK]"
2. Measure entity frequency via Wikipedia pageview API or proxy
3. Stratify by frequency and compare:
   - Accuracy (does BERT predict the right answer?)
   - Confidence (softmax probability)
   - Calibration (confidence vs accuracy alignment)
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

warnings.filterwarnings("ignore")

# Factual probes: (subject, relation_template, object, subject_type)
# Templates use [MASK] where the answer should go
FACTUAL_PROBES = [
    # Birthplace probes (high-freq entities tend to be famous people)
    ("Barack Obama", "[X] was born in [MASK].", "Hawaii", "politician"),
    ("Albert Einstein", "[X] was born in [MASK].", "Germany", "scientist"),
    ("Shakespeare", "[X] was born in [MASK].", "England", "writer"),
    ("Mozart", "[X] was born in [MASK].", "Austria", "musician"),
    ("Napoleon", "[X] was born in [MASK].", "Corsica", "historical"),
    ("Picasso", "[X] was born in [MASK].", "Spain", "artist"),
    ("Gandhi", "[X] was born in [MASK].", "India", "politician"),
    ("Confucius", "[X] was born in [MASK].", "China", "philosopher"),
    ("Aristotle", "[X] was born in [MASK].", "Greece", "philosopher"),
    ("Darwin", "[X] was born in [MASK].", "England", "scientist"),

    # Less famous entities (lower frequency, potentially more specific embeddings)
    ("Kepler", "[X] was born in [MASK].", "Germany", "scientist"),
    ("Mendel", "[X] was born in [MASK].", "Austria", "scientist"),
    ("Faraday", "[X] was born in [MASK].", "England", "scientist"),
    ("Planck", "[X] was born in [MASK].", "Germany", "scientist"),
    ("Bohr", "[X] was born in [MASK].", "Denmark", "scientist"),

    # Capital probes (relation: X is the capital of Y)
    ("Paris", "[X] is the capital of [MASK].", "France", "city"),
    ("Tokyo", "[X] is the capital of [MASK].", "Japan", "city"),
    ("Berlin", "[X] is the capital of [MASK].", "Germany", "city"),
    ("London", "[X] is the capital of [MASK].", "England", "city"),  # Could be UK
    ("Rome", "[X] is the capital of [MASK].", "Italy", "city"),
    ("Moscow", "[X] is the capital of [MASK].", "Russia", "city"),
    ("Beijing", "[X] is the capital of [MASK].", "China", "city"),
    ("Cairo", "[X] is the capital of [MASK].", "Egypt", "city"),
    ("Madrid", "[X] is the capital of [MASK].", "Spain", "city"),
    ("Canberra", "[X] is the capital of [MASK].", "Australia", "city"),
    ("Ottawa", "[X] is the capital of [MASK].", "Canada", "city"),
    ("Brasilia", "[X] is the capital of [MASK].", "Brazil", "city"),
    ("Wellington", "[X] is the capital of [MASK].", "Zealand", "city"),  # New Zealand
    ("Bern", "[X] is the capital of [MASK].", "Switzerland", "city"),
    ("Oslo", "[X] is the capital of [MASK].", "Norway", "city"),

    # Profession probes
    ("Einstein", "[X] was a famous [MASK].", "physicist", "scientist"),
    ("Picasso", "[X] was a famous [MASK].", "painter", "artist"),
    ("Mozart", "[X] was a famous [MASK].", "composer", "musician"),
    ("Shakespeare", "[X] was a famous [MASK].", "playwright", "writer"),
    ("Newton", "[X] was a famous [MASK].", "physicist", "scientist"),
    ("Darwin", "[X] was a famous [MASK].", "biologist", "scientist"),
    ("Freud", "[X] was a famous [MASK].", "psychologist", "scientist"),
    ("Beethoven", "[X] was a famous [MASK].", "composer", "musician"),
    ("Hemingway", "[X] was a famous [MASK].", "writer", "writer"),
    ("Tolstoy", "[X] was a famous [MASK].", "writer", "writer"),

    # Language probes (X speaks Y)
    ("French", "People in France speak [MASK].", "French", "language"),
    ("German", "People in Germany speak [MASK].", "German", "language"),
    ("Japanese", "People in Japan speak [MASK].", "Japanese", "language"),
    ("Spanish", "People in Spain speak [MASK].", "Spanish", "language"),
    ("Italian", "People in Italy speak [MASK].", "Italian", "language"),
    ("Portuguese", "People in Portugal speak [MASK].", "Portuguese", "language"),
    ("Russian", "People in Russia speak [MASK].", "Russian", "language"),
    ("Chinese", "People in China speak [MASK].", "Chinese", "language"),
    ("Arabic", "People in Egypt speak [MASK].", "Arabic", "language"),
    ("Korean", "People in Korea speak [MASK].", "Korean", "language"),
]

# Estimated Wikipedia popularity tiers (proxy for pretraining frequency)
# Based on typical pageview volumes: 1=very high, 2=high, 3=medium, 4=low
ENTITY_FREQUENCY_TIER = {
    # Very high frequency (millions of pageviews)
    "Barack Obama": 1, "Albert Einstein": 1, "Shakespeare": 1, "Napoleon": 1,
    "Paris": 1, "London": 1, "Tokyo": 1, "Rome": 1, "Berlin": 1, "Moscow": 1,
    "Beijing": 1, "Einstein": 1, "Newton": 1, "Darwin": 1,

    # High frequency
    "Mozart": 2, "Picasso": 2, "Gandhi": 2, "Confucius": 2, "Aristotle": 2,
    "Madrid": 2, "Cairo": 2, "Beethoven": 2, "Freud": 2, "Hemingway": 2,

    # Medium frequency
    "Kepler": 3, "Mendel": 3, "Faraday": 3, "Planck": 3, "Bohr": 3,
    "Canberra": 3, "Ottawa": 3, "Brasilia": 3, "Tolstoy": 3, "Wellington": 3,

    # Low frequency
    "Bern": 4, "Oslo": 4,

    # Language entries (not entities per se)
    "French": 2, "German": 2, "Japanese": 2, "Spanish": 2, "Italian": 2,
    "Portuguese": 3, "Russian": 2, "Chinese": 2, "Arabic": 2, "Korean": 3,
}


def load_bert_model():
    """Load BERT model and tokenizer for masked LM."""
    from transformers import BertTokenizer, BertForMaskedLM
    import torch

    print("Loading BERT-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # Use MPS if available (Apple Silicon), else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)
    return model, tokenizer, device


def probe_bert(
    model,
    tokenizer,
    device,
    subject: str,
    template: str,
    expected_answer: str,
    top_k: int = 10
) -> Dict:
    """
    Probe BERT with a factual query and return predictions.

    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        device: torch device
        subject: Entity name
        template: Template with [X] for subject and [MASK] for answer
        expected_answer: Ground truth answer
        top_k: Number of top predictions to return

    Returns:
        Dictionary with predictions, confidence, and correctness
    """
    import torch

    # Create the probe sentence
    sentence = template.replace("[X]", subject)

    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    if len(mask_idx) == 0:
        return {"error": "No [MASK] token found"}

    mask_idx = mask_idx[0].item()

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_idx]
        probs = torch.softmax(logits, dim=-1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tokenizer.decode([idx]).strip() for idx in top_indices]

    # Check if expected answer is in top predictions
    expected_lower = expected_answer.lower()
    rank = -1
    for i, token in enumerate(top_tokens):
        if token.lower() == expected_lower or expected_lower in token.lower():
            rank = i + 1
            break

    # Also check full vocabulary for expected answer
    expected_tokens = tokenizer.encode(expected_answer, add_special_tokens=False)
    if len(expected_tokens) == 1:
        expected_prob = probs[expected_tokens[0]].item()
        expected_rank_full = (probs > expected_prob).sum().item() + 1
    else:
        expected_prob = 0.0
        expected_rank_full = -1

    return {
        "sentence": sentence,
        "subject": subject,
        "expected": expected_answer,
        "top_predictions": list(zip(top_tokens, top_probs.cpu().numpy().tolist())),
        "top1_token": top_tokens[0],
        "top1_prob": top_probs[0].item(),
        "expected_prob": expected_prob,
        "rank_in_top10": rank,
        "rank_full": expected_rank_full,
        "correct_top1": top_tokens[0].lower() == expected_lower or expected_lower in top_tokens[0].lower(),
        "correct_top5": rank != -1 and rank <= 5,
        "correct_top10": rank != -1,
    }


def analyze_by_frequency(results: List[Dict]) -> Dict:
    """
    Analyze results stratified by entity frequency tier.

    Returns:
        Dictionary with metrics per frequency tier
    """
    tier_results = defaultdict(list)

    for r in results:
        if "error" in r:
            continue
        subject = r["subject"]
        tier = ENTITY_FREQUENCY_TIER.get(subject, 3)  # Default to medium
        tier_results[tier].append(r)

    analysis = {}
    for tier in sorted(tier_results.keys()):
        tier_data = tier_results[tier]
        n = len(tier_data)
        if n == 0:
            continue

        acc_top1 = sum(1 for r in tier_data if r["correct_top1"]) / n
        acc_top5 = sum(1 for r in tier_data if r["correct_top5"]) / n
        acc_top10 = sum(1 for r in tier_data if r["correct_top10"]) / n
        avg_conf = np.mean([r["top1_prob"] for r in tier_data])
        avg_expected_prob = np.mean([r["expected_prob"] for r in tier_data])

        # Confidence when wrong (overconfidence indicator)
        wrong_results = [r for r in tier_data if not r["correct_top1"]]
        conf_when_wrong = np.mean([r["top1_prob"] for r in wrong_results]) if wrong_results else 0

        # Confidence when right
        right_results = [r for r in tier_data if r["correct_top1"]]
        conf_when_right = np.mean([r["top1_prob"] for r in right_results]) if right_results else 0

        analysis[tier] = {
            "n_samples": n,
            "accuracy_top1": acc_top1,
            "accuracy_top5": acc_top5,
            "accuracy_top10": acc_top10,
            "avg_confidence": avg_conf,
            "avg_expected_prob": avg_expected_prob,
            "confidence_when_wrong": conf_when_wrong,
            "confidence_when_right": conf_when_right,
            "overconfidence_gap": conf_when_wrong - (1 - acc_top1),  # Should be ~0 if calibrated
        }

    return analysis


def analyze_context_diversity(results: List[Dict]) -> Dict:
    """
    Analyze by probe type (birthplace, capital, profession, language).
    Different relation types test different aspects of embedding specificity.
    """
    type_results = defaultdict(list)

    for r, probe in zip(results, FACTUAL_PROBES):
        if "error" in r:
            continue
        probe_type = probe[3]  # subject_type
        type_results[probe_type].append(r)

    analysis = {}
    for ptype, data in type_results.items():
        n = len(data)
        if n == 0:
            continue
        acc = sum(1 for r in data if r["correct_top1"]) / n
        avg_conf = np.mean([r["top1_prob"] for r in data])
        analysis[ptype] = {
            "n_samples": n,
            "accuracy_top1": acc,
            "avg_confidence": avg_conf,
        }

    return analysis


def compute_calibration_metrics(results: List[Dict], n_bins: int = 5) -> Dict:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.

    A well-calibrated model should have confidence ~ accuracy.
    Familiarity trap hypothesis: high-freq entities are overconfident.
    """
    valid_results = [r for r in results if "error" not in r]

    # Bin by confidence
    confidences = np.array([r["top1_prob"] for r in valid_results])
    accuracies = np.array([1 if r["correct_top1"] else 0 for r in valid_results])

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        bin_count = mask.sum()
        bins.append({
            "range": f"[{low:.2f}, {high:.2f})",
            "avg_confidence": bin_conf,
            "avg_accuracy": bin_acc,
            "count": int(bin_count),
            "gap": bin_conf - bin_acc,  # Positive = overconfident
        })

    # ECE: weighted average of |conf - acc| per bin
    ece = sum(b["count"] * abs(b["gap"]) for b in bins) / len(valid_results)

    # Overconfidence rate: fraction of bins where conf > acc
    overconf_bins = sum(1 for b in bins if b["gap"] > 0.05)  # 5% threshold

    return {
        "ece": ece,
        "n_bins": len(bins),
        "overconfident_bins": overconf_bins,
        "bins": bins,
    }


def run_experiment():
    """Main experiment: probe BERT and analyze familiarity trap."""
    print("=" * 60)
    print("BERT Familiarity Trap Analysis")
    print("=" * 60)
    print()

    # Load model
    model, tokenizer, device = load_bert_model()
    print()

    # Run all probes
    print(f"Running {len(FACTUAL_PROBES)} factual probes...")
    results = []

    for subject, template, answer, _ in FACTUAL_PROBES:
        result = probe_bert(model, tokenizer, device, subject, template, answer)
        results.append(result)

        status = "OK" if result.get("correct_top1") else "X "
        print(f"  [{status}] {subject}: {result.get('top1_token', 'ERROR')} "
              f"(expected: {answer}, conf: {result.get('top1_prob', 0):.3f})")

    print()

    # Overall stats
    valid_results = [r for r in results if "error" not in r]
    overall_acc = sum(1 for r in valid_results if r["correct_top1"]) / len(valid_results)
    print(f"Overall Top-1 Accuracy: {overall_acc:.1%}")
    print()

    # Analysis by frequency tier
    print("=" * 60)
    print("ANALYSIS BY ENTITY FREQUENCY")
    print("=" * 60)
    print("Tier 1 = Very High Freq (millions of pageviews)")
    print("Tier 4 = Low Freq")
    print()

    freq_analysis = analyze_by_frequency(results)

    print(f"{'Tier':<6} {'N':<4} {'Acc@1':<8} {'Acc@5':<8} {'Conf':<8} {'Conf|Wrong':<12} {'Conf|Right':<12}")
    print("-" * 70)

    for tier in sorted(freq_analysis.keys()):
        a = freq_analysis[tier]
        print(f"Tier {tier:<2} {a['n_samples']:<4} {a['accuracy_top1']:.1%}    "
              f"{a['accuracy_top5']:.1%}    {a['avg_confidence']:.3f}    "
              f"{a['confidence_when_wrong']:.3f}        {a['confidence_when_right']:.3f}")

    print()

    # Analysis by probe type
    print("=" * 60)
    print("ANALYSIS BY PROBE TYPE")
    print("=" * 60)
    type_analysis = analyze_context_diversity(results)

    print(f"{'Type':<12} {'N':<4} {'Acc@1':<8} {'Avg Conf':<10}")
    print("-" * 40)
    for ptype in sorted(type_analysis.keys()):
        a = type_analysis[ptype]
        print(f"{ptype:<12} {a['n_samples']:<4} {a['accuracy_top1']:.1%}    {a['avg_confidence']:.3f}")

    print()

    # Calibration analysis
    print("=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    cal = compute_calibration_metrics(results)

    print(f"Expected Calibration Error (ECE): {cal['ece']:.3f}")
    print(f"Overconfident bins: {cal['overconfident_bins']}/{cal['n_bins']}")
    print()
    print("Reliability Diagram:")
    print(f"{'Conf Range':<15} {'Avg Conf':<10} {'Avg Acc':<10} {'Gap':<10} {'N':<6}")
    print("-" * 55)
    for b in cal["bins"]:
        gap_indicator = " (+)" if b["gap"] > 0.05 else ""
        print(f"{b['range']:<15} {b['avg_confidence']:.3f}      {b['avg_accuracy']:.3f}      "
              f"{b['gap']:+.3f}{gap_indicator}     {b['count']}")

    print()

    # Key findings
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Compare high-freq vs low-freq
    if 1 in freq_analysis and len([t for t in freq_analysis if t >= 3]) > 0:
        high_freq = freq_analysis[1]
        low_freq_tiers = [freq_analysis[t] for t in freq_analysis if t >= 3]
        low_freq_acc = np.mean([t["accuracy_top1"] for t in low_freq_tiers])
        low_freq_conf_wrong = np.mean([t["confidence_when_wrong"] for t in low_freq_tiers])

        print(f"1. Accuracy comparison:")
        print(f"   - High-freq entities (Tier 1): {high_freq['accuracy_top1']:.1%}")
        print(f"   - Low-freq entities (Tier 3-4): {low_freq_acc:.1%}")

        acc_diff = low_freq_acc - high_freq["accuracy_top1"]
        if acc_diff > 0.05:
            print(f"   >>> FAMILIARITY TRAP DETECTED: Low-freq is {acc_diff:.1%} MORE accurate")
        elif acc_diff < -0.05:
            print(f"   >>> NO familiarity trap: High-freq is more accurate")
        else:
            print(f"   >>> Marginal difference ({acc_diff:+.1%})")

        print()
        print(f"2. Overconfidence when wrong:")
        print(f"   - High-freq: {high_freq['confidence_when_wrong']:.3f}")
        print(f"   - Low-freq: {low_freq_conf_wrong:.3f}")

        if high_freq["confidence_when_wrong"] > low_freq_conf_wrong + 0.02:
            print(f"   >>> High-freq entities show MORE overconfidence")

    print()

    # Parallel to KG findings
    print("=" * 60)
    print("PARALLEL TO KG COVERAGE PARADOX")
    print("=" * 60)
    print("""
KG Finding (from paper):
- Full Coverage (seen in many contexts): 32.3% accuracy
- Partial Zero (seen in fewer contexts): 59.5% accuracy
- Cause: Embedding dilution from diverse training contexts

BERT Hypothesis:
- High-freq entities (many Wikipedia mentions): potentially lower factual accuracy
- Low-freq entities (fewer mentions): potentially higher factual accuracy
- Same cause: Embedding dilution from diverse contexts
    """)

    # Save results
    output_dir = Path("/Users/i767700/Github/kg-bayesian-prior/outputs")
    output_dir.mkdir(exist_ok=True)

    save_data = {
        "overall_accuracy": overall_acc,
        "frequency_analysis": freq_analysis,
        "type_analysis": type_analysis,
        "calibration": {k: v for k, v in cal.items() if k != "bins"},
        "calibration_bins": cal["bins"],
        "detailed_results": results,
    }

    with open(output_dir / "bert_familiarity_trap.json", "w") as f:
        json.dump(save_data, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir / 'bert_familiarity_trap.json'}")

    return results, freq_analysis, type_analysis, cal


def extended_analysis():
    """
    Extended analysis: Compare embedding similarity for high vs low freq entities.
    High-freq entities should have more "diluted" (spread out) embeddings.
    """
    from transformers import BertTokenizer, BertModel
    import torch

    print("\n" + "=" * 60)
    print("EXTENDED: EMBEDDING ANALYSIS")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # Get embeddings for entities in different contexts
    high_freq_entities = ["Obama", "Einstein", "Paris", "London"]
    low_freq_entities = ["Kepler", "Mendel", "Bern", "Oslo"]

    contexts = [
        "{} is famous.",
        "{} is interesting.",
        "I read about {}.",
        "The history of {} is complex.",
    ]

    def get_entity_embeddings(entity, contexts):
        """Get BERT embeddings for entity in different contexts."""
        embeddings = []
        for ctx in contexts:
            sentence = ctx.format(entity)
            inputs = tokenizer(sentence, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            # Find entity tokens
            tokens = tokenizer.tokenize(sentence)
            entity_tokens = tokenizer.tokenize(entity)

            # Get CLS embedding as sentence representation
            cls_emb = outputs.last_hidden_state[0, 0].numpy()
            embeddings.append(cls_emb)

        return np.array(embeddings)

    print("\nEmbedding variance (higher = more spread across contexts):")
    print("-" * 50)

    high_freq_vars = []
    for entity in high_freq_entities:
        embs = get_entity_embeddings(entity, contexts)
        var = np.var(embs, axis=0).mean()
        high_freq_vars.append(var)
        print(f"[High-freq] {entity}: {var:.6f}")

    low_freq_vars = []
    for entity in low_freq_entities:
        embs = get_entity_embeddings(entity, contexts)
        var = np.var(embs, axis=0).mean()
        low_freq_vars.append(var)
        print(f"[Low-freq]  {entity}: {var:.6f}")

    print()
    print(f"Avg variance (High-freq): {np.mean(high_freq_vars):.6f}")
    print(f"Avg variance (Low-freq):  {np.mean(low_freq_vars):.6f}")

    if np.mean(high_freq_vars) > np.mean(low_freq_vars):
        print(">>> High-freq entities show MORE variance across contexts (dilution)")
    else:
        print(">>> Low-freq entities show MORE variance across contexts")


if __name__ == "__main__":
    results, freq_analysis, type_analysis, calibration = run_experiment()

    # Run extended embedding analysis
    extended_analysis()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
