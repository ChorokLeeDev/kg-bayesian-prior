#!/usr/bin/env python3
"""
Extended BERT Familiarity Trap Analysis

More balanced probe set focusing on entity linking scenarios.
Uses Wikipedia-derived factual knowledge with better frequency estimation.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import random

warnings.filterwarnings("ignore")

# Extended probes with actual Wikipedia frequency data approximation
# Format: (entity, template, answer, category, estimated_freq_tier)
# Tier 1-5: 1=very high (~10M+ pageviews), 5=low (~1K pageviews)
EXTENDED_PROBES = [
    # === PERSON - BIRTHPLACE (tests entity-location knowledge) ===
    # Very high freq people
    ("Barack Obama", "The birthplace of [X] is [MASK].", "Hawaii", "birthplace", 1),
    ("Donald Trump", "The birthplace of [X] is [MASK].", "York", "birthplace", 1),  # New York
    ("Elon Musk", "The birthplace of [X] is [MASK].", "Africa", "birthplace", 1),  # South Africa
    ("Taylor Swift", "The birthplace of [X] is [MASK].", "Pennsylvania", "birthplace", 1),

    # High freq people
    ("Albert Einstein", "The birthplace of [X] is [MASK].", "Germany", "birthplace", 2),
    ("Leonardo da Vinci", "The birthplace of [X] is [MASK].", "Italy", "birthplace", 2),
    ("William Shakespeare", "The birthplace of [X] is [MASK].", "England", "birthplace", 2),
    ("Napoleon Bonaparte", "The birthplace of [X] is [MASK].", "France", "birthplace", 2),  # Technically Corsica but French

    # Medium freq people
    ("Marie Curie", "The birthplace of [X] is [MASK].", "Poland", "birthplace", 3),
    ("Nikola Tesla", "The birthplace of [X] is [MASK].", "Croatia", "birthplace", 3),  # Then Austrian Empire
    ("Sigmund Freud", "The birthplace of [X] is [MASK].", "Austria", "birthplace", 3),
    ("Isaac Newton", "The birthplace of [X] is [MASK].", "England", "birthplace", 3),

    # Low freq people
    ("Niels Bohr", "The birthplace of [X] is [MASK].", "Denmark", "birthplace", 4),
    ("Max Planck", "The birthplace of [X] is [MASK].", "Germany", "birthplace", 4),
    ("Werner Heisenberg", "The birthplace of [X] is [MASK].", "Germany", "birthplace", 4),
    ("Gregor Mendel", "The birthplace of [X] is [MASK].", "Austria", "birthplace", 4),

    # Very low freq
    ("Enrico Fermi", "The birthplace of [X] is [MASK].", "Italy", "birthplace", 5),
    ("Paul Dirac", "The birthplace of [X] is [MASK].", "England", "birthplace", 5),
    ("Erwin Schrodinger", "The birthplace of [X] is [MASK].", "Austria", "birthplace", 5),
    ("Richard Feynman", "The birthplace of [X] is [MASK].", "York", "birthplace", 5),

    # === COMPANY - HEADQUARTERS (entity linking for organizations) ===
    # Very high freq companies
    ("Apple", "[X] is headquartered in [MASK].", "California", "headquarters", 1),
    ("Google", "[X] is headquartered in [MASK].", "California", "headquarters", 1),
    ("Amazon", "[X] is headquartered in [MASK].", "Seattle", "headquarters", 1),
    ("Microsoft", "[X] is headquartered in [MASK].", "Washington", "headquarters", 1),

    # High freq
    ("Tesla", "[X] is headquartered in [MASK].", "Texas", "headquarters", 2),  # Moved from CA
    ("Facebook", "[X] is headquartered in [MASK].", "California", "headquarters", 2),
    ("Netflix", "[X] is headquartered in [MASK].", "California", "headquarters", 2),
    ("IBM", "[X] is headquartered in [MASK].", "York", "headquarters", 2),

    # Medium freq
    ("Intel", "[X] is headquartered in [MASK].", "California", "headquarters", 3),
    ("Oracle", "[X] is headquartered in [MASK].", "Texas", "headquarters", 3),  # Moved
    ("Cisco", "[X] is headquartered in [MASK].", "California", "headquarters", 3),
    ("Adobe", "[X] is headquartered in [MASK].", "California", "headquarters", 3),

    # Low freq
    ("Nvidia", "[X] is headquartered in [MASK].", "California", "headquarters", 4),
    ("Salesforce", "[X] is headquartered in [MASK].", "California", "headquarters", 4),
    ("VMware", "[X] is headquartered in [MASK].", "California", "headquarters", 4),
    ("Qualcomm", "[X] is headquartered in [MASK].", "California", "headquarters", 4),

    # === PERSON - PROFESSION (tests person attribute knowledge) ===
    # Very high freq
    ("Einstein", "[X] worked as a [MASK].", "physicist", "profession", 1),
    ("Picasso", "[X] worked as a [MASK].", "painter", "profession", 1),
    ("Mozart", "[X] worked as a [MASK].", "composer", "profession", 1),
    ("Shakespeare", "[X] worked as a [MASK].", "playwright", "profession", 1),

    # High freq
    ("Darwin", "[X] worked as a [MASK].", "biologist", "profession", 2),
    ("Beethoven", "[X] worked as a [MASK].", "composer", "profession", 2),
    ("Michelangelo", "[X] worked as a [MASK].", "artist", "profession", 2),
    ("Galileo", "[X] worked as a [MASK].", "astronomer", "profession", 2),

    # Medium freq
    ("Pasteur", "[X] worked as a [MASK].", "scientist", "profession", 3),
    ("Faraday", "[X] worked as a [MASK].", "scientist", "profession", 3),
    ("Turing", "[X] worked as a [MASK].", "mathematician", "profession", 3),
    ("Euler", "[X] worked as a [MASK].", "mathematician", "profession", 3),

    # Low freq
    ("Gauss", "[X] worked as a [MASK].", "mathematician", "profession", 4),
    ("Cauchy", "[X] worked as a [MASK].", "mathematician", "profession", 4),
    ("Lagrange", "[X] worked as a [MASK].", "mathematician", "profession", 4),
    ("Laplace", "[X] worked as a [MASK].", "mathematician", "profession", 4),

    # === COUNTRY - CAPITAL (baseline factual knowledge - BERT is good at this) ===
    # Very high freq countries
    ("France", "The capital of [X] is [MASK].", "Paris", "capital", 1),
    ("Japan", "The capital of [X] is [MASK].", "Tokyo", "capital", 1),
    ("Germany", "The capital of [X] is [MASK].", "Berlin", "capital", 1),
    ("China", "The capital of [X] is [MASK].", "Beijing", "capital", 1),

    # High freq
    ("Italy", "The capital of [X] is [MASK].", "Rome", "capital", 2),
    ("Russia", "The capital of [X] is [MASK].", "Moscow", "capital", 2),
    ("Spain", "The capital of [X] is [MASK].", "Madrid", "capital", 2),
    ("Brazil", "The capital of [X] is [MASK].", "Brasilia", "capital", 2),

    # Medium freq
    ("Egypt", "The capital of [X] is [MASK].", "Cairo", "capital", 3),
    ("Thailand", "The capital of [X] is [MASK].", "Bangkok", "capital", 3),
    ("Poland", "The capital of [X] is [MASK].", "Warsaw", "capital", 3),
    ("Sweden", "The capital of [X] is [MASK].", "Stockholm", "capital", 3),

    # Low freq
    ("Norway", "The capital of [X] is [MASK].", "Oslo", "capital", 4),
    ("Finland", "The capital of [X] is [MASK].", "Helsinki", "capital", 4),
    ("Denmark", "The capital of [X] is [MASK].", "Copenhagen", "capital", 4),
    ("Portugal", "The capital of [X] is [MASK].", "Lisbon", "capital", 4),

    # Very low freq
    ("Slovenia", "The capital of [X] is [MASK].", "Ljubljana", "capital", 5),
    ("Croatia", "The capital of [X] is [MASK].", "Zagreb", "capital", 5),
    ("Slovakia", "The capital of [X] is [MASK].", "Bratislava", "capital", 5),
    ("Latvia", "The capital of [X] is [MASK].", "Riga", "capital", 5),

    # === LANGUAGE - COUNTRY (many-to-one mapping, tests ambiguity) ===
    ("French", "The official language of France is [MASK].", "French", "language", 2),
    ("German", "The official language of Germany is [MASK].", "German", "language", 2),
    ("Japanese", "The official language of Japan is [MASK].", "Japanese", "language", 2),
    ("Italian", "The official language of Italy is [MASK].", "Italian", "language", 2),
    ("Portuguese", "The official language of Brazil is [MASK].", "Portuguese", "language", 3),
    ("Dutch", "The official language of Netherlands is [MASK].", "Dutch", "language", 3),
    ("Swedish", "The official language of Sweden is [MASK].", "Swedish", "language", 3),
    ("Polish", "The official language of Poland is [MASK].", "Polish", "language", 3),

    # === ALTERNATIVE TEMPLATES (test template sensitivity) ===
    # Using different phrasing for same facts
    ("Einstein", "[X] is known for being a [MASK].", "physicist", "profession_alt", 1),
    ("Picasso", "[X] is known for being a [MASK].", "painter", "profession_alt", 1),
    ("France", "[X]'s capital city is [MASK].", "Paris", "capital_alt", 1),
    ("Germany", "[X]'s capital city is [MASK].", "Berlin", "capital_alt", 1),
]


def load_bert_model():
    """Load BERT model and tokenizer for masked LM."""
    from transformers import BertTokenizer, BertForMaskedLM
    import torch

    print("Loading BERT-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = model.to(device)
    return model, tokenizer, device


def probe_bert(model, tokenizer, device, entity: str, template: str, expected: str, top_k: int = 10) -> Dict:
    """Probe BERT with factual query."""
    import torch

    sentence = template.replace("[X]", entity)
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    mask_idx = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    if len(mask_idx) == 0:
        return {"error": "No [MASK] token found", "entity": entity}

    mask_idx = mask_idx[0].item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_idx]
        probs = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tokenizer.decode([idx]).strip() for idx in top_indices]

    expected_lower = expected.lower()
    rank = -1
    for i, token in enumerate(top_tokens):
        if token.lower() == expected_lower or expected_lower in token.lower() or token.lower() in expected_lower:
            rank = i + 1
            break

    expected_tokens = tokenizer.encode(expected, add_special_tokens=False)
    if len(expected_tokens) == 1:
        expected_prob = probs[expected_tokens[0]].item()
    else:
        expected_prob = 0.0

    return {
        "entity": entity,
        "sentence": sentence,
        "expected": expected,
        "top1_token": top_tokens[0],
        "top1_prob": top_probs[0].item(),
        "top5_tokens": top_tokens[:5],
        "expected_prob": expected_prob,
        "rank_in_top10": rank,
        "correct_top1": rank == 1,
        "correct_top5": rank != -1 and rank <= 5,
        "correct_top10": rank != -1,
    }


def analyze_results(results: List[Dict], probes: List[Tuple]) -> Dict:
    """Comprehensive analysis of results."""

    # By frequency tier
    tier_results = defaultdict(list)
    for r, p in zip(results, probes):
        if "error" not in r:
            tier = p[4]
            tier_results[tier].append(r)

    tier_analysis = {}
    for tier in sorted(tier_results.keys()):
        data = tier_results[tier]
        n = len(data)
        if n == 0:
            continue
        acc1 = sum(1 for r in data if r["correct_top1"]) / n
        acc5 = sum(1 for r in data if r["correct_top5"]) / n
        avg_conf = np.mean([r["top1_prob"] for r in data])

        wrong = [r for r in data if not r["correct_top1"]]
        conf_wrong = np.mean([r["top1_prob"] for r in wrong]) if wrong else 0

        right = [r for r in data if r["correct_top1"]]
        conf_right = np.mean([r["top1_prob"] for r in right]) if right else 0

        tier_analysis[tier] = {
            "n": n,
            "acc@1": acc1,
            "acc@5": acc5,
            "avg_conf": avg_conf,
            "conf_wrong": conf_wrong,
            "conf_right": conf_right,
        }

    # By category
    cat_results = defaultdict(list)
    for r, p in zip(results, probes):
        if "error" not in r:
            cat = p[3]
            cat_results[cat].append(r)

    cat_analysis = {}
    for cat, data in cat_results.items():
        n = len(data)
        if n == 0:
            continue
        acc = sum(1 for r in data if r["correct_top1"]) / n
        conf = np.mean([r["top1_prob"] for r in data])
        cat_analysis[cat] = {"n": n, "acc@1": acc, "avg_conf": conf}

    # Cross-analysis: frequency within each category
    cross_analysis = defaultdict(lambda: defaultdict(list))
    for r, p in zip(results, probes):
        if "error" not in r:
            cat, tier = p[3], p[4]
            cross_analysis[cat][tier].append(r)

    cross_summary = {}
    for cat in cross_analysis:
        cross_summary[cat] = {}
        for tier in sorted(cross_analysis[cat].keys()):
            data = cross_analysis[cat][tier]
            n = len(data)
            if n > 0:
                acc = sum(1 for r in data if r["correct_top1"]) / n
                cross_summary[cat][f"tier_{tier}"] = {"n": n, "acc@1": acc}

    return {
        "by_tier": tier_analysis,
        "by_category": cat_analysis,
        "cross_analysis": cross_summary,
    }


def main():
    print("=" * 70)
    print("EXTENDED BERT FAMILIARITY TRAP ANALYSIS")
    print("=" * 70)
    print()

    model, tokenizer, device = load_bert_model()
    print()

    print(f"Running {len(EXTENDED_PROBES)} probes...")
    results = []

    for entity, template, answer, category, tier in EXTENDED_PROBES:
        r = probe_bert(model, tokenizer, device, entity, template, answer)
        r["category"] = category
        r["tier"] = tier
        results.append(r)

        status = "OK" if r.get("correct_top1") else "X "
        print(f"  [{status}] T{tier} {category[:8]:8s} {entity[:15]:15s} -> {r.get('top1_token', 'ERR'):10s} "
              f"(exp: {answer}, conf: {r.get('top1_prob', 0):.3f})")

    print()

    # Analysis
    analysis = analyze_results(results, EXTENDED_PROBES)

    # Summary
    valid = [r for r in results if "error" not in r]
    overall = sum(1 for r in valid if r["correct_top1"]) / len(valid)
    print(f"Overall Accuracy@1: {overall:.1%}")
    print()

    # By tier
    print("=" * 70)
    print("RESULTS BY FREQUENCY TIER")
    print("(Tier 1 = Very High Freq, Tier 5 = Low Freq)")
    print("=" * 70)
    print(f"{'Tier':<6} {'N':>4} {'Acc@1':>8} {'Acc@5':>8} {'Conf':>8} {'Conf|W':>8} {'Conf|R':>8}")
    print("-" * 70)

    for tier in sorted(analysis["by_tier"].keys()):
        a = analysis["by_tier"][tier]
        print(f"Tier {tier:<3} {a['n']:>4} {a['acc@1']:>7.1%} {a['acc@5']:>7.1%} "
              f"{a['avg_conf']:>7.3f} {a['conf_wrong']:>7.3f} {a['conf_right']:>7.3f}")

    # Test hypothesis
    high_tiers = [analysis["by_tier"][t] for t in analysis["by_tier"] if t <= 2]
    low_tiers = [analysis["by_tier"][t] for t in analysis["by_tier"] if t >= 4]

    if high_tiers and low_tiers:
        high_acc = np.average([t["acc@1"] for t in high_tiers], weights=[t["n"] for t in high_tiers])
        low_acc = np.average([t["acc@1"] for t in low_tiers], weights=[t["n"] for t in low_tiers])
        print()
        print(f"HIGH-FREQ (Tier 1-2) weighted accuracy: {high_acc:.1%}")
        print(f"LOW-FREQ (Tier 4-5) weighted accuracy: {low_acc:.1%}")
        print(f"DIFFERENCE: {(low_acc - high_acc)*100:+.1f}pp")

        if low_acc > high_acc + 0.05:
            print(">>> FAMILIARITY TRAP CONFIRMED: Low-freq entities more accurate")
        elif high_acc > low_acc + 0.05:
            print(">>> NO familiarity trap: High-freq entities more accurate")
        else:
            print(">>> MARGINAL DIFFERENCE")

    print()

    # By category
    print("=" * 70)
    print("RESULTS BY PROBE CATEGORY")
    print("=" * 70)
    print(f"{'Category':<15} {'N':>4} {'Acc@1':>8} {'Avg Conf':>10}")
    print("-" * 45)

    for cat in sorted(analysis["by_category"].keys()):
        a = analysis["by_category"][cat]
        print(f"{cat:<15} {a['n']:>4} {a['acc@1']:>7.1%} {a['avg_conf']:>9.3f}")

    print()

    # Cross analysis
    print("=" * 70)
    print("FREQUENCY EFFECT WITHIN CATEGORIES")
    print("=" * 70)

    for cat in ["birthplace", "profession", "capital", "headquarters"]:
        if cat in analysis["cross_analysis"]:
            cross = analysis["cross_analysis"][cat]
            print(f"\n{cat.upper()}:")
            for key in sorted(cross.keys()):
                tier_num = key.split("_")[1]
                data = cross[key]
                print(f"  Tier {tier_num}: N={data['n']}, Acc@1={data['acc@1']:.1%}")

    # Save results
    output_dir = Path("/Users/i767700/Github/kg-bayesian-prior/outputs")
    save_data = {
        "overall_accuracy": overall,
        "analysis": analysis,
        "raw_results": results,
    }

    with open(output_dir / "bert_familiarity_trap_extended.json", "w") as f:
        json.dump(save_data, f, indent=2, default=float)

    print()
    print(f"Results saved to: {output_dir / 'bert_familiarity_trap_extended.json'}")
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
