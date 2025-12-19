#!/usr/bin/env python3
"""
Full Experiment: Uncertainty Quantification Methods Comparison
Run on Google Colab with A100 GPU (~30 minutes)

Usage (Colab):
    !git clone https://github.com/ChorokLeeDev/kg-bayesian-prior.git
    %cd kg-bayesian-prior
    !pip install -q torch-geometric gpytorch networkx pandas tqdm scikit-learn
    !python scripts/run_full_experiment.py
"""

import sys
sys.path.insert(0, '/content/kg-bayesian-prior')

import json
import gc
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.data import load_fb15k237
from src.models import DistMult, GPKGE
from src.models.ggpn import GGPN
from src.models.uncertain_kge import MCDropoutKGE
from src.utils.training import set_seed, NegativeSampler
from src.evaluation.calibration import expected_calibration_error, brier_score
from src.evaluation.ood_detection import compute_auroc, create_ood_dataset


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def evaluate(model, test_data, train_data, device, name, results):
    """Evaluate model on all metrics."""
    model.eval()

    # MRR
    sample = test_data.triples[np.random.choice(len(test_data), 2000, replace=False)]
    ranks = []
    with torch.no_grad():
        for i in range(0, len(sample), 200):
            batch = sample[i:i+200]
            h, r, t = [torch.tensor(batch[:,j], device=device) for j in range(3)]
            if hasattr(model, 'score_tails'):
                scores = model.score_tails(h, r)
            else:
                scores = torch.stack([
                    model(h[j].expand(test_data.num_entities),
                          r[j].expand(test_data.num_entities),
                          torch.arange(test_data.num_entities, device=device))
                    for j in range(len(h))
                ])
            target = scores[torch.arange(len(t), device=device), t]
            ranks.extend(((scores > target.unsqueeze(1)).sum(1) + 1).cpu().tolist())

    ranks = torch.tensor(ranks, dtype=torch.float)
    mrr = (1/ranks).mean().item()
    h1 = (ranks <= 1).float().mean().item()
    h3 = (ranks <= 3).float().mean().item()
    h10 = (ranks <= 10).float().mean().item()

    # ECE
    pos = test_data.triples[np.random.choice(len(test_data), 1000, replace=False)]
    neg = np.array([[h, r, np.random.randint(test_data.num_entities)] for h, r, t in pos])
    all_t = np.vstack([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    with torch.no_grad():
        h, r, t = [torch.tensor(all_t[:,j], device=device) for j in range(3)]
        if hasattr(model, 'base_model'):
            scores = model.base_model.score_triple(h, r, t)
        elif hasattr(model, 'score_triple'):
            scores = model.score_triple(h, r, t)
        else:
            scores = model(h, r, t)
        conf = torch.sigmoid(scores).cpu().numpy()

    ece, _ = expected_calibration_error(conf, labels)
    brier = brier_score(conf, labels)

    # AUROC
    id_t = test_data.triples[np.random.choice(len(test_data), 1000, replace=False)]
    ood_t = create_ood_dataset(train_data, test_data, "random", 1000)

    def get_unc(triples):
        with torch.no_grad():
            h, r, t = [torch.tensor(triples[:,j], device=device) for j in range(3)]
            if hasattr(model, 'predict_with_uncertainty'):
                pred = model.predict_with_uncertainty(h, r, t)
                return pred.get('total', pred.get('epistemic', torch.zeros(len(h)))).cpu().numpy()
            elif hasattr(model, 'predict_with_mc_samples'):
                _, var = model.predict_with_mc_samples(h, r, t, num_samples=10)
                return var.cpu().numpy()
            else:
                if hasattr(model, 'score_triple'):
                    s = model.score_triple(h, r, t)
                else:
                    s = model(h, r, t)
                p = torch.sigmoid(s)
                return (-p * torch.log(p + 1e-10) - (1-p) * torch.log(1-p + 1e-10)).cpu().numpy()

    auroc = compute_auroc(get_unc(id_t), get_unc(ood_t))

    results[name] = {
        "mrr": mrr, "hits@1": h1, "hits@3": h3, "hits@10": h10,
        "ece": ece, "brier": brier, "auroc": auroc
    }
    print(f"{name}: MRR={mrr:.4f}, ECE={ece:.4f}, AUROC={auroc:.4f}")
    return results[name]


def main():
    # Setup
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'None'}")

    train_data, valid_data, test_data = load_fb15k237()
    print(f"Data: {len(train_data):,} train, {len(test_data):,} test")

    results = {}
    neg_sampler = NegativeSampler(train_data.num_entities, num_negatives=10)
    neg_sampler.set_true_triples(train_data.triples)

    # === Model 1: DistMult ===
    print("\n" + "=" * 50)
    print("MODEL 1/4: DistMult")
    print("=" * 50)
    clear_mem()

    model = DistMult(train_data.num_entities, train_data.num_relations, embedding_dim=200).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for ep in (pbar := tqdm(range(50), desc="DistMult")):
        model.train()
        loss_sum, n = 0, 0
        for st in range(0, len(train_data), 1024):
            pos = torch.tensor(train_data.triples[st:st+1024], device=device)
            neg = neg_sampler(pos).to(device)
            opt.zero_grad()
            pos_s = model(pos[:,0], pos[:,1], pos[:,2])
            neg_s = model(neg[:,0], neg[:,1], neg[:,2]).view(len(pos), -1).mean(1)
            loss = F.relu(1 - pos_s + neg_s).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n += 1
        pbar.set_postfix(loss=f"{loss_sum/n:.4f}")

    evaluate(model, test_data, train_data, device, "DistMult", results)
    del model
    clear_mem()

    # === Model 2: MCDropout ===
    print("\n" + "=" * 50)
    print("MODEL 2/4: MCDropout")
    print("=" * 50)
    clear_mem()

    base = DistMult(train_data.num_entities, train_data.num_relations, embedding_dim=200, dropout=0.3)
    model = MCDropoutKGE(base, num_samples=20).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for ep in (pbar := tqdm(range(50), desc="MCDropout")):
        model.train()
        loss_sum, n = 0, 0
        for st in range(0, len(train_data), 1024):
            pos = torch.tensor(train_data.triples[st:st+1024], device=device)
            neg = neg_sampler(pos).to(device)
            opt.zero_grad()
            pos_s = model.base_model(pos[:,0], pos[:,1], pos[:,2])
            neg_s = model.base_model(neg[:,0], neg[:,1], neg[:,2]).view(len(pos), -1).mean(1)
            loss = F.relu(1 - pos_s + neg_s).mean()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            n += 1
        pbar.set_postfix(loss=f"{loss_sum/n:.4f}")

    evaluate(model, test_data, train_data, device, "DistMult+MCDropout", results)
    del model, base
    clear_mem()

    # === Model 3: GGPN ===
    print("\n" + "=" * 50)
    print("MODEL 3/4: GGPN")
    print("=" * 50)
    clear_mem()

    model = GGPN(
        train_data.num_entities,
        train_data.num_relations * 2,
        embedding_dim=50,
        hidden_dim=50,
        num_layers=1,
        num_rff=20
    ).to(device)
    model.set_graph(train_data)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for ep in (pbar := tqdm(range(50), desc="GGPN")):
        model.train()
        loss_sum, n = 0, 0
        for st in range(0, len(train_data), 512):
            pos = torch.tensor(train_data.triples[st:st+512], device=device)
            neg = neg_sampler(pos).to(device)
            opt.zero_grad()
            loss = model.loss(pos, neg)
            loss = loss['total'] if isinstance(loss, dict) else loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            n += 1
        pbar.set_postfix(loss=f"{loss_sum/n:.4f}")

    evaluate(model, test_data, train_data, device, "GGPN", results)
    del model
    clear_mem()

    # === Model 4: GP-KGE ===
    print("\n" + "=" * 50)
    print("MODEL 4/4: GP-KGE (Ours)")
    print("=" * 50)
    clear_mem()

    model = GPKGE(
        train_data.num_entities,
        train_data.num_relations,
        embedding_dim=100,
        kernel_type="rbf",
        num_inducing=200
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for ep in (pbar := tqdm(range(50), desc="GP-KGE")):
        model.train()
        loss_sum, n = 0, 0
        for st in range(0, len(train_data), 2048):
            pos = torch.tensor(train_data.triples[st:st+2048], device=device)
            neg = pos.clone()
            neg[:, 2] = torch.randint(0, train_data.num_entities, (len(pos),), device=device)
            opt.zero_grad()
            ps = model.score_triple(pos[:,0], pos[:,1], pos[:,2], use_mean=True)
            ns = model.score_triple(neg[:,0], neg[:,1], neg[:,2], use_mean=True)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([ps, ns]),
                torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            n += 1
        pbar.set_postfix(loss=f"{loss_sum/n:.4f}")

    evaluate(model, test_data, train_data, device, "GP-KGE (Ours)", results)
    del model
    clear_mem()

    # === FINAL RESULTS ===
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'MRR':>8} {'H@1':>8} {'H@10':>8} {'ECE':>8} {'AUROC':>8}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<20} {r['mrr']:>8.4f} {r['hits@1']:>8.4f} {r['hits@10']:>8.4f} {r['ece']:>8.4f} {r['auroc']:>8.4f}")

    if "GGPN" in results and "GP-KGE (Ours)" in results:
        imp = (results["GGPN"]["ece"] - results["GP-KGE (Ours)"]["ece"]) / results["GGPN"]["ece"] * 100
        print(f"\nCalibration Improvement over GGPN: {imp:.1f}%")

    # Save results
    with open('outputs/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to outputs/final_results.json")

    return results


if __name__ == "__main__":
    main()
