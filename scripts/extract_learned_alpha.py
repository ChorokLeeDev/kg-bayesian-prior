#!/usr/bin/env python3
"""Extract learned alpha values from CAGP models across datasets and seeds."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import json

from scripts.run_wn18rr_temporal import CAGP, train_model, setup_device
from src.data.loaders import load_fb15k237, load_wn18rr


def main():
    device = setup_device()
    results = {}

    for name, load_fn in [('WN18RR', load_wn18rr), ('FB15k-237', load_fb15k237)]:
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print(f"{'='*50}")

        train_ds, _, test_ds = load_fn()
        train = train_ds.triples
        n_ent = train_ds.num_entities
        n_rel = train_ds.num_relations

        alphas = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = CAGP(n_ent, n_rel)
            model.precompute_coverage(train)
            model = train_model(model, train, device, epochs=30)

            alpha_logit = model.alpha.item()
            alpha = torch.sigmoid(model.alpha).item()
            print(f"  Seed {seed}: alpha_logit={alpha_logit:.4f}, alpha={alpha:.4f}")
            alphas.append(alpha)

        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas)
        print(f"  Mean alpha: {mean_alpha:.4f} ± {std_alpha:.4f}")
        results[name] = {
            'alphas': alphas,
            'mean': float(mean_alpha),
            'std': float(std_alpha),
        }

    outfile = project_root / 'outputs' / 'learned_alpha_values.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
