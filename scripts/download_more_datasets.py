#!/usr/bin/env python3
"""Download additional KG datasets for Coverage Paradox paper."""
import os
import subprocess
import functools
print = functools.partial(print, flush=True)

BASE = '/Users/i767700/Github/kg-bayesian-prior/data/raw'

def run(cmd):
    """Run shell command."""
    print(f"Running: {cmd[:80]}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def download_openbiolink():
    """Download OpenBioLink dataset."""
    print("\n" + "="*50)
    print("Downloading OpenBioLink...")
    out_dir = os.path.join(BASE, 'openbiolink')
    os.makedirs(out_dir, exist_ok=True)

    # Try Zenodo
    url = "https://zenodo.org/record/3834052/files/HQ_DIR.zip?download=1"
    ok, _, _ = run(f'cd {out_dir} && curl -L -o openbiolink.zip "{url}"')
    if ok:
        run(f'cd {out_dir} && unzip -o openbiolink.zip')
        print("OpenBioLink downloaded!")
        return True
    return False

def download_nell995():
    """Download NELL-995 dataset."""
    print("\n" + "="*50)
    print("Downloading NELL-995...")
    out_dir = os.path.join(BASE, 'nell-995')
    os.makedirs(out_dir, exist_ok=True)

    # Try direct URL
    url = "http://cs.ucsb.edu/~xwhan/datasets/NELL-995.zip"
    ok, _, _ = run(f'cd {out_dir} && curl -L -o nell995.zip "{url}" --max-time 60')
    if ok:
        run(f'cd {out_dir} && unzip -o nell995.zip')
        print("NELL-995 downloaded!")
        return True

    # Try GitHub mirror
    url2 = "https://github.com/xwhan/DeepPath/raw/master/NELL-995.zip"
    ok, _, _ = run(f'cd {out_dir} && curl -L -o nell995.zip "{url2}" --max-time 60')
    if ok:
        run(f'cd {out_dir} && unzip -o nell995.zip')
        print("NELL-995 downloaded from GitHub!")
        return True

    return False

def download_countries():
    """Download Countries dataset (small, for sanity check)."""
    print("\n" + "="*50)
    print("Downloading Countries...")
    out_dir = os.path.join(BASE, 'countries')
    os.makedirs(out_dir, exist_ok=True)

    # Clone from a known source
    ok, _, _ = run(f'cd {out_dir} && git clone --depth 1 https://github.com/TimDettmers/ConvE.git tmp_conve 2>/dev/null')
    if ok:
        run(f'cd {out_dir} && mv tmp_conve/data/countries/* . 2>/dev/null; rm -rf tmp_conve')
        print("Countries downloaded!")
        return True
    return False

def convert_drkg_to_standard():
    """Convert DRKG to standard train/test format."""
    print("\n" + "="*50)
    print("Converting DRKG to standard format...")

    drkg_path = os.path.join(BASE, 'biomedical/drkg.tsv')
    out_dir = os.path.join(BASE, 'drkg')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(drkg_path):
        print("DRKG not found")
        return False

    # Read and convert
    import random
    random.seed(42)

    triples = []
    with open(drkg_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))

    print(f"Total triples: {len(triples)}")

    # Shuffle and split
    random.shuffle(triples)
    n = len(triples)
    train = triples[:int(0.8*n)]
    valid = triples[int(0.8*n):int(0.9*n)]
    test = triples[int(0.9*n):]

    # Save
    for split, data in [('train', train), ('valid', valid), ('test', test)]:
        with open(os.path.join(out_dir, f'{split}.txt'), 'w') as f:
            for h, r, t in data:
                f.write(f'{h}\t{r}\t{t}\n')
        print(f"Saved {split}.txt: {len(data)} triples")

    return True

def main():
    print("="*60)
    print("DOWNLOADING ADDITIONAL DATASETS")
    print("="*60)

    results = {}

    # Try each download
    results['OpenBioLink'] = download_openbiolink()
    results['NELL-995'] = download_nell995()
    results['Countries'] = download_countries()
    results['DRKG (convert)'] = convert_drkg_to_standard()

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, ok in results.items():
        status = "SUCCESS" if ok else "FAILED"
        print(f"  {name}: {status}")

    # List final datasets
    print("\n" + "="*60)
    print("AVAILABLE DATASETS")
    print("="*60)
    for d in sorted(os.listdir(BASE)):
        path = os.path.join(BASE, d)
        if os.path.isdir(path):
            files = os.listdir(path)
            has_train = any('train' in f for f in files)
            print(f"  {d}: {'✓' if has_train else '○'} ({len(files)} files)")

if __name__ == '__main__':
    main()
