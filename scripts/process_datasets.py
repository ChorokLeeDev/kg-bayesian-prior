#!/usr/bin/env python3
"""
Process downloaded KG datasets and convert to standard (h, r, t) format.
Reports statistics: name, entities, relations, triples for each dataset.
"""

import os
import csv
import gzip
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("/Users/i767700/Github/kg-bayesian-prior/data/raw")


def load_tsv_triples(file_path, delimiter="\t", skip_header=False, cols=(0, 1, 2)):
    """Load triples from a TSV/CSV file."""
    triples = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if skip_header and i == 0:
                continue
            parts = line.strip().split(delimiter)
            if len(parts) >= max(cols) + 1:
                h, r, t = parts[cols[0]], parts[cols[1]], parts[cols[2]]
                triples.append((h, r, t))
    return triples


def get_stats(triples):
    """Get entity and relation counts."""
    entities = set()
    relations = set()
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return len(entities), len(relations), len(triples)


def save_triples(triples, output_path):
    """Save triples in tab-separated format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def process_standard_dataset(name, path, delimiter="\t"):
    """Process standard train/test/valid split datasets."""
    train_file = path / "train.txt"
    test_file = path / "test.txt"
    valid_file = path / "valid.txt"

    all_triples = []
    for f in [train_file, test_file, valid_file]:
        if f.exists():
            all_triples.extend(load_tsv_triples(f, delimiter))

    if not all_triples:
        return None

    # Save combined triples
    output_file = path / "all_triples.tsv"
    save_triples(all_triples, output_file)

    return get_stats(all_triples)


def process_openbiolink():
    """Process OpenBioLink HQ dataset."""
    path = DATA_DIR / "openbiolink" / "HQ_DIR" / "graph_files"
    edges_file = path / "edges.csv"

    if not edges_file.exists():
        return None

    triples = []
    with open(edges_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))

    output_path = DATA_DIR / "openbiolink"
    output_file = output_path / "all_triples.tsv"
    save_triples(triples, output_file)

    return get_stats(triples)


def process_hetionet():
    """Process Hetionet dataset."""
    path = DATA_DIR / "hetionet"
    edges_file = path / "edges.tsv"

    if not edges_file.exists():
        return None

    triples = []
    with open(edges_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))

    output_file = path / "all_triples.tsv"
    save_triples(triples, output_file)

    return get_stats(triples)


def process_drkg():
    """Process DRKG dataset."""
    path = DATA_DIR / "biomedical"
    drkg_file = path / "drkg.tsv"

    if not drkg_file.exists():
        return None

    triples = []
    with open(drkg_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                h, r, t = parts[0], parts[1], parts[2]
                triples.append((h, r, t))

    output_file = path / "drkg_triples.tsv"
    save_triples(triples, output_file)

    return get_stats(triples)


def process_pharmkg():
    """Process PharmKG dataset."""
    path = DATA_DIR / "pharmkg"

    # Try PharmKG-8k first
    pk8k_path = path / "PharmKG-8k"
    if pk8k_path.exists():
        all_triples = []
        for split in ["train.tsv", "test.tsv", "valid.tsv"]:
            split_file = pk8k_path / split
            if split_file.exists():
                all_triples.extend(load_tsv_triples(split_file))

        if all_triples:
            output_file = path / "all_triples.tsv"
            save_triples(all_triples, output_file)
            return get_stats(all_triples)

    # Try PharmKG.csv
    csv_file = path / "PharmKG.csv"
    if csv_file.exists():
        triples = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 5:
                    # Entity1_ID,Entity1_type,Entity2_ID,Entity2_type,relation
                    h, t, r = row[0], row[2], row[4]
                    triples.append((h, r, t))

        if triples:
            output_file = path / "all_triples.tsv"
            save_triples(triples, output_file)
            return get_stats(triples)

    return None


def process_primekg():
    """Process PrimeKG dataset."""
    path = DATA_DIR / "primekg"
    csv_file = path / "primekg.csv"

    if not csv_file.exists():
        return None

    triples = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                # Find column indices
                h_idx = header.index("x_id") if "x_id" in header else 0
                r_idx = header.index("relation") if "relation" in header else 1
                t_idx = header.index("y_id") if "y_id" in header else 2

                for row in reader:
                    if len(row) > max(h_idx, r_idx, t_idx):
                        triples.append((row[h_idx], row[r_idx], row[t_idx]))
    except Exception as e:
        print(f"  Error processing PrimeKG: {e}")
        return None

    if triples:
        output_file = path / "all_triples.tsv"
        save_triples(triples, output_file)
        return get_stats(triples)

    return None


def process_conceptnet():
    """Process ConceptNet 5.7 assertions."""
    path = DATA_DIR / "conceptnet"

    # Check if already processed
    output_file = path / "all_triples_en.tsv"
    if output_file.exists():
        triples = load_tsv_triples(output_file)
        if triples:
            return get_stats(triples)

    gz_file = path / "conceptnet.csv.gz"
    if not gz_file.exists():
        return None

    triples = []
    try:
        with gzip.open(gz_file, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    # Format: assertion_uri, relation, head, tail, metadata
                    r = parts[1].replace("/r/", "")
                    h = parts[2]
                    t = parts[3]
                    # Only keep English concepts for manageable size
                    if "/c/en/" in h and "/c/en/" in t:
                        triples.append((h, r, t))
    except Exception as e:
        print(f"  Error processing ConceptNet: {e}")
        return None

    if triples:
        output_file = path / "all_triples_en.tsv"
        save_triples(triples, output_file)
        return get_stats(triples)

    return None


def process_nell():
    """Process NELL dataset."""
    path = DATA_DIR / "nell"

    all_triples = []
    for split in ["train.txt", "test.txt", "valid.txt"]:
        split_file = path / split
        if split_file.exists():
            all_triples.extend(load_tsv_triples(split_file))

    if all_triples:
        output_file = path / "all_triples.tsv"
        save_triples(all_triples, output_file)
        return get_stats(all_triples)

    return None


def process_yago310():
    """Process YAGO3-10 with ID-based format."""
    path = DATA_DIR / "yago3-10"

    # Load entity and relation mappings
    entity_file = path / "entity2id.txt"
    relation_file = path / "relation2id.txt"

    if not entity_file.exists():
        return None

    # Load entity mapping
    id2entity = {}
    with open(entity_file, "r", encoding="utf-8") as f:
        next(f)  # Skip count line
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                id2entity[parts[1]] = parts[0]

    # Load relation mapping
    id2relation = {}
    with open(relation_file, "r", encoding="utf-8") as f:
        next(f)  # Skip count line
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                id2relation[parts[1]] = parts[0]

    # Load triples
    triples = []
    for split in ["train2id.txt", "test2id.txt", "valid2id.txt"]:
        split_file = path / split
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                next(f)  # Skip count line
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        h_id, t_id, r_id = parts[0], parts[1], parts[2]
                        h = id2entity.get(h_id, h_id)
                        t = id2entity.get(t_id, t_id)
                        r = id2relation.get(r_id, r_id)
                        triples.append((h, r, t))

    if triples:
        output_file = path / "all_triples.tsv"
        save_triples(triples, output_file)
        return get_stats(triples)

    return None


def process_fb15k():
    """Process FB15k dataset."""
    path = DATA_DIR / "fb15k"

    all_triples = []
    for split in ["train.txt", "test.txt", "valid.txt"]:
        split_file = path / split
        if split_file.exists():
            # FB15k may have different column order
            with open(split_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                # Check if first line is a count
                if len(parts) == 1 and parts[0].isdigit():
                    # Format: count on first line, then h t r
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            h, t, r = parts[0], parts[1], parts[2]
                            all_triples.append((h, r, t))
                else:
                    # Standard format
                    f.seek(0)
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            all_triples.append((parts[0], parts[1], parts[2]))

    if all_triples:
        output_file = path / "all_triples.tsv"
        save_triples(all_triples, output_file)
        return get_stats(all_triples)

    return None


def process_wn18():
    """Process WN18 dataset."""
    path = DATA_DIR / "wn18"

    all_triples = []
    for split in ["train.txt", "test.txt", "valid.txt"]:
        split_file = path / split
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                parts = first_line.split()
                if len(parts) == 1 and parts[0].isdigit():
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            h, t, r = parts[0], parts[1], parts[2]
                            all_triples.append((h, r, t))
                else:
                    f.seek(0)
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            all_triples.append((parts[0], parts[1], parts[2]))

    if all_triples:
        output_file = path / "all_triples.tsv"
        save_triples(all_triples, output_file)
        return get_stats(all_triples)

    return None


def process_gdelt():
    """Process GDELT temporal dataset (ID-based format)."""
    path = DATA_DIR / "gdelt"

    all_triples = []
    for split in ["train.txt", "test.txt", "valid.txt"]:
        split_file = path / split
        if split_file.exists():
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # Format: h t r timestamp...
                        h, t, r = parts[0], parts[1], parts[2]
                        all_triples.append((h, r, t))

    if all_triples:
        output_file = path / "all_triples.tsv"
        save_triples(all_triples, output_file)
        return get_stats(all_triples)

    return None


def main():
    results = []

    # Standard datasets with train/test/valid splits
    standard_datasets = [
        ("FB15k-237", DATA_DIR / "fb15k-237"),
        ("WN18RR", DATA_DIR / "wn18rr"),
        ("ICEWS14", DATA_DIR / "icews14"),
        ("ICEWS18", DATA_DIR / "icews18"),
        ("Kinship", DATA_DIR / "kinship"),
        ("UMLS", DATA_DIR / "umls"),
        ("Nations", DATA_DIR / "nations"),
        ("Countries_S1", DATA_DIR / "countries"),
        ("NELL", DATA_DIR / "nell"),
        ("CoDEx-S", DATA_DIR / "codex-s"),
        ("CoDEx-M", DATA_DIR / "codex-m"),
        ("CoDEx-L", DATA_DIR / "codex-l"),
    ]

    print("=" * 70)
    print("Processing Knowledge Graph Datasets")
    print("=" * 70)

    for name, path in standard_datasets:
        print(f"\nProcessing {name}...")
        if path.exists():
            stats = process_standard_dataset(name, path)
            if stats:
                entities, relations, triples = stats
                results.append((name, entities, relations, triples))
                print(f"  Entities: {entities:,}, Relations: {relations}, Triples: {triples:,}")
            else:
                print(f"  No data found")
        else:
            print(f"  Directory not found: {path}")

    # Special datasets
    special_processors = [
        ("OpenBioLink", process_openbiolink),
        ("Hetionet", process_hetionet),
        ("DRKG", process_drkg),
        ("PharmKG", process_pharmkg),
        ("PrimeKG", process_primekg),
        ("ConceptNet", process_conceptnet),
        ("YAGO3-10", process_yago310),
        ("FB15k", process_fb15k),
        ("WN18", process_wn18),
        ("GDELT", process_gdelt),
    ]

    for name, processor in special_processors:
        print(f"\nProcessing {name}...")
        stats = processor()
        if stats:
            entities, relations, triples = stats
            results.append((name, entities, relations, triples))
            print(f"  Entities: {entities:,}, Relations: {relations}, Triples: {triples:,}")
        else:
            print(f"  No data found or processing failed")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Entities':>15} {'Relations':>12} {'Triples':>15}")
    print("-" * 70)

    total_entities = 0
    total_relations = 0
    total_triples = 0

    for name, entities, relations, triples in sorted(results, key=lambda x: -x[3]):
        print(f"{name:<20} {entities:>15,} {relations:>12} {triples:>15,}")
        total_entities += entities
        total_relations += relations
        total_triples += triples

    print("-" * 70)
    print(f"{'TOTAL':<20} {total_entities:>15,} {total_relations:>12} {total_triples:>15,}")
    print(f"\nTotal datasets: {len(results)}")


if __name__ == "__main__":
    main()
