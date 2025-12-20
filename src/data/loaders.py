"""
Data loaders for standard KG benchmark datasets.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from .kg_dataset import KGDataset


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def _download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return

    print(f"{desc}: {url}")

    def show_progress(block_num, block_size, total_size):
        pbar.update(block_size)

    with tqdm(total=None, unit='B', unit_scale=True, desc=dest.name) as pbar:
        urllib.request.urlretrieve(url, dest, show_progress)


def _load_triples(
    path: Path,
    entity_to_id: Optional[Dict[str, int]] = None,
    relation_to_id: Optional[Dict[str, int]] = None,
    sep: str = "\t",
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """
    Load triples from a TSV file.

    Returns:
        Tuple of (triples array, entity_to_id, relation_to_id)
    """
    df = pd.read_csv(path, sep=sep, header=None, names=["head", "relation", "tail"])

    # Build or update mappings
    if entity_to_id is None:
        entity_to_id = {}
    if relation_to_id is None:
        relation_to_id = {}

    all_entities = set(df["head"].unique()) | set(df["tail"].unique())
    all_relations = set(df["relation"].unique())

    for entity in all_entities:
        if entity not in entity_to_id:
            entity_to_id[entity] = len(entity_to_id)

    for relation in all_relations:
        if relation not in relation_to_id:
            relation_to_id[relation] = len(relation_to_id)

    # Convert to indices
    triples = np.array([
        [entity_to_id[h], relation_to_id[r], entity_to_id[t]]
        for h, r, t in zip(df["head"], df["relation"], df["tail"])
    ])

    return triples, entity_to_id, relation_to_id


def _download_fb15k237(data_dir: Path):
    """Download FB15k-237 dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master/FB15k-237"

    for split in ["train", "valid", "test"]:
        url = f"{base_url}/{split}.txt"
        dest = data_dir / f"{split}.txt"

        if not dest.exists():
            print(f"Downloading {split}.txt...")
            _download_file(url, dest, f"Downloading {split}")

    print("FB15k-237 download complete!")


def load_fb15k237(data_dir: Optional[Path] = None) -> Tuple[KGDataset, KGDataset, KGDataset]:
    """
    Load FB15k-237 dataset.

    FB15k-237 is a subset of Freebase containing 14,541 entities and 237 relations.
    It's a standard benchmark for knowledge graph embedding.

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "fb15k-237"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    # Download if needed
    if not train_path.exists():
        print("FB15k-237 not found. Downloading...")
        _download_fb15k237(data_dir)

    # Load all splits
    train_triples, entity_to_id, relation_to_id = _load_triples(train_path)
    valid_triples, entity_to_id, relation_to_id = _load_triples(
        valid_path, entity_to_id, relation_to_id
    )
    test_triples, entity_to_id, relation_to_id = _load_triples(
        test_path, entity_to_id, relation_to_id
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    train_dataset = KGDataset(
        triples=train_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    valid_dataset = KGDataset(
        triples=valid_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    test_dataset = KGDataset(
        triples=test_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )

    return train_dataset, valid_dataset, test_dataset


def _download_wn18rr(data_dir: Path):
    """Download WN18RR dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Primary source: DeepGraphLearning repo
    base_url = "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/wn18rr"

    for split in ["train", "valid", "test"]:
        url = f"{base_url}/{split}.txt"
        dest = data_dir / f"{split}.txt"

        if not dest.exists():
            print(f"Downloading {split}.txt...")
            _download_file(url, dest, f"Downloading {split}")

    print("WN18RR download complete!")


def load_wn18rr(data_dir: Optional[Path] = None) -> Tuple[KGDataset, KGDataset, KGDataset]:
    """
    Load WN18RR dataset.

    WN18RR is a subset of WordNet containing 40,943 entities and 11 relations.

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "wn18rr"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    if not train_path.exists():
        print("WN18RR not found. Downloading...")
        _download_wn18rr(data_dir)

    train_triples, entity_to_id, relation_to_id = _load_triples(train_path)
    valid_triples, entity_to_id, relation_to_id = _load_triples(
        valid_path, entity_to_id, relation_to_id
    )
    test_triples, entity_to_id, relation_to_id = _load_triples(
        test_path, entity_to_id, relation_to_id
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    return (
        KGDataset(train_triples, num_entities, num_relations, entity_to_id=entity_to_id, relation_to_id=relation_to_id),
        KGDataset(valid_triples, num_entities, num_relations, entity_to_id=entity_to_id, relation_to_id=relation_to_id),
        KGDataset(test_triples, num_entities, num_relations, entity_to_id=entity_to_id, relation_to_id=relation_to_id),
    )


def load_cn15k(data_dir: Optional[Path] = None) -> Tuple[KGDataset, KGDataset, KGDataset]:
    """
    Load CN15k dataset (ConceptNet with confidence scores).

    CN15k contains uncertain triples with confidence scores from ConceptNet.
    This is important for calibration experiments.

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "cn15k"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.txt"
    valid_path = data_dir / "valid.txt"
    test_path = data_dir / "test.txt"

    if not train_path.exists():
        print(f"Please download CN15k to {data_dir}")
        print("Download from: https://github.com/stasl0217/UKGE")
        _create_sample_uncertain_data(data_dir)

    def load_uncertain_triples(path, entity_to_id=None, relation_to_id=None):
        """Load triples with confidence scores."""
        if entity_to_id is None:
            entity_to_id = {}
        if relation_to_id is None:
            relation_to_id = {}

        triples = []
        confidences = []

        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    h, r, t, conf = parts
                    conf = float(conf)
                elif len(parts) == 3:
                    h, r, t = parts
                    conf = 1.0
                else:
                    continue

                if h not in entity_to_id:
                    entity_to_id[h] = len(entity_to_id)
                if t not in entity_to_id:
                    entity_to_id[t] = len(entity_to_id)
                if r not in relation_to_id:
                    relation_to_id[r] = len(relation_to_id)

                triples.append([entity_to_id[h], relation_to_id[r], entity_to_id[t]])
                confidences.append(conf)

        return np.array(triples), np.array(confidences), entity_to_id, relation_to_id

    train_triples, train_conf, entity_to_id, relation_to_id = load_uncertain_triples(train_path)
    valid_triples, valid_conf, entity_to_id, relation_to_id = load_uncertain_triples(
        valid_path, entity_to_id, relation_to_id
    )
    test_triples, test_conf, entity_to_id, relation_to_id = load_uncertain_triples(
        test_path, entity_to_id, relation_to_id
    )

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    return (
        KGDataset(train_triples, num_entities, num_relations, train_conf, entity_to_id, relation_to_id),
        KGDataset(valid_triples, num_entities, num_relations, valid_conf, entity_to_id, relation_to_id),
        KGDataset(test_triples, num_entities, num_relations, test_conf, entity_to_id, relation_to_id),
    )


def _download_yago310(data_dir: Path):
    """Download YAGO3-10 dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # YAGO3-10 from OpenKE
    base_url = "https://raw.githubusercontent.com/thunlp/OpenKE/OpenKE-PyTorch/benchmarks/YAGO3-10"

    for split in ["train", "valid", "test"]:
        # OpenKE uses train2id.txt format, we need to convert
        url = f"{base_url}/{split}2id.txt"
        dest = data_dir / f"{split}2id.txt"

        if not dest.exists():
            print(f"Downloading {split}2id.txt...")
            _download_file(url, dest, f"Downloading {split}")

    # Also download entity and relation mappings
    for mapping in ["entity2id", "relation2id"]:
        url = f"{base_url}/{mapping}.txt"
        dest = data_dir / f"{mapping}.txt"

        if not dest.exists():
            print(f"Downloading {mapping}.txt...")
            _download_file(url, dest, f"Downloading {mapping}")

    print("YAGO3-10 download complete!")


def load_yago310(data_dir: Optional[Path] = None) -> Tuple[KGDataset, KGDataset, KGDataset]:
    """
    Load YAGO3-10 dataset.

    YAGO3-10 is a subset of YAGO3 containing 123,182 entities and 37 relations.
    It has more relations than WN18RR but fewer than FB15k-237.

    This is a good middle-ground dataset for testing the hypothesis:
    "GP-KGE works better with more relations"

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)
    """
    if data_dir is None:
        data_dir = DATA_DIR / "yago3-10"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train2id.txt"

    if not train_path.exists():
        print("YAGO3-10 not found. Downloading...")
        _download_yago310(data_dir)

    def load_openke_format(path: Path) -> np.ndarray:
        """Load triples from OpenKE format (first line is count, rest are h t r)."""
        triples = []
        with open(path) as f:
            n = int(f.readline().strip())
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append([h, r, t])  # Convert to (h, r, t) format
        return np.array(triples)

    train_triples = load_openke_format(data_dir / "train2id.txt")
    valid_triples = load_openke_format(data_dir / "valid2id.txt")
    test_triples = load_openke_format(data_dir / "test2id.txt")

    # Load mappings
    with open(data_dir / "entity2id.txt") as f:
        num_entities = int(f.readline().strip())

    with open(data_dir / "relation2id.txt") as f:
        num_relations = int(f.readline().strip())

    return (
        KGDataset(train_triples, num_entities, num_relations),
        KGDataset(valid_triples, num_entities, num_relations),
        KGDataset(test_triples, num_entities, num_relations),
    )


def _create_sample_data(data_dir: Path):
    """Create sample data for testing when real data is not available."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic KG
    entities = [f"entity_{i}" for i in range(100)]
    relations = [f"relation_{i}" for i in range(10)]

    np.random.seed(42)

    def generate_triples(n):
        triples = []
        for _ in range(n):
            h = np.random.choice(entities)
            r = np.random.choice(relations)
            t = np.random.choice(entities)
            while t == h:
                t = np.random.choice(entities)
            triples.append(f"{h}\t{r}\t{t}")
        return "\n".join(triples)

    (data_dir / "train.txt").write_text(generate_triples(500))
    (data_dir / "valid.txt").write_text(generate_triples(50))
    (data_dir / "test.txt").write_text(generate_triples(50))

    print(f"Created sample data in {data_dir}")


def _create_sample_uncertain_data(data_dir: Path):
    """Create sample uncertain KG data for testing."""
    data_dir.mkdir(parents=True, exist_ok=True)

    entities = [f"entity_{i}" for i in range(100)]
    relations = [f"relation_{i}" for i in range(10)]

    np.random.seed(42)

    def generate_triples(n):
        triples = []
        for _ in range(n):
            h = np.random.choice(entities)
            r = np.random.choice(relations)
            t = np.random.choice(entities)
            while t == h:
                t = np.random.choice(entities)
            conf = np.random.uniform(0.5, 1.0)
            triples.append(f"{h}\t{r}\t{t}\t{conf:.3f}")
        return "\n".join(triples)

    (data_dir / "train.txt").write_text(generate_triples(500))
    (data_dir / "valid.txt").write_text(generate_triples(50))
    (data_dir / "test.txt").write_text(generate_triples(50))

    print(f"Created sample uncertain data in {data_dir}")
