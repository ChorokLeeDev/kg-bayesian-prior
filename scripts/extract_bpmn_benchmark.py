#!/usr/bin/env python3
"""
Extract BPMN Process Flow subset from SAP BKG for OOD benchmark experiments.
Creates train/valid/test splits suitable for KGE evaluation.
"""

import sys
import json
import random
from collections import defaultdict
from pathlib import Path
from hdbcli import dbapi

sys.stdout.reconfigure(line_buffering=True)

# Connection
HOST = "c60683ef-658b-4083-ba90-367437e95a0d.hana.prod-eu12.hanacloud.ondemand.com"
PORT = 443
USER = "LEE_RO"
PASSWORD = "hPuFQqx39IBOEaebqkEN"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw" / "bkg_bpmn"

def sparql(conn, query):
    cursor = conn.cursor()
    sql = f"SELECT * FROM SPARQL_TABLE('{query}')"
    cursor.execute(sql)
    return cursor.fetchall()

def extract_bpmn_triples(conn):
    """Extract all BPMN-related triples."""
    print("Extracting BPMN triples...", flush=True)

    # Get all BPMN entities with their types
    print("  Fetching BPMN entities...", flush=True)
    entities = sparql(conn, """
        SELECT ?s ?type WHERE {
            ?s a ?type .
            FILTER(CONTAINS(STR(?type), "BBO") || CONTAINS(STR(?type), "sbbo"))
        }
    """)

    entity_set = set()
    entity_types = {}
    for e, t in entities:
        entity_set.add(e)
        entity_types[e] = t

    print(f"  Found {len(entity_set):,} BPMN entities", flush=True)

    # Get all triples where subject is a BPMN entity
    print("  Fetching BPMN triples (subject-based)...", flush=True)
    triples_subj = sparql(conn, """
        SELECT ?s ?p ?o WHERE {
            ?s a ?type .
            ?s ?p ?o .
            FILTER(CONTAINS(STR(?type), "BBO") || CONTAINS(STR(?type), "sbbo"))
            FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
        }
    """)
    print(f"  Found {len(triples_subj):,} triples", flush=True)

    return entity_set, entity_types, triples_subj

def create_entity_relation_mapping(triples, entity_set):
    """Create entity and relation ID mappings."""
    entities = set()
    relations = set()

    valid_triples = []
    for s, p, o in triples:
        # Only keep triples where both s and o are entities (URIs in our set or other URIs)
        if s in entity_set:
            entities.add(s)
            relations.add(p)
            # For object, check if it's a URI (not literal)
            if isinstance(o, str) and o.startswith("http"):
                entities.add(o)
                valid_triples.append((s, p, o))

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    return valid_triples, entity2id, relation2id

def create_splits(triples, entity2id, relation2id, train_ratio=0.8, valid_ratio=0.1):
    """Create train/valid/test splits."""
    random.seed(42)
    random.shuffle(triples)

    n = len(triples)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train = triples[:train_end]
    valid = triples[train_end:valid_end]
    test = triples[valid_end:]

    return train, valid, test

def save_dataset(output_dir, train, valid, test, entity2id, relation2id, entity_types):
    """Save dataset in standard KGE format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_triples(filepath, triples, e2id, r2id):
        with open(filepath, 'w') as f:
            for s, p, o in triples:
                if s in e2id and o in e2id and p in r2id:
                    f.write(f"{e2id[s]}\t{r2id[p]}\t{e2id[o]}\n")

    print(f"Saving to {output_dir}...", flush=True)

    write_triples(output_dir / "train.txt", train, entity2id, relation2id)
    write_triples(output_dir / "valid.txt", valid, entity2id, relation2id)
    write_triples(output_dir / "test.txt", test, entity2id, relation2id)

    # Save mappings
    with open(output_dir / "entity2id.json", 'w') as f:
        json.dump(entity2id, f, indent=2)

    with open(output_dir / "relation2id.json", 'w') as f:
        json.dump(relation2id, f, indent=2)

    # Save entity types for analysis
    type_mapping = {entity2id.get(e, -1): t for e, t in entity_types.items() if e in entity2id}
    with open(output_dir / "entity_types.json", 'w') as f:
        json.dump(type_mapping, f, indent=2)

    # Save stats
    stats = {
        "entities": len(entity2id),
        "relations": len(relation2id),
        "train_triples": len(train),
        "valid_triples": len(valid),
        "test_triples": len(test),
        "total_triples": len(train) + len(valid) + len(test)
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  Entities: {stats['entities']:,}", flush=True)
    print(f"  Relations: {stats['relations']:,}", flush=True)
    print(f"  Train: {stats['train_triples']:,}", flush=True)
    print(f"  Valid: {stats['valid_triples']:,}", flush=True)
    print(f"  Test: {stats['test_triples']:,}", flush=True)

    return stats

def main():
    print("Connecting to BKG...", flush=True)
    conn = dbapi.connect(
        address=HOST, port=PORT, user=USER, password=PASSWORD,
        encrypt=True, sslValidateCertificate=False
    )
    print("Connected!", flush=True)

    # Extract
    entity_set, entity_types, triples = extract_bpmn_triples(conn)

    # Process
    print("\nProcessing triples...", flush=True)
    valid_triples, entity2id, relation2id = create_entity_relation_mapping(triples, entity_set)
    print(f"  Valid entity-entity triples: {len(valid_triples):,}", flush=True)

    # Split
    print("\nCreating splits...", flush=True)
    train, valid, test = create_splits(valid_triples, entity2id, relation2id)

    # Save
    print("\nSaving dataset...", flush=True)
    stats = save_dataset(OUTPUT_DIR, train, valid, test, entity2id, relation2id, entity_types)

    conn.close()
    print(f"\nDone! Dataset saved to {OUTPUT_DIR}", flush=True)

    return stats

if __name__ == "__main__":
    main()
