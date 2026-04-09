#!/usr/bin/env python3
"""
Explore SAP BKG for potential NeurIPS paper usage.
1. Extract benchmark subset (BPMN process flows, Fiori apps)
2. Schema evolution analysis (temporal patterns)
3. Statistics for comparison table
"""

from hdbcli import dbapi
import json
from collections import defaultdict

# Connection details
HOST = "c60683ef-658b-4083-ba90-367437e95a0d.hana.prod-eu12.hanacloud.ondemand.com"
PORT = 443
USER = "LEE_RO"
PASSWORD = "hPuFQqx39IBOEaebqkEN"

def connect():
    return dbapi.connect(
        address=HOST, port=PORT, user=USER, password=PASSWORD,
        encrypt=True, sslValidateCertificate=False
    )

def sparql(conn, query):
    """Execute SPARQL query wrapped in SQL."""
    cursor = conn.cursor()
    sql = f"SELECT * FROM SPARQL_TABLE('{query}')"
    cursor.execute(sql)
    return cursor.fetchall()

# =============================================================================
# 1. BENCHMARK SUBSET EXTRACTION
# =============================================================================

def explore_bpmn_subset(conn):
    """Explore BPMN process flow data for benchmark extraction."""
    print("\n" + "="*80)
    print("1. BPMN PROCESS FLOW BENCHMARK SUBSET")
    print("="*80)

    # Get BPMN entity types and counts
    print("\n--- BPMN Entity Types ---")
    results = sparql(conn, """
        SELECT ?type (COUNT(?s) as ?count) WHERE {
            ?s a ?type .
            FILTER(CONTAINS(STR(?type), "BBO") || CONTAINS(STR(?type), "sbbo"))
        }
        GROUP BY ?type
        ORDER BY DESC(?count)
        LIMIT 20
    """)
    bpmn_stats = {}
    for row in results:
        type_name = row[0].split("#")[-1] if "#" in row[0] else row[0].split("/")[-1]
        bpmn_stats[type_name] = int(row[1])
        print(f"  {type_name}: {row[1]}")

    # Get BPMN relations
    print("\n--- BPMN Relations ---")
    results = sparql(conn, """
        SELECT ?p (COUNT(*) as ?count) WHERE {
            ?s ?p ?o .
            ?s a ?type .
            FILTER(CONTAINS(STR(?type), "BBO") || CONTAINS(STR(?type), "sbbo"))
        }
        GROUP BY ?p
        ORDER BY DESC(?count)
        LIMIT 20
    """)
    bpmn_relations = {}
    for row in results:
        pred_name = row[0].split("#")[-1] if "#" in row[0] else row[0].split("/")[-1]
        bpmn_relations[pred_name] = int(row[1])
        print(f"  {pred_name}: {row[1]}")

    # Sample BPMN triples
    print("\n--- Sample BPMN Triples ---")
    results = sparql(conn, """
        SELECT ?s ?p ?o WHERE {
            ?s a <https://www.irit.fr/recherches/MELODI/ontologies/BBO#Task> .
            ?s ?p ?o .
        }
        LIMIT 20
    """)
    for row in results[:10]:
        s = row[0].split("/")[-1][:40]
        p = row[1].split("#")[-1] if "#" in row[1] else row[1].split("/")[-1]
        o = str(row[2])[:40]
        print(f"  {s} -- {p} --> {o}")

    return bpmn_stats, bpmn_relations

def explore_fiori_subset(conn):
    """Explore Fiori app data for benchmark extraction."""
    print("\n" + "="*80)
    print("1b. FIORI APP BENCHMARK SUBSET")
    print("="*80)

    # Get Fiori entity types
    print("\n--- Fiori Entity Types ---")
    results = sparql(conn, """
        SELECT ?type (COUNT(?s) as ?count) WHERE {
            ?s a ?type .
            FILTER(CONTAINS(STR(?type), "fiori"))
        }
        GROUP BY ?type
        ORDER BY DESC(?count)
    """)
    fiori_stats = {}
    for row in results:
        type_name = row[0].split("/")[-1]
        fiori_stats[type_name] = int(row[1])
        print(f"  {type_name}: {row[1]}")

    # Get Fiori relations
    print("\n--- Fiori Relations ---")
    results = sparql(conn, """
        SELECT ?p (COUNT(*) as ?count) WHERE {
            ?s ?p ?o .
            ?s a ?type .
            FILTER(CONTAINS(STR(?type), "fiori"))
        }
        GROUP BY ?p
        ORDER BY DESC(?count)
        LIMIT 20
    """)
    fiori_relations = {}
    for row in results:
        pred_name = row[0].split("#")[-1] if "#" in row[0] else row[0].split("/")[-1]
        fiori_relations[pred_name] = int(row[1])
        print(f"  {pred_name}: {row[1]}")

    # Sample Fiori app relationships
    print("\n--- Sample Fiori App Triples ---")
    results = sparql(conn, """
        SELECT ?app ?p ?o WHERE {
            ?app a <http://schema.sap.com/fiori/FioriApp> .
            ?app ?p ?o .
            FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
        }
        LIMIT 30
    """)
    for row in results[:15]:
        app = row[0].split("/")[-1][:30]
        p = row[1].split("/")[-1]
        o = str(row[2])[:40]
        print(f"  {app} -- {p} --> {o}")

    return fiori_stats, fiori_relations

# =============================================================================
# 2. SCHEMA EVOLUTION ANALYSIS
# =============================================================================

def analyze_schema_evolution(conn):
    """Analyze temporal patterns in schema evolution."""
    print("\n" + "="*80)
    print("2. SCHEMA EVOLUTION ANALYSIS")
    print("="*80)

    # Check for timestamp/version metadata
    print("\n--- Looking for Temporal Metadata ---")
    results = sparql(conn, """
        SELECT DISTINCT ?p WHERE {
            ?s ?p ?o .
            FILTER(
                CONTAINS(LCASE(STR(?p)), "date") ||
                CONTAINS(LCASE(STR(?p)), "time") ||
                CONTAINS(LCASE(STR(?p)), "version") ||
                CONTAINS(LCASE(STR(?p)), "created") ||
                CONTAINS(LCASE(STR(?p)), "modified") ||
                CONTAINS(LCASE(STR(?p)), "release")
            )
        }
        LIMIT 30
    """)
    temporal_predicates = []
    for row in results:
        temporal_predicates.append(row[0])
        print(f"  {row[0]}")

    # Sample version/release info
    if temporal_predicates:
        print("\n--- Sample Temporal Values ---")
        for pred in temporal_predicates[:5]:
            try:
                results = sparql(conn, f"""
                    SELECT ?s ?o WHERE {{
                        ?s <{pred}> ?o
                    }}
                    LIMIT 5
                """)
                for row in results:
                    s = row[0].split("/")[-1][:30]
                    print(f"  {s}: {row[1]}")
            except:
                pass

    # Check named graph versioning
    print("\n--- Named Graph Versioning Patterns ---")
    results = sparql(conn, """
        SELECT DISTINCT ?g WHERE {
            GRAPH ?g { ?s ?p ?o }
        }
        LIMIT 50
    """)
    graph_versions = defaultdict(list)
    for row in results:
        g = row[0]
        # Extract version pattern (e.g., g1, g2, g3)
        parts = g.split("/")[-1]
        if "_g" in parts or "-g" in parts:
            base = parts.rsplit("_g", 1)[0] if "_g" in parts else parts.rsplit("-g", 1)[0]
            graph_versions[base].append(parts)

    print("  Graphs with multiple versions:")
    for base, versions in sorted(graph_versions.items(), key=lambda x: -len(x[1]))[:10]:
        if len(versions) > 1:
            print(f"    {base}: {sorted(versions)}")

    # CDS View evolution
    print("\n--- CDS View Naming Patterns (potential versions) ---")
    results = sparql(conn, """
        SELECT ?view WHERE {
            ?view a <http://schema.sap.com/cds/CDSView>
        }
        LIMIT 200
    """)
    view_families = defaultdict(list)
    for row in results:
        name = row[0].split("/")[-1]
        # Group by removing trailing numbers/versions
        import re
        base = re.sub(r'_?\d+$', '', name)
        if base != name:
            view_families[base].append(name)

    for base, views in sorted(view_families.items(), key=lambda x: -len(x[1]))[:10]:
        if len(views) > 1:
            print(f"    {base}: {sorted(views)[:5]}")

    return temporal_predicates, graph_versions

# =============================================================================
# 3. STATISTICS FOR COMPARISON TABLE
# =============================================================================

def compute_comparison_stats(conn):
    """Compute statistics for comparison with FB15k-237, YAGO."""
    print("\n" + "="*80)
    print("3. COMPARISON TABLE STATISTICS")
    print("="*80)

    stats = {}

    # Total triples
    print("\n--- Computing Statistics ---")
    results = sparql(conn, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
    stats['total_triples'] = int(results[0][0])
    print(f"  Total triples: {stats['total_triples']:,}")

    # Unique entities (subjects + objects that are URIs)
    results = sparql(conn, """
        SELECT (COUNT(DISTINCT ?e) as ?c) WHERE {
            { ?e ?p ?o } UNION { ?s ?p ?e }
            FILTER(isURI(?e))
        }
    """)
    stats['entities'] = int(results[0][0])
    print(f"  Unique entities: {stats['entities']:,}")

    # Unique relations
    results = sparql(conn, "SELECT (COUNT(DISTINCT ?p) as ?c) WHERE { ?s ?p ?o }")
    stats['relations'] = int(results[0][0])
    print(f"  Unique relations: {stats['relations']:,}")

    # Entity types
    results = sparql(conn, """
        SELECT (COUNT(DISTINCT ?t) as ?c) WHERE {
            ?s a ?t
        }
    """)
    stats['entity_types'] = int(results[0][0])
    print(f"  Entity types: {stats['entity_types']:,}")

    # Average degree
    stats['avg_degree'] = round(2 * stats['total_triples'] / max(stats['entities'], 1), 2)
    print(f"  Avg degree: {stats['avg_degree']}")

    # Relation distribution (for sparsity analysis)
    print("\n--- Relation Distribution ---")
    results = sparql(conn, """
        SELECT ?p (COUNT(*) as ?c) WHERE { ?s ?p ?o }
        GROUP BY ?p
        ORDER BY DESC(?c)
        LIMIT 10
    """)
    stats['top_relations'] = []
    for row in results:
        rel = row[0].split("#")[-1] if "#" in row[0] else row[0].split("/")[-1]
        count = int(row[1])
        pct = round(100 * count / stats['total_triples'], 2)
        stats['top_relations'].append((rel, count, pct))
        print(f"    {rel}: {count:,} ({pct}%)")

    return stats

def print_comparison_table(bkg_stats):
    """Print comparison table with standard benchmarks."""
    print("\n" + "="*80)
    print("COMPARISON TABLE (for paper)")
    print("="*80)

    # Standard benchmark stats
    benchmarks = {
        'FB15k-237': {
            'entities': 14541,
            'relations': 237,
            'train': 272115,
            'valid': 17535,
            'test': 20466,
            'total_triples': 310116,
            'domain': 'Freebase (general)'
        },
        'WN18RR': {
            'entities': 40943,
            'relations': 11,
            'train': 86835,
            'valid': 3034,
            'test': 3134,
            'total_triples': 93003,
            'domain': 'WordNet (lexical)'
        },
        'YAGO3-10': {
            'entities': 123182,
            'relations': 37,
            'train': 1079040,
            'valid': 5000,
            'test': 5000,
            'total_triples': 1089040,
            'domain': 'Wikipedia (general)'
        },
        'ICEWS14': {
            'entities': 7128,
            'relations': 230,
            'train': 72826,
            'valid': 8941,
            'test': 8963,
            'total_triples': 90730,
            'domain': 'Events (temporal)'
        },
        'SAP-BKG': {
            'entities': bkg_stats['entities'],
            'relations': bkg_stats['relations'],
            'total_triples': bkg_stats['total_triples'],
            'domain': 'Enterprise (schema)'
        }
    }

    print("\n| Dataset    | Entities   | Relations | Triples     | Domain           |")
    print("|------------|------------|-----------|-------------|------------------|")
    for name, s in benchmarks.items():
        print(f"| {name:<10} | {s['entities']:>10,} | {s['relations']:>9,} | {s['total_triples']:>11,} | {s['domain']:<16} |")

def main():
    print("Connecting to SAP BKG...")
    conn = connect()
    print("Connected!\n")

    # 1. Benchmark subset exploration
    bpmn_stats, bpmn_relations = explore_bpmn_subset(conn)
    fiori_stats, fiori_relations = explore_fiori_subset(conn)

    # 2. Schema evolution analysis
    temporal_preds, graph_versions = analyze_schema_evolution(conn)

    # 3. Comparison statistics
    bkg_stats = compute_comparison_stats(conn)
    print_comparison_table(bkg_stats)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: POTENTIAL BENCHMARK SUBSETS")
    print("="*80)

    bpmn_total = sum(bpmn_stats.values())
    fiori_total = sum(fiori_stats.values())

    print(f"""
    1. BPMN Process Flows:
       - Entities: ~{bpmn_total:,} (Tasks, SequenceFlows, Gateways, Events)
       - Relations: {len(bpmn_relations)} types (has_part, has_incoming, has_outgoing, etc.)
       - OOD scenario: New process patterns / unseen task sequences

    2. Fiori App Catalog:
       - Entities: ~{fiori_total:,} (Apps, Intents, Catalogs, Roles)
       - Relations: {len(fiori_relations)} types
       - OOD scenario: New apps connecting to existing business objects

    3. CDS View Schema:
       - Entities: ~127K CDS views, 5.4M fields
       - Relations: field mappings, foreign keys
       - OOD scenario: New S4/HANA release introduces new views/fields

    Recommendation: BPMN subset is most suitable for OOD benchmark because:
       - Clear semantic relations (process flows)
       - Manageable size for experiments
       - Natural OOD: new business processes
    """)

    conn.close()
    print("\nConnection closed.")

if __name__ == "__main__":
    main()
