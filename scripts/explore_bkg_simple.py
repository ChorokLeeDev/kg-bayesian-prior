#!/usr/bin/env python3
"""
Simplified BKG exploration with unbuffered output.
"""

import sys
from hdbcli import dbapi

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

HOST = "c60683ef-658b-4083-ba90-367437e95a0d.hana.prod-eu12.hanacloud.ondemand.com"
PORT = 443
USER = "LEE_RO"
PASSWORD = "hPuFQqx39IBOEaebqkEN"

def sparql(conn, query):
    cursor = conn.cursor()
    sql = f"SELECT * FROM SPARQL_TABLE('{query}')"
    cursor.execute(sql)
    return cursor.fetchall()

print("Connecting...", flush=True)
conn = dbapi.connect(
    address=HOST, port=PORT, user=USER, password=PASSWORD,
    encrypt=True, sslValidateCertificate=False
)
print("Connected!", flush=True)

# 1. BPMN Statistics
print("\n=== 1. BPMN PROCESS FLOWS ===", flush=True)

print("\nBPMN Entity Types:", flush=True)
results = sparql(conn, """
    SELECT ?type (COUNT(?s) as ?count) WHERE {
        ?s a ?type .
        FILTER(CONTAINS(STR(?type), "BBO") || CONTAINS(STR(?type), "sbbo"))
    }
    GROUP BY ?type ORDER BY DESC(?count) LIMIT 15
""")
bpmn_entities = 0
for r in results:
    name = r[0].split("#")[-1] if "#" in r[0] else r[0].split("/")[-1]
    count = int(r[1])
    bpmn_entities += count
    print(f"  {name}: {count:,}", flush=True)
print(f"  TOTAL BPMN entities: {bpmn_entities:,}", flush=True)

print("\nBPMN Relations:", flush=True)
results = sparql(conn, """
    SELECT DISTINCT ?p WHERE {
        ?s ?p ?o .
        ?s a ?type .
        FILTER(CONTAINS(STR(?type), "BBO"))
    } LIMIT 20
""")
bpmn_rels = []
for r in results:
    name = r[0].split("#")[-1] if "#" in r[0] else r[0].split("/")[-1]
    bpmn_rels.append(name)
    print(f"  {name}", flush=True)
print(f"  Total BPMN relation types: {len(bpmn_rels)}", flush=True)

# 2. Fiori Statistics
print("\n=== 2. FIORI APP CATALOG ===", flush=True)

print("\nFiori Entity Types:", flush=True)
results = sparql(conn, """
    SELECT ?type (COUNT(?s) as ?count) WHERE {
        ?s a ?type .
        FILTER(CONTAINS(STR(?type), "fiori"))
    }
    GROUP BY ?type ORDER BY DESC(?count)
""")
fiori_entities = 0
for r in results:
    name = r[0].split("/")[-1]
    count = int(r[1])
    fiori_entities += count
    print(f"  {name}: {count:,}", flush=True)
print(f"  TOTAL Fiori entities: {fiori_entities:,}", flush=True)

# 3. Check for temporal/version metadata
print("\n=== 3. SCHEMA EVOLUTION (Temporal Metadata) ===", flush=True)

print("\nLooking for version/date predicates:", flush=True)
results = sparql(conn, """
    SELECT DISTINCT ?p WHERE {
        ?s ?p ?o .
        FILTER(
            CONTAINS(LCASE(STR(?p)), "version") ||
            CONTAINS(LCASE(STR(?p)), "release") ||
            CONTAINS(LCASE(STR(?p)), "created") ||
            CONTAINS(LCASE(STR(?p)), "modified")
        )
    } LIMIT 20
""")
temporal_preds = []
for r in results:
    temporal_preds.append(r[0])
    print(f"  {r[0]}", flush=True)

if temporal_preds:
    print("\nSample temporal values:", flush=True)
    for pred in temporal_preds[:3]:
        try:
            results = sparql(conn, f"""
                SELECT ?o WHERE {{ ?s <{pred}> ?o }} LIMIT 3
            """)
            for r in results:
                print(f"  {pred.split('/')[-1]}: {r[0]}", flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)

# 4. Named graph versioning
print("\nNamed graph versioning patterns:", flush=True)
results = sparql(conn, """
    SELECT DISTINCT ?g WHERE {
        GRAPH ?g { ?s ?p ?o }
    } LIMIT 100
""")
from collections import defaultdict
versions = defaultdict(list)
for r in results:
    g = r[0].split("/")[-1]
    if "_g" in g or "-g" in g:
        base = g.rsplit("_g", 1)[0] if "_g" in g else g.rsplit("-g", 1)[0]
        versions[base].append(g)

multi_version = [(k, v) for k, v in versions.items() if len(v) > 1]
print(f"  Graphs with multiple versions: {len(multi_version)}", flush=True)
for base, vs in sorted(multi_version, key=lambda x: -len(x[1]))[:5]:
    print(f"    {base}: {len(vs)} versions", flush=True)

# 5. Full statistics
print("\n=== 4. COMPARISON TABLE STATISTICS ===", flush=True)

print("\nTotal triples:", flush=True)
results = sparql(conn, "SELECT (COUNT(*) as ?c) WHERE { ?s ?p ?o }")
total = int(results[0][0])
print(f"  {total:,}", flush=True)

print("\nUnique relations:", flush=True)
results = sparql(conn, "SELECT (COUNT(DISTINCT ?p) as ?c) WHERE { ?s ?p ?o }")
rels = int(results[0][0])
print(f"  {rels:,}", flush=True)

print("\nEntity types:", flush=True)
results = sparql(conn, "SELECT (COUNT(DISTINCT ?t) as ?c) WHERE { ?s a ?t }")
types = int(results[0][0])
print(f"  {types:,}", flush=True)

# Print comparison table
print("\n=== COMPARISON TABLE ===", flush=True)
print("| Dataset    | Entities   | Relations | Triples     | Domain           |", flush=True)
print("|------------|------------|-----------|-------------|------------------|", flush=True)
print("| FB15k-237  |     14,541 |       237 |     310,116 | Freebase         |", flush=True)
print("| WN18RR     |     40,943 |        11 |      93,003 | WordNet          |", flush=True)
print("| YAGO3-10   |    123,182 |        37 |   1,089,040 | Wikipedia        |", flush=True)
print("| ICEWS14    |      7,128 |       230 |      90,730 | Events           |", flush=True)
print(f"| SAP-BKG    |        TBD |     {rels:,} |  {total:,} | Enterprise       |", flush=True)

# Summary
print("\n=== SUMMARY: OOD BENCHMARK POTENTIAL ===", flush=True)
print(f"""
1. BPMN Process Flows (~{bpmn_entities:,} entities):
   - Semantic relations: has_part, has_incoming, has_outgoing, sourceRef, targetRef
   - OOD scenario: Novel process patterns / unseen task sequences
   - Suitable for: Structural OOD detection

2. Fiori App Catalog (~{fiori_entities:,} entities):
   - Relations: app-to-catalog, app-to-intent, role-to-catalog
   - OOD scenario: New apps connecting to existing business objects
   - Suitable for: Entity-level OOD detection

3. Schema Evolution:
   - Temporal predicates found: {len(temporal_preds)}
   - Multi-version graphs: {len(multi_version)}
   - OOD scenario: S4/HANA release introduces new CDS views/fields
   - Suitable for: Temporal OOD splits

RECOMMENDATION:
   BPMN subset is most suitable for OOD benchmark:
   - Clear semantic structure (process flows)
   - ~100K entities (manageable size)
   - Natural OOD: new business processes appearing over time
""", flush=True)

conn.close()
print("\nDone!", flush=True)
