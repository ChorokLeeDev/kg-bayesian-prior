#!/usr/bin/env python3
"""
Fiori App Catalog에서 novel-context OOD 시나리오가 가능한지 탐색.
"""

import sys
from collections import defaultdict
from hdbcli import dbapi

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
print("Connected!\n", flush=True)

# 1. Fiori 관계 종류와 개수
print("=== 1. FIORI 관계 종류 ===", flush=True)
results = sparql(conn, """
    SELECT ?p (COUNT(*) as ?count) WHERE {
        ?s ?p ?o .
        ?s a ?type .
        FILTER(CONTAINS(STR(?type), "fiori"))
        FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    }
    GROUP BY ?p
    ORDER BY DESC(?count)
""")
fiori_relations = []
for r in results:
    rel = r[0].split("/")[-1]
    count = int(r[1])
    fiori_relations.append((rel, count))
    print(f"  {rel}: {count:,}", flush=True)
print(f"\nTotal relation types: {len(fiori_relations)}", flush=True)

# 2. 릴리스별 분포
print("\n=== 2. 릴리스별 APP 분포 ===", flush=True)
results = sparql(conn, """
    SELECT ?release (COUNT(DISTINCT ?app) as ?count) WHERE {
        ?app a <http://schema.sap.com/fiori/FioriApp> .
        ?app <http://schema.sap.com/fiori/releaseID> ?release .
    }
    GROUP BY ?release
    ORDER BY ?release
""")
releases = []
for r in results:
    releases.append((r[0], int(r[1])))
    print(f"  Release {r[0]}: {r[1]} apps", flush=True)

# 3. App-Catalog 연결 패턴
print("\n=== 3. APP-CATALOG 연결 패턴 ===", flush=True)
results = sparql(conn, """
    SELECT ?app ?catalog WHERE {
        ?app a <http://schema.sap.com/fiori/FioriApp> .
        ?app ?rel ?catalog .
        ?catalog a <http://schema.sap.com/fiori/BusinessCatalog> .
    }
    LIMIT 50
""")
print(f"Sample App-Catalog connections ({len(results)} shown):", flush=True)
for r in results[:10]:
    app = r[0].split("/")[-1][:30]
    cat = r[1].split("/")[-1][:30]
    print(f"  {app} → {cat}", flush=True)

# 4. 하나의 App이 여러 Catalog에 연결되는 경우
print("\n=== 4. MULTI-CATALOG APPS (Novel-context 가능성) ===", flush=True)
results = sparql(conn, """
    SELECT ?app (COUNT(DISTINCT ?catalog) as ?catCount) WHERE {
        ?app a <http://schema.sap.com/fiori/FioriApp> .
        ?app ?rel ?catalog .
        ?catalog a <http://schema.sap.com/fiori/BusinessCatalog> .
    }
    GROUP BY ?app
    HAVING (COUNT(DISTINCT ?catalog) > 1)
    ORDER BY DESC(?catCount)
    LIMIT 20
""")
multi_catalog_apps = []
for r in results:
    app = r[0].split("/")[-1]
    count = int(r[1])
    multi_catalog_apps.append((app, count))
    print(f"  {app}: {count} catalogs", flush=True)
print(f"\nApps with multiple catalogs: {len(multi_catalog_apps)}", flush=True)

# 5. 릴리스별 App-Catalog 연결 변화 (핵심!)
print("\n=== 5. 릴리스별 APP-CATALOG 연결 (Novel-context 검증) ===", flush=True)
results = sparql(conn, """
    SELECT ?release ?app ?catalog WHERE {
        ?app a <http://schema.sap.com/fiori/FioriApp> .
        ?app <http://schema.sap.com/fiori/releaseID> ?release .
        ?app ?rel ?catalog .
        ?catalog a <http://schema.sap.com/fiori/BusinessCatalog> .
    }
    ORDER BY ?app ?release
""")

# App별로 릴리스에 따른 catalog 변화 추적
app_release_catalogs = defaultdict(lambda: defaultdict(set))
for r in results:
    release = r[0]
    app = r[1].split("/")[-1]
    catalog = r[2].split("/")[-1]
    app_release_catalogs[app][release].add(catalog)

# 릴리스 간 새로운 catalog 연결 찾기
novel_contexts = []
for app, release_cats in app_release_catalogs.items():
    sorted_releases = sorted(release_cats.keys())
    if len(sorted_releases) > 1:
        for i in range(1, len(sorted_releases)):
            prev_release = sorted_releases[i-1]
            curr_release = sorted_releases[i]
            prev_cats = release_cats[prev_release]
            curr_cats = release_cats[curr_release]
            new_cats = curr_cats - prev_cats
            if new_cats:
                novel_contexts.append((app, prev_release, curr_release, new_cats))

print(f"Apps with NEW catalog connections in later releases: {len(novel_contexts)}", flush=True)
for app, prev, curr, new_cats in novel_contexts[:10]:
    print(f"  {app}: {prev}→{curr}, new catalogs: {new_cats}", flush=True)

# 6. Summary
print("\n=== SUMMARY ===", flush=True)
print(f"""
Fiori App Catalog 분석 결과:

1. 관계 종류: {len(fiori_relations)}개
   - BPMN (11개)보다 다양할 수 있음

2. 릴리스: {len(releases)}개
   - Temporal split 가능

3. Multi-catalog apps: {len(multi_catalog_apps)}개
   - 하나의 App이 여러 Catalog에 연결됨

4. Novel-context 후보: {len(novel_contexts)}개
   - 기존 App이 새 릴리스에서 새 Catalog에 연결되는 경우
   - 이게 우리가 찾는 패턴!

결론: {
    "유효함 - Novel-context OOD 실험 가능" if len(novel_contexts) > 100
    else "제한적 - Novel-context 케이스가 적음 (" + str(len(novel_contexts)) + "개)"
}
""", flush=True)

conn.close()
print("Done!", flush=True)
