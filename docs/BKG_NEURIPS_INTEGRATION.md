# SAP Business Knowledge Graph (BKG) - NeurIPS Paper Integration Analysis

## Overview

This document summarizes the exploration of SAP's Business Knowledge Graph (BKG) for potential inclusion in the NeurIPS 2026 paper on OOD detection in knowledge graphs.

**Access Details:**
- Host: `c60683ef-658b-4083-ba90-367437e95a0d.hana.prod-eu12.hanacloud.ondemand.com`
- User: `LEE_RO` (read-only access)
- Query Method: SPARQL wrapped in SQL via `SPARQL_TABLE()` function

**Connection Example:**
```python
from hdbcli import dbapi
conn = dbapi.connect(
    address=HOST, port=443, user=USER, password=PASSWORD,
    encrypt=True, sslValidateCertificate=False
)
cursor = conn.cursor()
cursor.execute("SELECT * FROM SPARQL_TABLE('SELECT ?s ?p ?o WHERE {?s ?p ?o} LIMIT 10')")
```

---

## 1. Graph Statistics Summary

### Full BKG
| Metric | Value |
|--------|-------|
| Total Triples | **90,599,025** |
| Unique Relations | 169 |
| Entity Types | 66 |
| Named Graphs | 7 multi-version families |

### Comparison with Standard Benchmarks

| Dataset | Entities | Relations | Triples | Domain |
|---------|----------|-----------|---------|--------|
| FB15k-237 | 14,541 | 237 | 310,116 | Freebase (general) |
| WN18RR | 40,943 | 11 | 93,003 | WordNet (lexical) |
| YAGO3-10 | 123,182 | 37 | 1,089,040 | Wikipedia (general) |
| ICEWS14 | 7,128 | 230 | 90,730 | Events (temporal) |
| **SAP-BKG (full)** | **~11M** | **169** | **90.6M** | Enterprise (schema) |
| **SAP-BKG-BPMN** | **93,600** | **11** | **284,722** | Enterprise (process) |

**Key Observation:** BKG is 290× larger than FB15k-237 and 83× larger than YAGO3-10 by triple count.

---

## 2. Potential Benchmark Subsets

### 2.1 BPMN Process Flows (Recommended) - EXTRACTED

**Extracted Dataset:** `data/raw/bkg_bpmn/`

| Split | Triples |
|-------|---------|
| Train | 227,777 |
| Valid | 28,472 |
| Test | 28,473 |
| **Total** | **284,722** |

**Statistics:**
- Total entities: **93,600**
- Relation types: **11**

| Entity Type | Count |
|-------------|-------|
| SequenceFlow | 20,630 |
| Task | 12,344 |
| Lane | 4,149 |
| EndNoneEvent | 2,316 |
| ExclusiveGateway | 2,022 |
| Pool | 1,947 |
| StartNoneEvent | 1,935 |
| BPMNDiagram | 1,568 |
| TextAnnotation | 1,442 |
| CollapsedSubprocess | 956 |

**Relations (extracted):**
| ID | Relation | Description |
|----|----------|-------------|
| 0 | solutionActivity | Links to business activity |
| 1 | solutionProcessFlowDiagramElement | Links to diagram element |
| 2 | displayName | Display name |
| 3 | label | RDFS label |
| 4 | seeAlso | RDFS seeAlso |
| 5 | has_incoming | Incoming control flow |
| 6 | has_outgoing | Outgoing control flow |
| 7 | has_part | Containment (parent→child) |
| 8 | has_sourceRef | Sequence flow source |
| 9 | has_targetRef | Sequence flow target |
| 10 | is_partOf | Containment (child→parent) |

**OOD Scenarios:**
1. **Structural OOD**: New process patterns (unseen task→gateway→task sequences)
2. **Entity OOD**: New task types or subprocess patterns
3. **Temporal OOD**: Processes added in new S4/HANA releases

**Why Suitable:**
- Clear semantic structure (process flows are graphs)
- Manageable size (~52K entities, similar to WN18RR)
- Natural OOD interpretation: "Can model detect novel business processes?"

### 2.2 Fiori App Catalog

**Statistics:**
- Total entities: **18,371**
- Relation types: ~10

| Entity Type | Count |
|-------------|-------|
| FioriApp | 6,968 |
| Intent | 5,731 |
| BusinessCatalog | 2,340 |
| FioriResource | 2,240 |
| BusinessRole | 509 |
| Page | 354 |
| Space | 229 |

**OOD Scenarios:**
1. New Fiori apps connecting to existing business objects
2. New intents/catalogs in release updates

**Why Less Suitable:**
- Smaller subset (18K entities)
- Relations are primarily app-to-catalog mappings (less semantic richness)

### 2.3 CDS View Schema

**Statistics:**
- CDS Views: **126,782**
- Fields: **5,405,032**
- Foreign Keys: **1,015,766**

**OOD Scenarios:**
- New S4/HANA release introduces new CDS views/fields
- Schema evolution over time

**Why Less Suitable:**
- Primarily metadata (field definitions, not semantic knowledge)
- Relations are structural (dataElement, foreignKey), not semantic

---

## 3. Schema Evolution Analysis

### Temporal Metadata Available

| Predicate | Sample Values | Coverage |
|-----------|---------------|----------|
| `odata/version` | "1.0" | OData entities |
| `fiori/releaseID` | "S35" | Fiori apps |
| `fiori/externalReleaseName` | "2508" | Fiori apps |
| `fiori/releaseName` | - | Fiori apps |

### Named Graph Versioning

Multiple version families detected:
- `csns`: 13 versions
- `odata2`: 13 versions  
- `cds_description`: 12 versions
- `cds_view`: 12 versions
- `odata4`: 6 versions

**Implication:** Can create temporal splits based on graph versions for OOD evaluation.

---

## 4. Recommended Paper Integration

### Option A: Full Evaluation Section (High Effort)

Add BKG as a third experimental domain alongside academic benchmarks:

```
Table X: OOD Detection on Enterprise Knowledge Graph

| Method | BPMN Temporal | BPMN Structural | Fiori New-App |
|--------|---------------|-----------------|---------------|
| Energy | ... | ... | ... |
| Coverage | ... | ... | ... |
| CAGP | ... | ... | ... |
```

**Effort:** ~2-3 weeks for data extraction, experiments, analysis

### Option B: Case Study Subsection (Medium Effort)

Add as "4.4 Enterprise KG Case Study" or in Appendix:

> "To validate our findings on real-world enterprise KGs, we evaluate on a subset of SAP's Business Knowledge Graph (BKG), containing 52K business process entities and 169 relation types. Unlike academic benchmarks, BKG exhibits..."

**Effort:** ~1 week

### Option C: Statistics Comparison Only (Low Effort)

Add BKG to comparison tables to show scale difference:

> "Enterprise KGs like SAP BKG contain 90M+ triples—290× larger than FB15k-237—making scalable OOD detection critical."

**Effort:** ~1 day (just add numbers to paper)

---

## 5. Data Extraction Scripts

### Extract BPMN Subset
```python
# scripts/extract_bpmn_benchmark.py
query = """
SELECT ?s ?p ?o WHERE {
    ?s a ?type .
    ?s ?p ?o .
    FILTER(CONTAINS(STR(?type), "BBO") || CONTAINS(STR(?type), "sbbo"))
}
"""
```

### Extract Fiori Subset
```python
# scripts/extract_fiori_benchmark.py
query = """
SELECT ?s ?p ?o WHERE {
    ?s a ?type .
    ?s ?p ?o .
    FILTER(CONTAINS(STR(?type), "fiori"))
}
"""
```

---

## 6. Caveats and Limitations

1. **Data Sensitivity**: BKG contains SAP schema information. Need to confirm with Richard Detlefs what can be published.

2. **Schema vs. Semantic KG**: BKG is primarily a metadata graph (CDS views, fields) rather than a semantic KG (people, places). May need to frame carefully in paper.

3. **Access Restrictions**: LEE_RO has SELECT-only access. Cannot modify or export large datasets without coordination.

4. **Temporal Ground Truth**: While version predicates exist, may not have clean temporal splits for OOD evaluation.

---

## 7. Next Steps

- [ ] Confirm data usage permissions with Richard Detlefs (AI4LBD team)
- [x] Extract BPMN subset to local files for experiments (`data/raw/bkg_bpmn/`)
- [ ] Create temporal/structural OOD splits (need version metadata per entity)
- [ ] Run baseline + CAGP experiments on BKG-BPMN
- [ ] Decide integration level (A/B/C) based on results

---

## References

- Contact: Richard Detlefs (richard.detlefs@sap.com), AI4LBD SE
- PassVault: 0000610704 (LEE_RO credentials)
- BKG Documentation: Internal SAP (coordinate with AI4LBD team)
