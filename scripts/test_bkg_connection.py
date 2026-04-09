#!/usr/bin/env python3
"""Test connection to SAP Business Knowledge Graph (BKG) via HANA SPARQL endpoint."""

from hdbcli import dbapi

# Connection details (drop :443 suffix per Richard's email)
HOST = "c60683ef-658b-4083-ba90-367437e95a0d.hana.prod-eu12.hanacloud.ondemand.com"
PORT = 443
USER = "LEE_RO"
PASSWORD = "hPuFQqx39IBOEaebqkEN"

def test_connection():
    """Test basic connectivity."""
    print(f"Connecting to {HOST}:{PORT} as {USER}...")

    try:
        conn = dbapi.connect(
            address=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            encrypt=True,
            sslValidateCertificate=False
        )
        print("✓ Connection successful!")
        return conn
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None

def explore_schemas(conn):
    """List available schemas."""
    print("\n--- Available Schemas ---")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT SCHEMA_NAME
        FROM SYS.SCHEMAS
        WHERE SCHEMA_NAME NOT LIKE 'SYS%'
          AND SCHEMA_NAME NOT LIKE '_SYS%'
        ORDER BY SCHEMA_NAME
    """)
    schemas = cursor.fetchall()
    for schema in schemas[:20]:  # Limit output
        print(f"  {schema[0]}")
    if len(schemas) > 20:
        print(f"  ... and {len(schemas) - 20} more")
    return [s[0] for s in schemas]

def test_sparql_basic(conn):
    """Test basic SPARQL query using SPARQL_TABLE function."""
    print("\n--- Testing SPARQL Query ---")
    cursor = conn.cursor()

    # Basic SPARQL query wrapped in SQL
    query = """
    SELECT * FROM SPARQL_TABLE('
        SELECT ?s ?p ?o WHERE {?s ?p ?o}
        LIMIT 10
    ')
    """

    try:
        cursor.execute(query)
        results = cursor.fetchall()
        print(f"✓ SPARQL query successful! Got {len(results)} results:")
        for row in results:
            print(f"  {row}")
        return True
    except Exception as e:
        print(f"✗ SPARQL query failed: {e}")
        return False

def explore_graph_metadata(conn):
    """Try to get graph metadata - classes, predicates, counts."""
    print("\n--- Exploring Graph Metadata ---")
    cursor = conn.cursor()

    queries = {
        "Named Graphs": """
            SELECT * FROM SPARQL_TABLE('
                SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }
                LIMIT 20
            ')
        """,
        "Classes (types)": """
            SELECT * FROM SPARQL_TABLE('
                SELECT DISTINCT ?type (COUNT(?s) as ?count) WHERE {
                    ?s a ?type
                }
                GROUP BY ?type
                ORDER BY DESC(?count)
                LIMIT 20
            ')
        """,
        "Predicates": """
            SELECT * FROM SPARQL_TABLE('
                SELECT DISTINCT ?p (COUNT(*) as ?count) WHERE {
                    ?s ?p ?o
                }
                GROUP BY ?p
                ORDER BY DESC(?count)
                LIMIT 20
            ')
        """,
        "Total triples estimate": """
            SELECT * FROM SPARQL_TABLE('
                SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }
            ')
        """
    }

    for name, query in queries.items():
        print(f"\n{name}:")
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            for row in results:
                print(f"  {row}")
        except Exception as e:
            print(f"  Error: {e}")

def explore_domain_entities(conn):
    """Look for domain entities (not just schema metadata)."""
    print("\n--- Looking for Domain Entities ---")
    cursor = conn.cursor()

    queries = {
        "Business objects (non-schema types)": """
            SELECT * FROM SPARQL_TABLE('
                SELECT DISTINCT ?type (COUNT(?s) as ?count) WHERE {
                    ?s a ?type .
                    FILTER(!CONTAINS(STR(?type), "schema.sap.com/s4/Field"))
                    FILTER(!CONTAINS(STR(?type), "schema.sap.com/odata"))
                    FILTER(!CONTAINS(STR(?type), "schema.sap.com/cds"))
                    FILTER(!CONTAINS(STR(?type), "ForeignKey"))
                    FILTER(!CONTAINS(STR(?type), "DataElement"))
                    FILTER(!CONTAINS(STR(?type), "Domain"))
                    FILTER(!CONTAINS(STR(?type), "ABAPTable"))
                    FILTER(!CONTAINS(STR(?type), "Alias"))
                }
                GROUP BY ?type
                ORDER BY DESC(?count)
                LIMIT 30
            ')
        """,
        "Sample entities with labels": """
            SELECT * FROM SPARQL_TABLE('
                SELECT ?s ?label WHERE {
                    ?s <http://www.w3.org/2000/01/rdf-schema#label> ?label
                }
                LIMIT 20
            ')
        """,
        "Entity relationships (non-schema)": """
            SELECT * FROM SPARQL_TABLE('
                SELECT DISTINCT ?p (COUNT(*) as ?count) WHERE {
                    ?s ?p ?o .
                    FILTER(!CONTAINS(STR(?p), "schema.sap.com"))
                    FILTER(!CONTAINS(STR(?p), "w3.org/1999/02/22-rdf-syntax-ns"))
                }
                GROUP BY ?p
                ORDER BY DESC(?count)
                LIMIT 30
            ')
        """,
        "Named graph distribution": """
            SELECT * FROM SPARQL_TABLE('
                SELECT ?g (COUNT(*) as ?count) WHERE {
                    GRAPH ?g { ?s ?p ?o }
                }
                GROUP BY ?g
                ORDER BY DESC(?count)
                LIMIT 20
            ')
        """
    }

    for name, query in queries.items():
        print(f"\n{name}:")
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            for row in results:
                print(f"  {row}")
        except Exception as e:
            print(f"  Error: {e}")

def main():
    conn = test_connection()
    if not conn:
        return

    explore_schemas(conn)

    if test_sparql_basic(conn):
        explore_graph_metadata(conn)
        explore_domain_entities(conn)

    conn.close()
    print("\n--- Connection closed ---")

if __name__ == "__main__":
    main()
