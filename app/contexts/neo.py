from llama_index.graph_stores.neo4j import Neo4jGraphStore

graph_store = Neo4jGraphStore(
    username="neo4j",
    password="Kemil@1491",
    url="bolt://127.0.0.1:7687",   # ✅ use bolt
)