from elasticsearch import Elasticsearch


es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "081203")   # 替换成你的 ES 用户名和密码
)

def create_index(index_name="faq"):
    if es.indices.exists(index=index_name):
        return
    es.indices.create(index=index_name, mappings={
        "properties": {
            "id": {"type": "integer"},
            "question": {"type": "text"},
            "answer": {"type": "text"}
        }
    })

def insert_docs(docs, index_name="faq"):
    for doc in docs:
        es.index(index=index_name, id=doc["id"], document=doc)