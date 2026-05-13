from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

connections.connect("default", host="localhost", port="19530")

def create_milvus_collection(name="faq_vectors", dim=384):
    if name in [c.name for c in Collection.list()]:
        collection = Collection(name)
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="FAQ vectors")
    collection = Collection(name, schema)
    collection.create_index("embedding", {"index_type":"IVF_FLAT", "metric_type":"L2", "params":{"nlist":128}})
    return collection

def insert_vectors(collection, ids, vectors):
    collection.insert([ids, vectors])
    collection.load()