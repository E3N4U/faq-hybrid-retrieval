from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vec = model.encode(["test query"])

print("向量维度:", len(vec[0])) 