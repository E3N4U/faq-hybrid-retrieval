from sentence_transformers import CrossEncoder
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)