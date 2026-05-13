from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def load_faq(path=r"D:\FAQProject\data\faq.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def encode_faq_questions(faq_list):
    questions = [item["question"] for item in faq_list]
    vectors = model.encode(questions, convert_to_numpy=True)
    return vectors