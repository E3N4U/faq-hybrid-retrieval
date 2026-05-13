from pymilvus import connections

try:
    connections.connect(
        alias="default",
        host="127.0.0.1",
        port="19530"
    )
    print("✅ Connected to Milvus successfully!")
except Exception as e:
    print("❌ Failed to connect to Milvus:", e)