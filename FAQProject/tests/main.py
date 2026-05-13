from pymilvus import connections

connections.connect(alias="default", host="127.0.0.1", port="19530")