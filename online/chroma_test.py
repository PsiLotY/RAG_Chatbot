from chroma_functions import *

_chroma_client, _collection = init_chroma()
print(_chroma_client.list_collections())
entries = _collection.get(include=["documents", "metadatas"])
print(len(entries["ids"]))