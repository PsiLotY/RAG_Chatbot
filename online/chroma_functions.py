import chromadb
import hashlib
import torch

from datetime import date
from sentence_transformers import SentenceTransformer

from utils import logger

log = logger(__name__)

# Singleton to store ChromaDB instance
_chroma_client = None
_collection = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def init_chroma(collection_name="hdm_website", storage_path="./chroma_storage"):
    """Initializes and returns a persistent ChromaDB client and collection.

    Args:
        collection_name (str): The name of the collection.
        storage_path (str): The directory where ChromaDB stores data.

    Returns:
        tuple: (chroma_client, collection) initialized once and reused globally.
    """
    global _chroma_client, _collection
    
    if _chroma_client is None or _collection is None:
        _chroma_client = chromadb.PersistentClient(path=storage_path)
        _collection = _chroma_client.get_or_create_collection(
            collection_name, 
            metadata={
            "   description": "vectorstore containing hdm website content",
                "created": str(date.today()),
            },
        )
        log.info("Successfully created Singleton instance of chroma.")

    return _chroma_client, _collection

def save_to_chromadb(url, title, content, doc_type):
    """Saves data directly to ChromaDB with metadata.

    Args:
        url (str): The URL of the document.
        title (str): The title of the document.
        content (str): The content of the document.
        doc_type (str): The type of document (e.g., 'webpage', 'pdf').
    """
    try:
        _, collection = init_chroma()

        metadata = {
            "title": title,
            "accessed": str(date.today()),
            "type": doc_type,
            "url": url,
        }
        doc_id = f"{doc_type}-{hashlib.sha1(url.encode()).hexdigest()}"
        embeddings = _embedding_model.encode(content, convert_to_list=True)

        collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[embeddings],
        )

        log.info(f"Document added to ChromaDB: {doc_id}")
    except Exception as e:
        log.error(f"Failed to save to ChromaDB for URL {url}: {e}")

def get_closest_document(prompt):
    """Returns the closest document to a given prompt.

    Args:
        prompt (str): The prompt to search for.
    """
    try:
        _, collection = init_chroma()
        prompt_embedding = _embedding_model.encode(prompt, convert_to_tensor=True)
        
        results = collection.query(
            query_embeddings=[prompt_embedding.tolist()], 
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        log.info(f"Closest document found: {metadatas}")
        return [{"document": doc, "metadata": meta, "distance": dist} for doc, meta, dist in zip(documents, metadatas, distances)]
    except Exception as e:
        log.error(f"Failed to get closest document: {e}")
        return None
        