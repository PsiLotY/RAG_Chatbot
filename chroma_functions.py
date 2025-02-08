"""This module contains a ChromaDB class to help interact more easily and more customised with the ChromaDB database.
"""

import hashlib
from datetime import date

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from utils import logger

log = logger(__name__)


class ChromaDB:
    """A class to interact with the ChromaDB database.
    This is a singleton, so that it can be accessed anywhere in the code.
    """

    _instance = None  # Store the single instance of ChromaDB

    def __new__(
        cls, collection_name: str = "hdm_collection", storage_path: str = "./chroma_storage"
    ):
        """Creates a new ChromaDB instance if it does not exist. This is a singleton implementation.

        Args:
            collection_name (str, optional): The name of the collection to be chosen or created.
                                             Defaults to "hdm_collection".
            storage_path (str, optional): The path where the data of the database should be stored and accessed.
                                          Defaults to "./chroma_storage".

        Returns:
            ChromaDB: The single instance of ChromaDB
        """
        if cls._instance is None:
            cls._instance = super(ChromaDB, cls).__new__(cls)
            cls._instance._init_chroma(collection_name, storage_path)
        return cls._instance

    def _init_chroma(self, collection_name: str, storage_path: str):
        """Constructor for the ChromaDB class.

        Args:
            collection_name (str, optional): The name of the collection to be chosen or created.
                                             Defaults to "hdm_collection".
            storage_path (str, optional): The path where the data of the database should be stored and accessed.
                                          Defaults to "./chroma_storage".
        """
        self.client = chromadb.PersistentClient(path=storage_path)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={
                "description": "vectorstore containing hdm website content",
                "created": str(date.today()),
            },
        )
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v1"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        log.info("ChromaDB instance initialized.")

    def count_tokens(self, text: str) -> int:
        """Counts tokens using the DeepSeek tokenizer."""
        return len(self.tokenizer.encode(text, truncation=False))

    def save_data(self, url: str, title: str, content: str, doc_type: str):
        """Takes information about a document and saves it to the ChromaDB.

        Args:
            url (str): The URL where the document is from.
            title (str): The title of the document.
            content (str): The content of the document.
            doc_type (str): The type of the document.
        """
        metadata = {
            "title": title,
            "accessed": str(date.today()),
            "type": doc_type,
            "url": url,
            "token_count": self.count_tokens(content),
        }
        doc_id = f"{doc_type}-{hashlib.sha1(url.encode()).hexdigest()}"
        document_embedding = self.embedding_model.encode(
            f"document: {content}", convert_to_list=True
        )

        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id],
            embeddings=[document_embedding],
        )
        log.info("Document added to ChromaDB: %s", doc_id)

    def get_closest_document(self, query: str):
        """Encodes the query based on the embedding model and retrieves
        the closest document from the collection.

        Args:
            query (str): The query to search for in the collection.

        Returns:
            document (dict): The document, metadata, and distance of the closest document.
        """
        query = query.lower()
        query_embedding = self.embedding_model.encode(f"query: {query}", convert_to_tensor=True)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        log.info("Closest document found: %s", metadatas)
        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]
