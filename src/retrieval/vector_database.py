import chromadb
from chromadb.config import Settings
import logging
from typing import List, Dict, Any, Optional
import hashlib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, collection_name: str = "evidence_documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Evidence documents for hallucination detection"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        if not documents:
            logger.warning("No documents provided to add")
            return []
        
        doc_ids = [hashlib.md5(doc.encode()).hexdigest() for doc in documents]
        
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in documents]
        
        try:
            logger.info(f"Upserting {len(documents)} documents to vector database")
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=doc_ids
            )
            logger.info("Documents upserted successfully")
            return doc_ids
        except Exception as e:
            logger.error(f"Error upserting documents: {e}")
            return []
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            count = self.collection.count()
            if count == 0:
                logger.warning("Search attempted on an empty collection.")
                return []
            
            effective_n_results = min(n_results, count)

            logger.info(f"Searching for {effective_n_results} similar documents to: {query[:100]}...")
            results = self.collection.query(
                query_texts=[query],
                n_results=effective_n_results
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'document': doc,
                        'distance': results['distances'][0][i] if results['distances'] else 0.0,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'id': results['ids'][0][i] if results['ids'] else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.collection.get(ids=[doc_id])
            if results['documents'] and results['documents'][0]:
                return {
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0] if results['metadatas'] else {},
                    'id': doc_id
                }
            return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Evidence documents for hallucination detection"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def update_document(self, doc_id: str, document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        try:
            update_data = {
                'ids': [doc_id],
                'documents': [document]
            }
            
            if metadata:
                update_data['metadatas'] = [metadata]
            
            self.collection.update(**update_data)
            logger.info(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
