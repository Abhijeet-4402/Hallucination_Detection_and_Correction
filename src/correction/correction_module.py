import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()


DATABASE_NAME = "hallucination_log.db"

def initialize_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            Question TEXT NOT NULL,
            RawAnswer TEXT NOT NULL,
            CorrectedAnswer TEXT,
            Citations TEXT,
            ConfidenceScore REAL
        )
    """)
    conn.commit()
    conn.close()

def calculate_confidence_score(corrected_answer: str, source_documents: list) -> float:
    if not source_documents:
        return 0.0

    model = SentenceTransformer('all-MiniLM-L6-v2')
    answer_embedding = model.encode(corrected_answer, convert_to_tensor=True)

    # Combine all source document content into a single string for embedding
    # This assumes that the combined semantic meaning of all sources is relevant
    # for confidence against the corrected answer.
    source_contents = [doc.page_content for doc in source_documents]
    if not source_contents: # Handle case where page_content might be empty
        return 0.0
    
    source_embeddings = model.encode(source_contents, convert_to_tensor=True)

    # Calculate cosine similarity between the answer and each source document
    cosine_scores = util.cos_sim(answer_embedding, source_embeddings)
    
    # Return the maximum similarity score as the confidence
    return round(float(cosine_scores.max()), 4)

def log_hallucination_data(question: str, raw_answer: str, corrected_answer: str, citations: list, confidence_score: float):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        # Convert citations list to a string for storage
        citations_str = "; ".join(citations)
        cursor.execute("""
            INSERT INTO logs (Question, RawAnswer, CorrectedAnswer, Citations, ConfidenceScore)
            VALUES (?, ?, ?, ?, ?)
        """, (question, raw_answer, corrected_answer, citations_str, confidence_score))
        conn.commit()
        print("Successfully logged data to database.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def correct_and_regenerate(question: str, raw_answer: str, evidence: list) -> dict:
    # Initialize Google Generative AI model
    # Make sure to set your GOOGLE_API_KEY as an environment variable
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # Placeholder for the retriever. In a real scenario, 'evidence' would be processed
    # by a retriever from Shubh's module to get relevant documents.
    # For now, we'll simulate a retriever by creating a simple one or adapting the 'evidence'.
    # This part needs to be properly integrated with Shubh's retrieval module.

    # For demonstration, let's assume 'evidence' is a list of strings directly usable.
    # In a real RAG setup, you'd convert this into Document objects and use a VectorStoreRetriever.
    
    # As per the task, we need to use RetrievalQA chain.
    # For this placeholder, we'll need a dummy retriever or adapt the input 'evidence'.
    # Since we don't have a live retriever, we will assume 'evidence' can be directly passed
    # as context. This will be refined when integrating with the actual retrieval module.
    
    # A basic approach for now: create a mock retriever or adapt the chain to use 'evidence' directly.
    # Let's create a very basic mock retriever for now to allow the chain to be set up.
    from langchain_core.retrievers import BaseRetriever
    class MockRetriever(BaseRetriever):
        def _get_relevant_documents(self, query):
            # In a real scenario, this would interact with ChromaDB/Wikipedia
            from langchain.docstore.document import Document
            return [Document(page_content=doc) for doc in evidence]

    mock_retriever = MockRetriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" all documents into the context
        retriever=mock_retriever, # This will be replaced by actual retriever
        return_source_documents=True # To extract citations later
    )

    # Regenerate the answer using the RAG chain
    # We'll use the original question to query the RAG chain
    response = qa_chain.invoke({"query": question})
    corrected_answer = response["result"]
    source_documents = response["source_documents"]

    # Extracting citations from source_documents
    citations_list = []
    for doc in source_documents:
        if doc.metadata and doc.metadata.get("source"):
            citations_list.append(doc.metadata.get("source"))
        else:
            # If no explicit source, use a truncated version of the content as a citation
            citations_list.append(doc.page_content[:50] + "...")
    citations = list(set(citations_list)) # Ensure unique citations
    
    # Calculate confidence score
    confidence_score = calculate_confidence_score(corrected_answer, source_documents)

    # Log the data to the database
    log_hallucination_data(question, raw_answer, corrected_answer, citations, confidence_score)

    print(f"Received question: {question}")
    print(f"Received raw answer: {raw_answer}")
    print(f"Received evidence: {evidence}")
    return {
        "CorrectedAnswer": corrected_answer,
        "Citations": citations,
        "ConfidenceScore": confidence_score
    }

if __name__ == "__main__":
    # This module is now designed to be imported and used by other modules.
    # For testing, please run the test.py file in this directory.
    print("Correction module loaded. Use test.py to run tests.")
