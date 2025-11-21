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

   
    source_contents = [doc.page_content for doc in source_documents]
    if not source_contents: 
        return 0.0
    
    source_embeddings = model.encode(source_contents, convert_to_tensor=True)

   
    cosine_scores = util.cos_sim(answer_embedding, source_embeddings)
    
   
    return round(float(cosine_scores.max()), 4)

def log_hallucination_data(question: str, raw_answer: str, corrected_answer: str, citations: list, confidence_score: float):
    if os.getenv("DISABLE_DB_LOGGING", "False").lower() == "true":
        print("Database logging disabled via environment variable.")
        return

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
       
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
   
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=os.getenv("GEMINI_API_KEY"))

   
    from langchain_core.retrievers import BaseRetriever
    class MockRetriever(BaseRetriever):
        def _get_relevant_documents(self, query):
           
            from langchain.docstore.document import Document
            return [Document(page_content=doc) for doc in evidence]

    mock_retriever = MockRetriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  
        retriever=mock_retriever, 
        return_source_documents=True
    )

    
    response = qa_chain.invoke({"query": question})
    corrected_answer = response["result"]
    source_documents = response["source_documents"]

    
    citations_list = []
    for doc in source_documents:
        if doc.metadata and doc.metadata.get("source"):
            citations_list.append(doc.metadata.get("source"))
        else:
            
            citations_list.append(doc.page_content[:50] + "...")
    citations = list(set(citations_list))
    
    
    confidence_score = calculate_confidence_score(corrected_answer, source_documents)

    
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
   
    print("Correction module loaded. Use test.py to run tests.")
