from flask import Flask, request, jsonify
import sys
import os
import logging

# Add the src directory to the system path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.main import HallucinationAnalysisPipeline
from src.retrieval.retrieval_module import retrieve_evidence
from src.correction.correction_module import correct_and_regenerate

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the pipeline
pipeline = HallucinationAnalysisPipeline()

# --- Main API route (POST only) ---
@app.route('/detect_hallucination', methods=['POST'])
def detect_hallucination():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_question = data['question']
    logger.info(f"Received question: {user_question}")

    try:
        # 1. Retrieve Evidence
        logger.info("Retrieving evidence...")
        evidence_docs = retrieve_evidence(user_question)
        
        # 2. Generate & Detect
        logger.info("Generating answer and detecting hallucinations...")
        detection_result = pipeline.generate_and_detect(user_question, evidence_docs)
        
        raw_answer = detection_result.get('raw_answer', "Error generating answer")
        is_hallucination = detection_result.get('is_hallucination', False)
        confidence_score = detection_result.get('confidence_score', 0.0)
        
        corrected_answer = "N/A"
        
        # 3. Correct if necessary
        citations = []
        if is_hallucination:
            logger.info("Hallucination detected. Correcting...")
            correction_result = correct_and_regenerate(user_question, raw_answer, evidence_docs)
            corrected_answer = correction_result.get('CorrectedAnswer', "Could not correct answer")
            citations = correction_result.get('Citations', [])
            # Update confidence score from correction if available, or keep detection score
            if correction_result.get('ConfidenceScore'):
                 confidence_score = correction_result.get('ConfidenceScore')
        else:
            corrected_answer = raw_answer # If no hallucination, the raw answer is the correct one
            
        return jsonify({
            "question": user_question,
            "raw_answer": raw_answer,
            "corrected_answer": corrected_answer,
            "confidence_score": confidence_score,
            "is_hallucination": is_hallucination,
            "citations": citations
        }), 200

    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello! Flask server is running and /hello route works."

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
