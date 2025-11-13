from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Simulated hallucination detection pipeline ---
def run_hallucination_detection_pipeline(question: str):
    print(f"Flask received question: {question}")
    
    if "capital" in question.lower():
        raw_answer = "The capital of Australia is Sydney, and the Prime Minister is Winnie the Pooh."
        corrected_answer = "The capital of Australia is Canberra, and the current Prime Minister is Anthony Albanese."
        confidence_score = 0.92
    else:
        raw_answer = "The quick brown fox jumps over the lazy dog."
        corrected_answer = "The fast brown fox leaps across the tired canine."
        confidence_score = 0.75
        
    return {
        "question": question,
        "raw_answer": raw_answer,
        "corrected_answer": corrected_answer,
        "confidence_score": confidence_score
    }

# --- Test route (GET) ---
@app.route('/hello', methods=['GET'])
def hello_world():
    return "Hello! Flask server is running and /hello route works."

# --- Main API route (POST only) ---
@app.route('/detect_hallucination', methods=['POST'])
def detect_hallucination():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    user_question = data['question']
    result = run_hallucination_detection_pipeline(user_question)
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
