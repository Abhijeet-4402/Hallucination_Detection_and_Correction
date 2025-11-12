# How to Use the Correction Module

This guide explains how to integrate and use the **Correction Module** in the AI Hallucination Detection System. This module is designed to regenerate answers using retrieval-augmented generation (RAG) and provide confidence scores with citations.

---

## 📋 Overview

The Correction Module takes a raw LLM answer and evidence documents, then regenerates a fact-based answer with citations and confidence scores.

**Input:** Question + Raw Answer + Evidence Documents  
**Output:** Corrected Answer + Citations + Confidence Score + Database Log

---

## 🚀 Quick Start

### 1. Import the Module
```python
from correction_module import correct_and_regenerate, initialize_database
```

### 2. Initialize Database (First Time Only)
```python
initialize_database()  # Creates hallucination_log.db and logs table
```

### 3. Use the Main Function
```python
result = correct_and_regenerate(
    question="Who discovered penicillin?",
    raw_answer="Penicillin was discovered by Alexander Fleming.",
    evidence=[
        "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital, London.",
        "Penicillium notatum is the mold from which penicillin is derived."
    ]
)
```

---

## 📥 Input Parameters

### `correct_and_regenerate(question, raw_answer, evidence)`

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `question` | `str` | The original user question | `"Who discovered penicillin?"` |
| `raw_answer` | `str` | The raw answer from Gemini/LLM | `"Penicillin was discovered by Alexander Fleming."` |
| `evidence` | `list[str]` | List of evidence documents from retrieval module | `["Alexander Fleming discovered...", "Penicillium notatum..."]` |

### Input Requirements:
- ✅ **Question**: Must be a non-empty string
- ✅ **Raw Answer**: Must be a non-empty string  
- ✅ **Evidence**: Must be a list of strings (can be empty list)
- ✅ **API Key**: `GOOGLE_API_KEY` must be set in environment variables

---

## 📤 Output Format

The function returns a dictionary with the following structure:

```python
{
    "CorrectedAnswer": "Alexander Fleming discovered penicillin in 1928.",
    "Citations": [
        "Alexander Fleming discovered penicillin in 1928 at...",
        "Penicillium notatum is the mold from which penicil..."
    ],
    "ConfidenceScore": 0.8611
}
```

### Output Fields:

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `CorrectedAnswer` | `str` | Regenerated answer using RAG | Non-empty string |
| `Citations` | `list[str]` | List of source citations | List of strings |
| `ConfidenceScore` | `float` | Semantic similarity score | 0.0 to 1.0 |

---

## 🔧 Integration Examples

### Example 1: Basic Usage
```python
from correction_module import correct_and_regenerate

# Your inputs from other modules
question = "What is the capital of France?"
raw_answer = "The capital of France is Paris."
evidence = [
    "Paris is the capital and largest city of France.",
    "France is a country in Western Europe."
]

# Call the correction module
result = correct_and_regenerate(question, raw_answer, evidence)

# Use the results
print(f"Corrected Answer: {result['CorrectedAnswer']}")
print(f"Citations: {result['Citations']}")
print(f"Confidence: {result['ConfidenceScore']:.2%}")
```

### Example 2: Integration with Detection Module
```python
# After Abhijeet's detection module finds a hallucination
if detection_result["status"] == "hallucination_detected":
    # Get evidence from Shubh's retrieval module
    evidence = retrieval_result["documents"]
    
    # Use correction module
    corrected_result = correct_and_regenerate(
        question=user_question,
        raw_answer=llm_raw_answer,
        evidence=evidence
    )
    
    # Pass to Harshita's frontend module
    frontend_data = {
        "answer": corrected_result["CorrectedAnswer"],
        "citations": corrected_result["Citations"],
        "confidence": corrected_result["ConfidenceScore"]
    }
```

### Example 3: Batch Processing
```python
questions_and_answers = [
    {
        "question": "Who wrote Romeo and Juliet?",
        "raw_answer": "Shakespeare wrote Romeo and Juliet.",
        "evidence": ["William Shakespeare wrote Romeo and Juliet in 1597."]
    },
    {
        "question": "What is photosynthesis?",
        "raw_answer": "Photosynthesis is how plants make food.",
        "evidence": ["Photosynthesis is the process by which plants convert light energy into chemical energy."]
    }
]

results = []
for item in questions_and_answers:
    result = correct_and_regenerate(
        item["question"],
        item["raw_answer"], 
        item["evidence"]
    )
    results.append(result)
```

---

## 🗄️ Database Integration

### Automatic Logging
Every call to `correct_and_regenerate()` automatically logs data to `hallucination_log.db`:

```sql
-- Table: logs
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    Question TEXT NOT NULL,
    RawAnswer TEXT NOT NULL,
    CorrectedAnswer TEXT,
    Citations TEXT,
    ConfidenceScore REAL
);
```

### Querying Logs
```python
import sqlite3

# Connect to database
conn = sqlite3.connect('hallucination_log.db')
cursor = conn.cursor()

# Get all logs
cursor.execute("SELECT * FROM logs")
logs = cursor.fetchall()

# Get logs with high confidence
cursor.execute("SELECT * FROM logs WHERE ConfidenceScore > 0.8")
high_confidence_logs = cursor.fetchall()

conn.close()
```

---

## ⚠️ Error Handling

### Common Issues and Solutions:

#### 1. API Key Not Set
```python
# Error: GOOGLE_API_KEY not found
# Solution: Set environment variable
import os
os.environ["GOOGLE_API_KEY"] = "your_api_key_here"

# Or use .env file
from dotenv import load_dotenv
load_dotenv()
```

#### 2. Empty Evidence List
```python
# This is OK - module will handle gracefully
result = correct_and_regenerate(
    question="Test question",
    raw_answer="Test answer", 
    evidence=[]  # Empty list is fine
)
# Result: ConfidenceScore will be 0.0
```

#### 3. Database Connection Issues
```python
# If database file is locked or corrupted
# Solution: Delete hallucination_log.db and reinitialize
import os
if os.path.exists('hallucination_log.db'):
    os.remove('hallucination_log.db')
initialize_database()
```

---

## 🧪 Testing

### Unit Test Example
```python
def test_correction_module():
    # Test data
    question = "Who discovered penicillin?"
    raw_answer = "Penicillin was discovered by Alexander Fleming."
    evidence = [
        "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital, London.",
        "Penicillium notatum is the mold from which penicillin is derived."
    ]
    
    # Run correction
    result = correct_and_regenerate(question, raw_answer, evidence)
    
    # Assertions
    assert isinstance(result, dict)
    assert "CorrectedAnswer" in result
    assert "Citations" in result
    assert "ConfidenceScore" in result
    assert 0.0 <= result["ConfidenceScore"] <= 1.0
    assert len(result["Citations"]) > 0
    
    print("✅ All tests passed!")

# Run test
test_correction_module()
```

---

## 🔄 Integration Workflow

### Complete Pipeline Integration
```python
# 1. User asks question (Harshita's frontend)
user_question = "Who discovered penicillin?"

# 2. Get raw answer from LLM (Abhijeet's module)
raw_answer = llm_module.generate_answer(user_question)

# 3. Retrieve evidence (Shubh's module)
evidence_documents = retrieval_module.get_evidence(user_question)

# 4. Detect hallucination (Abhijeet's detection module)
detection_result = detection_module.check_hallucination(raw_answer, evidence_documents)

# 5. If hallucination detected, correct the answer (Your module)
if detection_result["needs_correction"]:
    corrected_result = correct_and_regenerate(
        question=user_question,
        raw_answer=raw_answer,
        evidence=evidence_documents
    )
    
    # 6. Display corrected answer (Harshita's frontend)
    final_answer = corrected_result["CorrectedAnswer"]
    citations = corrected_result["Citations"]
    confidence = corrected_result["ConfidenceScore"]
else:
    # Use original answer if no correction needed
    final_answer = raw_answer
    citations = []
    confidence = 1.0

# 7. Log everything (Manuradha's testing module)
testing_module.log_interaction(user_question, raw_answer, final_answer, citations, confidence)
```

---

## 📊 Performance Notes

### Expected Performance:
- **Processing Time**: 2-5 seconds per question (depends on evidence length)
- **Confidence Score Range**: 0.0 (no similarity) to 1.0 (perfect match)
- **Memory Usage**: ~200MB (due to sentence-transformers model)
- **Database Size**: ~1KB per logged interaction

### Optimization Tips:
- Cache the sentence-transformers model if processing multiple questions
- Use batch processing for multiple questions
- Monitor database size and clean old logs periodically

---

## 🆘 Troubleshooting

### Issue: "Model not found" error
**Solution**: Update the model name in `correction_module.py`:
```python
# Change this line:
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
```

### Issue: Low confidence scores
**Possible Causes**:
- Evidence documents are not relevant to the question
- Raw answer and evidence are semantically different
- Evidence list is empty

**Solutions**:
- Improve evidence retrieval quality
- Check if evidence documents match the question topic
- Ensure evidence list is not empty

### Issue: Database errors
**Solution**: Reinitialize the database:
```python
import os
if os.path.exists('hallucination_log.db'):
    os.remove('hallucination_log.db')
initialize_database()
```

---

## 📞 Support

If you encounter issues:
1. Check the error message carefully
2. Verify your API key is set correctly
3. Ensure all dependencies are installed
4. Check the database file permissions
5. Contact the module developer (Khushboo) for assistance

---

## 🔗 Related Files

- `correction_module.py` - Main module code
- `requirements.txt` - Dependencies
- `README.md` - Project overview
- `myTask.md` - Task specifications
- `.env.example` - Environment variables template
