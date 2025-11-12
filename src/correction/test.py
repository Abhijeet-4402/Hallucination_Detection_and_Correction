#!/usr/bin/env python3
"""
Test file for the correction module.
This file contains test code that was extracted from correction_module.py
to prevent test execution when running the correction module.
"""

import os
from correction_module import initialize_database, correct_and_regenerate

def test_correction_module():
    """
    Test function for the correction module.
    Initializes database and runs example test cases.
    """
    # Initialize database
    initialize_database()
    print("Database 'hallucination_log.db' initialized and 'logs' table created.")

    # --- Unit Test / Example Usage ---
    print("\n--- Running Example Test ---")
    test_question = "Who discovered penicillin?"
    test_raw_answer = "Penicillin was discovered by Alexander Fleming."
    test_evidence = [
        "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital, London.",
        "Penicillium notatum is the mold from which penicillin is derived."
    ]

    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY not set. Skipping RAG and confidence score calculation.")
        # Provide a dummy response if API key is not set
        result = {
            "CorrectedAnswer": "(API Key not set) " + test_raw_answer,
            "Citations": ["No RAG (API Key missing)"],
            "ConfidenceScore": 0.0
        }
    else:
        result = correct_and_regenerate(test_question, test_raw_answer, test_evidence)

    print("\n--- Test Results ---")
    print(f"Original Question: {test_question}")
    print(f"Raw AI Answer: {test_raw_answer}")
    print(f"Corrected Answer: {result['CorrectedAnswer']}")
    print(f"Citations: {result['Citations']}")
    print(f"Confidence Score: {result['ConfidenceScore']}")

def test_database_operations():
    """
    Test database initialization and basic operations.
    """
    print("\n--- Testing Database Operations ---")
    initialize_database()
    print("Database initialization test completed successfully.")

if __name__ == "__main__":
    print("=== Correction Module Test Suite ===")
    
    # Test database operations
    test_database_operations()
    
    # Test main correction functionality
    test_correction_module()
    
    print("\n=== Test Suite Completed ===")
