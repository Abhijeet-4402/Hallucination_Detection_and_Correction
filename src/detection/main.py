"""
Main Integration Script for the Detection Module

This script demonstrates how to use the detection module, provides a static
demo with sample data, and includes an interactive mode for real-time testing.
"""

import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the src directory to the path
# Assumes this script is in a directory like 'src/detection/'
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detection.detection_module import HallucinationDetector, DetectionResult
from src.detection.gemini_integration import GeminiLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HallucinationAnalysisPipeline: # IMPROVEMENT: Renamed for clarity
    """Main pipeline class for the detection module"""
    
    def __init__(self):
        """Initialize the detection pipeline"""
        self.gemini_llm = None
        self.detector = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize Gemini LLM and detection components"""
        try:
            logger.info("Initializing Hallucination Analysis Pipeline...")
            self.gemini_llm = GeminiLLM()
            self.detector = HallucinationDetector()
            logger.info("🎉 Pipeline ready!")
        except Exception as e:
            logger.error(f"❌ Error initializing pipeline: {e}")
            raise
    
    def generate_and_detect(self, question: str, evidence_docs: List[str]) -> Dict[str, Any]:
        """
        Complete pipeline: Generate answer and detect hallucinations.
        """
        logger.info(f"\n🔍 Processing question: {question}")
        
        # Step 1: Generate answer using Gemini Pro
        logger.info("🤖 Generating answer with Gemini Pro...")
        raw_answer = self.gemini_llm.generate_answer(question)
        logger.info(f"📝 Generated answer: {raw_answer}")
        
        # **CRITICAL FIX**: Check if answer generation failed before proceeding.
        if raw_answer.startswith("Error:"):
            logger.error("Answer generation failed. Skipping hallucination detection.")
            return {
                'question': question,
                'raw_answer': raw_answer,
                'is_hallucination': True, # Treat generation failure as a form of failure/hallucination
                'confidence_score': 1.0,
                'detection_method': 'generation_error',
                'details': {'error': raw_answer}
            }
        
        # Step 2: Detect hallucinations
        logger.info("🔍 Detecting hallucinations...")
        detection_result = self.detector.detect_hallucination(raw_answer, evidence_docs)
        
        # Step 3: Prepare and print results
        # **IMPROVEMENT**: Simplified the result dictionary to avoid redundancy.
        result = detection_result.to_dict()
        result['question'] = question
        
        logger.info("\n📊 Detection Results:")
        logger.info(f"  - Hallucination Detected: {result['is_hallucination']}")
        logger.info(f"  - Confidence Score: {result['confidence_score']:.3f}")
        logger.info(f"  - Detection Method: {result['detection_method']}")
        
        if result['is_hallucination']:
            logger.warning("⚠️  Hallucination detected! Answer should be corrected.")
        else:
            logger.info("✅ No hallucination detected. Answer appears reliable.")
            
        return result

def run_interactive_mode():
    """**IMPROVEMENT**: Allows for real-time testing with user-provided questions."""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Entering Interactive Mode")
    logger.info("Type 'exit' or 'quit' at any prompt to end.")
    logger.info("=" * 60)
    
    pipeline = HallucinationAnalysisPipeline()
    
    while True:
        question = input("\nEnter a question: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        # In a real scenario, you'd call Member 1's retrieval module here.
        # For this demo, we'll use a placeholder or ask the user.
        print("\nEnter a piece of evidence (or leave blank if none):")
        evidence_input = input("> ").strip()
        evidence_docs = [evidence_input] if evidence_input else []
        
        pipeline.generate_and_detect(question, evidence_docs)

def main():
    """Main function to run the script."""
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("❌ GEMINI_API_KEY not found in environment variables.")
        logger.error("Please create a .env file with your Gemini API key.")
        return
    
    # Use argparse to select mode
    import argparse
    parser = argparse.ArgumentParser(description="Run the Detection Module.")
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='interactive', 
        choices=['interactive'],
        help="Mode to run the script in (default: interactive)."
    )
    args = parser.parse_args()

    if args.mode == 'interactive':
        run_interactive_mode()

if __name__ == "__main__":
    main()