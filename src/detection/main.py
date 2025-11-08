import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detection.detection_module import HallucinationDetector, DetectionResult
from src.detection.gemini_integration import GeminiLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HallucinationAnalysisPipeline:
    def __init__(self):
        self.gemini_llm = None
        self.detector = None
        self._initialize_components()
    
    def _initialize_components(self):
        try:
            logger.info("Initializing Hallucination Analysis Pipeline...")
            self.gemini_llm = GeminiLLM()
            self.detector = HallucinationDetector()
            logger.info("Pipeline ready!")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    def generate_and_detect(self, question: str, evidence_docs: List[str]) -> Dict[str, Any]:
        logger.info(f"\nProcessing question: {question}")
        logger.info("Generating answer with Gemini Pro...")
        raw_answer = self.gemini_llm.generate_answer(question)
        logger.info(f"Generated answer: {raw_answer}")
        
        if raw_answer.startswith("Error:"):
            logger.error("Answer generation failed. Skipping hallucination detection.")
            return {
                'question': question,
                'raw_answer': raw_answer,
                'is_hallucination': True,
                'confidence_score': 1.0,
                'detection_method': 'generation_error',
                'details': {'error': raw_answer}
            }
        
        logger.info("Detecting hallucinations...")
        detection_result = self.detector.detect_hallucination(raw_answer, evidence_docs)
        
        result = detection_result.to_dict()
        result['question'] = question
        
        logger.info("\nDetection Results:")
        logger.info(f"  - Hallucination Detected: {result['is_hallucination']}")
        logger.info(f"  - Confidence Score: {result['confidence_score']:.3f}")
        logger.info(f"  - Detection Method: {result['detection_method']}")
        
        if result['is_hallucination']:
            logger.warning("Hallucination detected! Answer should be corrected.")
        else:
            logger.info("No hallucination detected. Answer appears reliable.")
            
        return result

def run_interactive_mode():
    logger.info("\n" + "=" * 60)
    logger.info("Entering Interactive Mode")
    logger.info("Type 'exit' or 'quit' at any prompt to end.")
    logger.info("=" * 60)
    
    pipeline = HallucinationAnalysisPipeline()
    
    while True:
        question = input("\nEnter a question: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        print("\nEnter a piece of evidence (or leave blank if none):")
        evidence_input = input("> ").strip()
        evidence_docs = [evidence_input] if evidence_input else []
        
        pipeline.generate_and_detect(question, evidence_docs)

def main():
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not found in environment variables.")
        logger.error("Please create a .env file with your Gemini API key.")
        return
    
    import argparse
    parser = argparse.ArgumentParser(description="Run the Detection Module.")
    parser.add_argument(
        'mode', 
        nargs='?', 
        default='interactive', 
        choices=['interactive']
    )
    args = parser.parse_args()

    if args.mode == 'interactive':
        run_interactive_mode()

if __name__ == "__main__":
    main()
