import logging
import argparse
import sys
import os
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.retrieval.retrieval_module import EvidenceRetriever, retrieve_evidence
from src.retrieval.dataset_loader import TruthfulQALoader
from src.retrieval.test_retrieval import run_all_tests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_retrieval():
    logger.info("Starting retrieval module demonstration...")
    sample_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When was the Declaration of Independence signed?",
        "What is the speed of light?"
    ]
    retriever = EvidenceRetriever(max_evidence_docs=3)
    for i, question in enumerate(sample_questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Question {i}: {question}")
        logger.info(f"{'='*60}")
        try:
            evidence_docs = retriever.retrieve_evidence(question)
            logger.info(f"Retrieved {len(evidence_docs)} evidence documents:")
            for j, doc in enumerate(evidence_docs, 1):
                logger.info(f"\nDocument {j}:")
                logger.info(f"{doc[:200]}..." if len(doc) > 200 else doc)
            evidence_with_scores = retriever.get_evidence_with_scores(question)
            logger.info(f"\nSimilarity Scores:")
            for j, result in enumerate(evidence_with_scores, 1):
                logger.info(f"Document {j}: {result['similarity_score']:.3f} (Relevant: {result['is_relevant']})")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
    stats = retriever.get_cache_stats()
    logger.info(f"\n{'='*60}")
    logger.info("Cache Statistics:")
    logger.info(f"{'='*60}")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

def demo_truthfulqa():
    logger.info("Starting TruthfulQA dataset demonstration...")
    try:
        loader = TruthfulQALoader()
        info = loader.get_dataset_info()
        logger.info(f"\nDataset Information:")
        logger.info(f"{'='*40}")
        for key, value in info.items():
            logger.info(f"{key}: {value}")
        samples = loader.get_sample_questions(num_samples=3)
        logger.info(f"\nSample Questions:")
        logger.info(f"{'='*40}")
        for i, sample in enumerate(samples, 1):
            logger.info(f"\nQuestion {i}:")
            logger.info(f"Question: {sample['question']}")
            logger.info(f"Best Answer: {sample['best_answer']}")
            logger.info(f"Category: {sample['category']}")
        categories = loader.get_all_categories()
        logger.info(f"\nAvailable Categories:")
        logger.info(f"{'='*40}")
        for category in categories[:10]:
            logger.info(f"- {category}")
        if len(categories) > 10:
            logger.info(f"... and {len(categories) - 10} more categories")
    except Exception as e:
        logger.error(f"Error with TruthfulQA demo: {e}")

def interactive_mode():
    logger.info("Starting interactive mode...")
    logger.info("Enter questions to retrieve evidence (type 'quit' to exit)")
    retriever = EvidenceRetriever(max_evidence_docs=3)
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode...")
                break
            if not question:
                logger.info("Please enter a valid question.")
                continue
            logger.info(f"\nRetrieving evidence for: {question}")
            evidence_docs = retriever.retrieve_evidence(question)
            if evidence_docs:
                logger.info(f"\nFound {len(evidence_docs)} evidence documents:")
                for i, doc in enumerate(evidence_docs, 1):
                    logger.info(f"\n--- Document {i} ---")
                    logger.info(doc[:300] + "..." if len(doc) > 300 else doc)
            else:
                logger.info("No evidence documents found.")
        except KeyboardInterrupt:
            logger.info("\nExiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Retrieval Module")
    parser.add_argument('--mode', choices=['demo', 'truthfulqa', 'interactive', 'test'], default='demo')
    parser.add_argument('--question', type=str)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logger.info("Retrieval Module - AI Hallucination Detection System")
    logger.info("="*70)
    try:
        if args.mode == 'demo':
            if args.question:
                logger.info(f"Processing question: {args.question}")
                evidence_docs = retrieve_evidence(args.question)
                logger.info(f"Retrieved {len(evidence_docs)} evidence documents:")
                for i, doc in enumerate(evidence_docs, 1):
                    logger.info(f"\nDocument {i}:")
                    logger.info(doc[:300] + "..." if len(doc) > 300 else doc)
            else:
                demo_retrieval()
        elif args.mode == 'truthfulqa':
            demo_truthfulqa()
        elif args.mode == 'interactive':
            interactive_mode()
        elif args.mode == 'test':
            success = run_all_tests()
            if not success:
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)
    logger.info("\nProgram completed successfully!")

if __name__ == "__main__":
    main()
