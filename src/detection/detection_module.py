"""
Core Detection Logic for AI Hallucination Detection (Optimized with Batching)

This module implements a robust, claim-level, three-step hallucination detection process.
NLI inference is now batched for significant performance improvement.
"""

# FIX: This import MUST be at the top of the file.
# It solves the NameError by letting Python handle type hints more flexibly.
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch
from nltk.tokenize import sent_tokenize
import nltk

# Configure logging FIRST, so the logger is available for all subsequent calls.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download the necessary NLTK tokenizer models if they are not already present.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')


class DetectionResult:
    """Container for detection results with detailed analysis."""
    def __init__(self, is_hallucination: bool, confidence_score: float,
                 detection_method: str, raw_answer: str, evidence_docs: List[str],
                 details: Dict[str, Any] = None):
        self.is_hallucination = is_hallucination
        self.confidence_score = confidence_score
        self.detection_method = detection_method
        self.raw_answer = raw_answer
        self.evidence_docs = evidence_docs
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization."""
        return {
            'is_hallucination': self.is_hallucination,
            'confidence_score': self.confidence_score,
            'detection_method': self.detection_method,
            'raw_answer': self.raw_answer,
            'evidence_docs': self.evidence_docs,
            'details': self.details
        }

class HallucinationDetector:
    """Main class for hallucination detection with batched NLI inference."""

    def __init__(self,
                 similarity_model: str = "all-MiniLM-L6-v2",
                 nli_model: str = "roberta-large-mnli",
                 similarity_threshold: float = 0.5,
                 contradiction_threshold: float = 0.98,
                 entailment_threshold: float = 0.92,
                 device: str = "cpu"):

        self.similarity_model_name = similarity_model
        self.nli_model_name = nli_model
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.entailment_threshold = entailment_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._load_models(similarity_model, nli_model)

        self.CONTRADICTION_INDEX = 0
        self.ENTAILMENT_INDEX = 2

    def _load_models(self, similarity_model_name: str, nli_model_name: str):
        """Load the required ML models."""
        try:
            logger.info(f"Loading semantic similarity model: {similarity_model_name}")
            self.similarity_model = SentenceTransformer(similarity_model_name, device=self.device)

            logger.info(f"Loading NLI model: {nli_model_name}")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
            self.nli_model.eval() # Set model to evaluation mode

            logger.info("Models loaded successfully!")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def detect_hallucination(self, answer: str, evidence_docs: List[str]) -> DetectionResult:
        """
        Detects hallucination using an efficient batched NLI workflow.
        """
        if not answer.strip():
            return DetectionResult(False, 1.0, "empty_answer", answer, evidence_docs, details={"reason": "Answer was empty."})

        if not evidence_docs:
            return DetectionResult(True, 1.0, "no_evidence", answer, [], details={"reason": "No evidence documents were provided."})

        answer_claims = sent_tokenize(answer)
        all_evidence_sentences = [sent for doc in evidence_docs for sent in sent_tokenize(doc) if sent.strip()]

        if not all_evidence_sentences:
            return DetectionResult(True, 1.0, "no_evidence", answer, evidence_docs, details={"reason": "Evidence documents contain no text."})

        for claim in answer_claims:
            # --- Step 1 & 2: Batched NLI Check ---
            nli_pairs = [(evidence_sent, claim) for evidence_sent in all_evidence_sentences]

            if not nli_pairs: continue

            tokenized_input = self.nli_tokenizer(nli_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)

            with torch.no_grad():
                logits = self.nli_model(**tokenized_input).logits

            all_probs = torch.softmax(logits, dim=1)
            entailment_probs = all_probs[:, self.ENTAILMENT_INDEX]
            contradiction_probs = all_probs[:, self.CONTRADICTION_INDEX]

            max_entailment_score, _ = torch.max(entailment_probs, dim=0)
            if max_entailment_score.item() > self.entailment_threshold:
                logger.info(f"Claim successfully verified by NLI entailment: '{claim}'")
                continue

            max_contradiction_score, max_contradiction_idx = torch.max(contradiction_probs, dim=0)
            if max_contradiction_score.item() > self.contradiction_threshold:
                details = {
                    "problem_claim": claim,
                    "contradictory_evidence": all_evidence_sentences[max_contradiction_idx],
                    "contradiction_score": max_contradiction_score.item(),
                }
                logger.warning(f"Contradiction detected for claim: '{claim}'")
                return DetectionResult(True, max_contradiction_score.item(), "contradiction", answer, evidence_docs, details)

            # --- Step 3: Fallback to Semantic Similarity ---
            claim_embedding = self.similarity_model.encode(claim, convert_to_tensor=True)
            evidence_embeddings = self.similarity_model.encode(all_evidence_sentences, convert_to_tensor=True)

            similarity_scores = util.pytorch_cos_sim(claim_embedding, evidence_embeddings)[0]
            max_similarity_score = torch.max(similarity_scores).item()

            if max_similarity_score < self.similarity_threshold:
                details = {
                    "problem_claim": claim,
                    "reason": "Low similarity to all evidence.",
                    "max_similarity_score": max_similarity_score,
                    "closest_evidence": all_evidence_sentences[torch.argmax(similarity_scores).item()]
                }
                logger.warning(f"Unsupported claim due to low similarity: '{claim}'")
                return DetectionResult(True, 1 - max_similarity_score, "low_similarity", answer, evidence_docs, details)

            logger.info(f"Claim considered supported by similarity: '{claim}' (Score: {max_similarity_score:.3f})")

        details = {"reason": "All claims in the answer were successfully verified against the evidence."}
        logger.info("Answer verified. No hallucination detected.")
        return DetectionResult(False, 1.0, "verified", answer, evidence_docs, details)