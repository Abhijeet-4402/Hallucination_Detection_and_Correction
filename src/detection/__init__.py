"""
Detection Module
AI Hallucination Detection System

This module is responsible for:
1. Generating answers using Gemini Pro
2. Detecting hallucinations using semantic similarity and NLI
3. Providing detection results to other modules
"""

from .detection_module import  DetectionResult
from .gemini_integration import GeminiLLM
from .detection_module import HallucinationDetector

__all__ = ['DetectionResult', 'GeminiLLM', 'HallucinationDetector']
