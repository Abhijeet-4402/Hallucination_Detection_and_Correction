"""
Fallback Gemini integration that uses the LangChain wrapper for consistency.
This version includes robust retry logic and configurable model selection.
"""

import os
import logging
import time
from typing import Optional
from dotenv import load_dotenv

# Import the LangChain component and specific exceptions for better error handling
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core import exceptions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiLLM:
    """Fallback Gemini LLM using the LangChain wrapper with enhanced robustness."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model_name: Optional[str] = None):
        """
        Initializes the Gemini LLM via LangChain.
        
        Args:
            api_key: The Google API key. Defaults to GEMINI_API_KEY from .env.
            model_name: The model to use (e.g., 'gemini-1.5-pro-latest'). Defaults to GEMINI_MODEL from .env or 'gemini-pro'.
        """
        logger.info("Initializing Gemini LLM via LangChain...")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-pro")
        self.model = None
        self.api_working = False
        self._initialize_llm()

    def _initialize_llm(self):
        """
        Configures the LangChain ChatGoogleGenerativeAI model.
        """
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found. Cannot initialize Gemini LLM.")
            return

        try:
            # Use the LangChain class to initialize the model with the configurable name
            self.model = ChatGoogleGenerativeAI(
                model=self.model_name, 
                google_api_key=self.api_key,
                convert_system_message_to_human=True
            )
            # A simple, low-cost API call to verify credentials and model access.
            self.model.invoke("test") 
            self.api_working = True
            logger.info(f"Gemini LLM ({self.model_name}) initialized successfully via LangChain.")

        except Exception as e:
            logger.warning(f"Gemini API is not available or failed to initialize: {e}")
            self.use_fallback(str(e))

    def use_fallback(self, reason: str):
        """Switches to fallback mode and logs the reason."""
        self.api_working = False
        logger.warning(f"Switching to fallback mode. Reason: {reason}")

    def generate_answer(self, question: str, max_retries: int = 3) -> str:
        """
        Generates an answer using the LangChain model with exponential backoff for retries.
        """
        if not self.api_working or not self.model:
            return "Error: Unable to access Gemini API. Please check your configuration and API key."

        last_exception = None
        for attempt in range(max_retries):
            try:
                # Use the .invoke() method for LangChain
                response = self.model.invoke(question)
                return response.content.strip()
            except exceptions.ResourceExhausted as e:
                wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                last_exception = e
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"An unexpected error occurred during content generation: {e}")
                last_exception = e
                break # Don't retry on unexpected errors

        error_message = f"Error: Failed to generate answer from API after {max_retries} retries."
        if last_exception:
            error_message += f" Last error: {last_exception}"
        return error_message

# A more robust singleton pattern
_gemini_llm_instance = None

def get_gemini_llm() -> GeminiLLM:
    """Singleton pattern to get the global Gemini LLM instance."""
    global _gemini_llm_instance
    if _gemini_llm_instance is None:
        _gemini_llm_instance = GeminiLLM()
    return _gemini_llm_instance

def generate_answer(question: str) -> str:
    """Convenience function to generate an answer using the global instance."""
    llm = get_gemini_llm()
    return llm.generate_answer(question)