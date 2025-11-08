import wikipedia
import re
import logging
import warnings
from bs4 import GuessedAtParserWarning
from typing import List, Optional
from collections import OrderedDict

warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaRetriever:
    def __init__(self, max_chars: int = 2000, max_results: int = 5):
        self.max_chars = max_chars
        self.max_results = max_results

    def retrieve_evidence_documents(self, question: str) -> List[str]:
        logger.info(f"Starting evidence retrieval for question: {question}")
        
        documents = []
        seen_titles = set()

        keywords = self._extract_keywords(question)
        search_queries = self._generate_search_queries(question, keywords)
        logger.info(f"Generated search queries: {search_queries}")

        candidate_titles = []
        for query in search_queries:
            try:
                results = wikipedia.search(query, results=self.max_results)
                candidate_titles.extend(results)
            except Exception as e:
                logger.warning(f"Wikipedia search failed for query '{query}': {e}")
        
        unique_candidate_titles = list(OrderedDict.fromkeys(candidate_titles))
        
        logger.info(f"Fetching content for top {len(unique_candidate_titles)} candidate titles...")
        for title in unique_candidate_titles:
            if len(documents) >= self.max_results:
                break
            
            if title not in seen_titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False, redirect=True)
                    content = page.content
                    documents.append(content)
                    seen_titles.add(page.title)
                    logger.info(f"Added evidence document from search result: {page.title}")
                except wikipedia.exceptions.PageError:
                    logger.warning(f"Could not find Wikipedia page for '{title}'. Skipping.")
                except wikipedia.exceptions.DisambiguationError as e:
                    logger.warning(f"Disambiguation page for '{title}'. Skipping. Options: {e.options[:3]}")
                except Exception as page_e:
                    logger.warning(f"Could not retrieve content for page '{title}': {page_e}")

        logger.info(f"Retrieved {len(documents)} unique evidence documents from Wikipedia.")
        return documents

    def _extract_keywords(self, question: str) -> List[str]:
        stop_words = {
            'a','about','above','after','again','against','all','am','an','and','any','are','as','at',
            'be','because','been','before','being','below','between','both','but','by','can','did','do',
            'does','doing','down','during','each','few','for','from','further','had','has','have','having',
            'he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','it',
            'its','itself','just','me','more','most','my','myself','no','nor','not','now','of','off','on',
            'once','only','or','other','our','ours','ourselves','out','over','own','s','same','she','should',
            'so','some','such','t','than','that','the','their','theirs','them','themselves','then','there',
            'these','they','this','those','through','to','too','under','until','up','very','was','we','were',
            'what','when','where','which','while','who','whom','why','will','with','you','your','yours',
            'yourself','yourselves','known'
        }
        
        proper_nouns = [word for word in re.findall(r'\b[A-Z][a-z]+\b', question)]
        words = re.findall(r'\b\w+\b', question.lower())
        other_words = [word for word in words if word not in stop_words and word not in [p.lower() for p in proper_nouns]]
        keywords = proper_nouns + other_words
        return list(OrderedDict.fromkeys(keywords))[:5]

    def _generate_search_queries(self, question: str, keywords: List[str]) -> List[str]:
        search_queries = []
        if keywords:
            search_queries.append(" ".join(keywords))
        search_queries.append(question)
        search_queries.extend(keywords)
        return list(OrderedDict.fromkeys(search_queries))

    def get_page_summary(self, page_title: str) -> Optional[str]:
        try:
            return wikipedia.summary(page_title, sentences=3, auto_suggest=False, redirect=True)
        except Exception as e:
            logger.error(f"Error getting summary for {page_title}: {e}")
            return None
