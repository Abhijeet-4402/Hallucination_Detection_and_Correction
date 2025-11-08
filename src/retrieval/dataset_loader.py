import logging
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TruthfulQALoader:
    def __init__(self, dataset_name: str = "truthful_qa", subset: str = "generation"):
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = None
        self._load_dataset()
    
    def _load_dataset(self):
        try:
            logger.info(f"Loading TruthfulQA dataset: {self.dataset_name}/{self.subset}")
            self.dataset = load_dataset(self.dataset_name, self.subset)
            logger.info("Dataset loaded successfully")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_sample_questions(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        if not self.dataset:
            logger.error("Dataset not loaded")
            return []
        try:
            split_name = 'validation' if 'validation' in self.dataset else 'train'
            samples = self.dataset[split_name].select(range(min(num_samples, len(self.dataset[split_name]))))
            sample_list = []
            for i, sample in enumerate(samples):
                sample_dict = {
                    'id': i,
                    'question': sample.get('question', ''),
                    'best_answer': sample.get('best_answer', ''),
                    'correct_answers': sample.get('correct_answers', []),
                    'incorrect_answers': sample.get('incorrect_answers', []),
                    'category': sample.get('category', ''),
                    'source': 'truthful_qa'
                }
                sample_list.append(sample_dict)
            logger.info(f"Retrieved {len(sample_list)} sample questions")
            return sample_list
        except Exception as e:
            logger.error(f"Error getting sample questions: {e}")
            return []

    def get_questions_by_category(self, category: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        if not self.dataset:
            logger.error("Dataset not loaded")
            return []
        try:
            split_name = 'validation' if 'validation' in self.dataset else 'train'
            all_samples = self.dataset[split_name]
            category_samples = [sample for sample in all_samples if sample.get('category', '') == category]
            selected_samples = category_samples[:num_samples]
            sample_list = []
            for i, sample in enumerate(selected_samples):
                sample_dict = {
                    'id': i,
                    'question': sample.get('question', ''),
                    'best_answer': sample.get('best_answer', ''),
                    'correct_answers': sample.get('correct_answers', []),
                    'incorrect_answers': sample.get('incorrect_answers', []),
                    'category': sample.get('category', ''),
                    'source': 'truthful_qa'
                }
                sample_list.append(sample_dict)
            logger.info(f"Retrieved {len(sample_list)} questions from category: {category}")
            return sample_list
        except Exception as e:
            logger.error(f"Error getting questions by category: {e}")
            return []

    def get_all_categories(self) -> List[str]:
        if not self.dataset:
            logger.error("Dataset not loaded")
            return []
        try:
            split_name = 'validation' if 'validation' in self.dataset else 'train'
            all_samples = self.dataset[split_name]
            categories = set()
            for sample in all_samples:
                category = sample.get('category', '')
                if category:
                    categories.add(category)
            category_list = sorted(list(categories))
            logger.info(f"Found {len(category_list)} categories: {category_list}")
            return category_list
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_dataset_info(self) -> Dict[str, Any]:
        if not self.dataset:
            return {'error': 'Dataset not loaded'}
        try:
            info = {
                'dataset_name': self.dataset_name,
                'subset': self.subset,
                'splits': list(self.dataset.keys()),
                'total_samples': sum(len(self.dataset[split]) for split in self.dataset.keys()),
                'categories': self.get_all_categories()
            }
            for split in self.dataset.keys():
                info[f'{split}_samples'] = len(self.dataset[split])
            return info
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {'error': str(e)}

    def export_to_csv(self, filename: str, num_samples: int = 100) -> bool:
        if not self.dataset:
            logger.error("Dataset not loaded")
            return False
        try:
            split_name = 'validation' if 'validation' in self.dataset else 'train'
            samples = self.dataset[split_name].select(range(min(num_samples, len(self.dataset[split_name]))))
            df = pd.DataFrame(samples)
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(df)} samples to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

_loader = None

def get_truthfulqa_loader() -> TruthfulQALoader:
    global _loader
    if _loader is None:
        _loader = TruthfulQALoader()
    return _loader
