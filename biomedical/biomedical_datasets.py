"""
Biomedical QA datasets for evaluation.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset
import requests


class BiomedicalDatasets:
    """Load and manage biomedical QA evaluation datasets."""
    
    def __init__(self, cache_dir: str = "./data/biomedical"):
        """
        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_pubmedqa(self, split: str = "test") -> List[Dict]:
        """
        Load PubMedQA dataset.
        
        PubMedQA contains yes/no/maybe questions about PubMed abstracts.
        
        Args:
            split: Dataset split ('train', 'dev', 'test')
            
        Returns:
            List of examples
        """
        print(f"📥 Loading PubMedQA ({split})...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
            
            examples = []
            for item in dataset:
                examples.append({
                    "question": item["question"],
                    "context": item["context"]["contexts"][0] if item["context"]["contexts"] else "",
                    "answer": item["final_decision"],
                    "pmid": item["pubid"],
                    "long_answer": item["long_answer"],
                    "dataset": "pubmedqa"
                })
            
            print(f"✓ Loaded {len(examples)} PubMedQA examples")
            return examples
            
        except Exception as e:
            print(f"❌ Error loading PubMedQA: {e}")
            return []
    
    def load_bioasq(self, year: int = 2024) -> List[Dict]:
        """
        Load BioASQ dataset (requires manual download).
        
        BioASQ is a biomedical semantic indexing and question answering challenge.
        Download from: http://bioasq.org/
        
        Args:
            year: BioASQ challenge year
            
        Returns:
            List of examples
        """
        print(f"📥 Loading BioASQ {year}...")
        
        bioasq_path = self.cache_dir / f"bioasq_{year}.json"
        
        if not bioasq_path.exists():
            print(f"⚠️ BioASQ dataset not found at {bioasq_path}")
            print("   Download from: http://bioasq.org/participate/challenges")
            return []
        
        try:
            with open(bioasq_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = []
            for item in data.get("questions", []):
                examples.append({
                    "question": item["body"],
                    "answer": item.get("ideal_answer", ""),
                    "type": item.get("type", ""),
                    "snippets": [s["text"] for s in item.get("snippets", [])],
                    "pmids": [s["pmid"] for s in item.get("snippets", [])],
                    "dataset": "bioasq"
                })
            
            print(f"✓ Loaded {len(examples)} BioASQ examples")
            return examples
            
        except Exception as e:
            print(f"❌ Error loading BioASQ: {e}")
            return []
    
    def load_medqa(self, subset: str = "4_options") -> List[Dict]:
        """
        Load MedQA dataset.
        
        MedQA contains medical exam questions from USMLE, MCMLE, etc.
        
        Args:
            subset: '4_options' or '5_options'
            
        Returns:
            List of examples
        """
        print(f"📥 Loading MedQA ({subset})...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset("bigbio/med_qa", subset, split="test")
            
            examples = []
            for item in dataset:
                examples.append({
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "answer_idx": item["answer_idx"],
                    "meta_info": item.get("meta_info", ""),
                    "dataset": "medqa"
                })
            
            print(f"✓ Loaded {len(examples)} MedQA examples")
            return examples
            
        except Exception as e:
            print(f"❌ Error loading MedQA: {e}")
            print("   Try: pip install datasets")
            return []
    
    def load_covid_qa(self) -> List[Dict]:
        """
        Load COVID-QA dataset.
        
        COVID-QA contains questions about COVID-19 research papers.
        
        Returns:
            List of examples
        """
        print("📥 Loading COVID-QA...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset("covid_qa_deepset", split="train")
            
            examples = []
            for item in dataset:
                examples.append({
                    "question": item["question"],
                    "context": item["context"],
                    "answer": item["answers"]["text"][0] if item["answers"]["text"] else "",
                    "dataset": "covid_qa"
                })
            
            print(f"✓ Loaded {len(examples)} COVID-QA examples")
            return examples
            
        except Exception as e:
            print(f"❌ Error loading COVID-QA: {e}")
            return []
    
    def create_custom_eval_set(
        self,
        corpus_path: str,
        n_samples: int = 100,
        difficulty_levels: List[int] = [1, 2, 3]
    ) -> List[Dict]:
        """
        Create custom evaluation set from PubMed corpus.
        
        Samples papers and creates questions requiring different hop counts.
        
        Args:
            corpus_path: Path to PubMed corpus JSONL
            n_samples: Number of samples to create
            difficulty_levels: List of hop counts to include
            
        Returns:
            List of evaluation examples
        """
        print(f"📥 Creating custom eval set ({n_samples} samples)...")
        
        # Load corpus
        corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(json.loads(line))
        
        # Sample diverse papers
        import random
        random.seed(42)
        sampled = random.sample(corpus, min(n_samples, len(corpus)))
        
        # Create examples with different difficulty levels
        examples = []
        for i, article in enumerate(sampled):
            hop = difficulty_levels[i % len(difficulty_levels)]
            
            examples.append({
                "question": f"[To be generated for hop={hop}]",
                "answer": "[To be determined]",
                "document": f"(PMID: {article['pmid']}) {article['title']}\n{article['abstract']}",
                "pmid": article['pmid'],
                "hop_count": hop,
                "dataset": "custom"
            })
        
        print(f"✓ Created {len(examples)} custom examples")
        return examples
    
    def get_all_datasets(self) -> Dict[str, List[Dict]]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary mapping dataset name to examples
        """
        datasets = {}
        
        # Try loading each dataset
        try:
            datasets["pubmedqa"] = self.load_pubmedqa("test")
        except:
            datasets["pubmedqa"] = []
        
        try:
            datasets["bioasq"] = self.load_bioasq()
        except:
            datasets["bioasq"] = []
        
        try:
            datasets["medqa"] = self.load_medqa()
        except:
            datasets["medqa"] = []
        
        try:
            datasets["covid_qa"] = self.load_covid_qa()
        except:
            datasets["covid_qa"] = []
        
        # Filter empty datasets
        datasets = {k: v for k, v in datasets.items() if v}
        
        print(f"\n📊 Dataset Summary:")
        for name, examples in datasets.items():
            print(f"  {name}: {len(examples)} examples")
        
        return datasets
    
    def save_evaluation_set(
        self,
        examples: List[Dict],
        output_path: str,
        format: str = "jsonl"
    ):
        """
        Save evaluation set to file.
        
        Args:
            examples: List of examples
            output_path: Output file path
            format: 'jsonl' or 'json'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved {len(examples)} examples to {output_path}")
    
    def load_evaluation_set(self, input_path: str) -> List[Dict]:
        """
        Load evaluation set from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            List of examples
        """
        examples = []
        input_path = Path(input_path)
        
        if input_path.suffix == '.jsonl':
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
        
        print(f"📥 Loaded {len(examples)} examples from {input_path}")
        return examples


def download_sample_datasets():
    """
    Download sample biomedical datasets for quick testing.
    """
    datasets = BiomedicalDatasets()
    
    print("=" * 80)
    print("Downloading Sample Biomedical Datasets")
    print("=" * 80)
    
    # PubMedQA (small, easy to download)
    pubmedqa = datasets.load_pubmedqa("test")
    if pubmedqa:
        datasets.save_evaluation_set(
            pubmedqa[:100],
            "./data/biomedical/pubmedqa_test.jsonl"
        )
    
    # COVID-QA
    covid_qa = datasets.load_covid_qa()
    if covid_qa:
        datasets.save_evaluation_set(
            covid_qa[:100],
            "./data/biomedical/covid_qa_test.jsonl"
        )
    
    print("\n✅ Sample datasets downloaded!")


if __name__ == "__main__":
    # Download sample datasets
    download_sample_datasets()
    
    # Or load all available
    # datasets = BiomedicalDatasets()
    # all_data = datasets.get_all_datasets()
