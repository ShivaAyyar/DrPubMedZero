"""
PubMed corpus manager for downloading and preprocessing biomedical literature.
"""

import os
import json
import gzip
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm
from Bio import Entrez, Medline
import xml.etree.ElementTree as ET


class PubMedCorpusManager:
    """Handles PubMed corpus download, processing, and indexing."""
    
    def __init__(self, save_path: str, email: str = "your_email@example.com"):
        """
        Args:
            save_path: Directory to save corpus
            email: Email for NCBI Entrez (required by NCBI)
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Set up Entrez
        Entrez.email = email
        Entrez.api_key = os.environ.get("NCBI_API_KEY", None)  # Optional, increases rate limit
        
        self.corpus_file = self.save_path / "pubmed-corpus.jsonl"
        
    def download_pubmed_abstracts(
        self,
        query: str = "(cancer OR drug resistance) AND (gene OR protein)",
        max_results: int = 50000,
        date_range: Optional[tuple] = ("2019/01/01", "2024/12/31"),
        batch_size: int = 500
    ) -> List[Dict]:
        """
        Download PubMed abstracts using Entrez API.
        
        Args:
            query: PubMed search query
            max_results: Maximum number of papers to download
            date_range: Tuple of (start_date, end_date) in YYYY/MM/DD format
            batch_size: Number of records per API call
            
        Returns:
            List of article dictionaries
        """
        print(f"Searching PubMed with query: {query}")
        
        # Build search query with date range
        if date_range:
            query = f"{query} AND {date_range[0]}:{date_range[1]}[PDAT]"
        
        # Search for PMIDs
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            pmids = search_results["IdList"]
            print(f"Found {len(pmids)} articles")
            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
        
        # Fetch article details in batches
        articles = []
        for i in tqdm(range(0, len(pmids), batch_size), desc="Downloading articles"):
            batch_pmids = pmids[i:i+batch_size]
            
            try:
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_pmids,
                    rettype="medline",
                    retmode="text"
                )
                records = Medline.parse(fetch_handle)
                
                for record in records:
                    article = self._parse_medline_record(record)
                    if article:
                        articles.append(article)
                
                fetch_handle.close()
                
            except Exception as e:
                print(f"Error fetching batch {i}: {e}")
                continue
        
        print(f"Downloaded {len(articles)} articles with abstracts")
        return articles
    
    def _parse_medline_record(self, record: Dict) -> Optional[Dict]:
        """Parse a Medline record into structured format."""
        # Skip articles without abstracts
        if "AB" not in record or not record["AB"]:
            return None
        
        article = {
            "pmid": record.get("PMID", ""),
            "title": record.get("TI", ""),
            "abstract": record.get("AB", ""),
            "authors": record.get("AU", []),
            "journal": record.get("JT", ""),
            "pub_date": record.get("DP", ""),
            "mesh_terms": record.get("MH", []),
            "doi": self._extract_doi(record),
            "text": f"{record.get('TI', '')} {record.get('AB', '')}",  # Combined for indexing
        }
        
        return article
    
    def _extract_doi(self, record: Dict) -> str:
        """Extract DOI from record."""
        if "AID" in record:
            for aid in record["AID"]:
                if "[doi]" in aid:
                    return aid.replace(" [doi]", "")
        return ""
    
    def save_corpus(self, articles: List[Dict]):
        """Save articles to JSONL format."""
        print(f"Saving corpus to {self.corpus_file}")
        
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(articles)} articles")
    
    def load_corpus(self) -> List[Dict]:
        """Load corpus from JSONL file."""
        if not self.corpus_file.exists():
            raise FileNotFoundError(f"Corpus not found at {self.corpus_file}")
        
        articles = []
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                articles.append(json.loads(line))
        
        print(f"Loaded {len(articles)} articles from corpus")
        return articles
    
    def download_pmc_full_text(
        self,
        pmids: List[str],
        save_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Download full-text articles from PubMed Central (PMC).
        Note: Only works for Open Access articles.
        
        Args:
            pmids: List of PubMed IDs
            save_dir: Directory to save full-text articles
            
        Returns:
            Dictionary mapping PMID to full text
        """
        if save_dir is None:
            save_dir = self.save_path / "pmc_full_text"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        full_texts = {}
        
        for pmid in tqdm(pmids, desc="Downloading PMC full text"):
            try:
                # Convert PMID to PMCID
                link_handle = Entrez.elink(
                    dbfrom="pubmed",
                    db="pmc",
                    id=pmid,
                    linkname="pubmed_pmc"
                )
                link_results = Entrez.read(link_handle)
                link_handle.close()
                
                if not link_results[0]["LinkSetDb"]:
                    continue
                
                pmcid = link_results[0]["LinkSetDb"][0]["Link"][0]["Id"]
                
                # Fetch full text XML
                fetch_handle = Entrez.efetch(
                    db="pmc",
                    id=pmcid,
                    rettype="xml",
                    retmode="text"
                )
                xml_content = fetch_handle.read()
                fetch_handle.close()
                
                # Parse XML and extract text
                full_text = self._extract_text_from_pmc_xml(xml_content)
                full_texts[pmid] = full_text
                
                # Save to file
                with open(save_dir / f"{pmid}.txt", 'w', encoding='utf-8') as f:
                    f.write(full_text)
                
            except Exception as e:
                print(f"Could not fetch full text for PMID {pmid}: {e}")
                continue
        
        print(f"Downloaded {len(full_texts)} full-text articles")
        return full_texts
    
    def _extract_text_from_pmc_xml(self, xml_content: bytes) -> str:
        """Extract plain text from PMC XML."""
        try:
            root = ET.fromstring(xml_content)
            
            # Extract sections
            sections = []
            for sec in root.findall(".//sec"):
                title = sec.find("title")
                if title is not None and title.text:
                    sections.append(f"\n## {title.text}\n")
                
                for p in sec.findall(".//p"):
                    if p.text:
                        sections.append(p.text)
            
            return "\n".join(sections)
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return ""
    
    def create_training_seeds(
        self,
        n_seeds: int = 1000,
        filter_keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Create initial training seeds from corpus for proposer.
        
        Args:
            n_seeds: Number of seed documents
            filter_keywords: Keywords to filter relevant papers
            
        Returns:
            List of seed documents with metadata
        """
        articles = self.load_corpus()
        
        # Filter by keywords if provided
        if filter_keywords:
            filtered = []
            for article in articles:
                text = article["text"].lower()
                if any(kw.lower() in text for kw in filter_keywords):
                    filtered.append(article)
            articles = filtered
        
        # Sample diverse papers
        import random
        random.seed(42)
        seeds = random.sample(articles, min(n_seeds, len(articles)))
        
        # Format for proposer
        formatted_seeds = []
        for article in seeds:
            formatted_seeds.append({
                "document": f"(PMID: {article['pmid']}, Title: \"{article['title']}\")\n{article['abstract']}",
                "pmid": article["pmid"],
                "title": article["title"],
                "metadata": {
                    "journal": article["journal"],
                    "pub_date": article["pub_date"],
                    "mesh_terms": article["mesh_terms"]
                }
            })
        
        return formatted_seeds
    
    def get_corpus_statistics(self) -> Dict:
        """Get corpus statistics."""
        articles = self.load_corpus()
        
        stats = {
            "total_articles": len(articles),
            "avg_abstract_length": sum(len(a["abstract"].split()) for a in articles) / len(articles),
            "date_range": {
                "earliest": min(a["pub_date"] for a in articles if a["pub_date"]),
                "latest": max(a["pub_date"] for a in articles if a["pub_date"])
            },
            "unique_journals": len(set(a["journal"] for a in articles if a["journal"])),
            "with_mesh_terms": sum(1 for a in articles if a["mesh_terms"]),
            "with_doi": sum(1 for a in articles if a["doi"])
        }
        
        return stats


if __name__ == "__main__":
    # Example usage
    manager = PubMedCorpusManager(
        save_path="./corpus/pubmed",
        email="your_email@example.com"
    )
    
    # Download abstracts
    articles = manager.download_pubmed_abstracts(
        query="(breast cancer OR lung cancer) AND (gene therapy OR targeted therapy)",
        max_results=10000,
        date_range=("2020/01/01", "2024/12/31")
    )
    
    # Save corpus
    manager.save_corpus(articles)
    
    # Get statistics
    stats = manager.get_corpus_statistics()
    print("\nCorpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
