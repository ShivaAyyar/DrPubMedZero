"""
Biomedical retriever server using PubMedBERT embeddings.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm


class BiomedicalRetrieverServer:
    """
    Retriever server for PubMed corpus using biomedical embeddings.
    Compatible with Dr. Zero's search interface.
    """
    
    def __init__(
        self,
        corpus_path: str,
        index_path: Optional[str] = None,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        topk: int = 3
    ):
        """
        Args:
            corpus_path: Path to JSONL corpus file
            index_path: Path to FAISS index (will create if doesn't exist)
            model_name: HuggingFace model for embeddings
            device: Device for model inference
            topk: Number of results to return
        """
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path) if index_path else None
        self.device = device
        self.topk = topk
        
        # Load corpus
        print(f"📚 Loading corpus from {self.corpus_path}")
        self.corpus = self._load_corpus()
        print(f"✓ Loaded {len(self.corpus)} documents")
        
        # Load model and tokenizer
        print(f"🤖 Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        
        # Load or build index
        if self.index_path and self.index_path.exists():
            print(f"📇 Loading FAISS index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            print(f"✓ Loaded index with {self.index.ntotal} vectors")
        else:
            print("🔨 Building FAISS index...")
            self.index = self._build_index()
            if self.index_path:
                print(f"💾 Saving index to {self.index_path}")
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(self.index_path))
                print("✓ Index saved")
    
    def _load_corpus(self) -> List[Dict]:
        """Load corpus from JSONL."""
        corpus = []
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                corpus.append(json.loads(line))
        return corpus
    
    def _build_index(self, batch_size: int = 32) -> faiss.Index:
        """Build FAISS index from corpus."""
        # Encode all documents
        embeddings = []
        
        for i in tqdm(range(0, len(self.corpus), batch_size), desc="Encoding corpus"):
            batch = self.corpus[i:i+batch_size]
            texts = [doc["text"] for doc in batch]
            batch_embeddings = self._encode_batch(texts)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        print(f"✓ Built index with {index.ntotal} vectors (dim={dimension})")
        return index
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts to embeddings."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def search(self, query: str, topk: Optional[int] = None) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            topk: Number of results (overrides default)
            
        Returns:
            List of relevant documents with scores
        """
        if topk is None:
            topk = self.topk
        
        # Encode query
        query_embedding = self._encode_batch([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, topk)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.corpus):
                doc = self.corpus[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def search_multiple(self, queries: List[str]) -> List[List[Dict]]:
        """
        Search for multiple queries (batch).
        
        Args:
            queries: List of search queries
            
        Returns:
            List of result lists
        """
        return [self.search(query) for query in queries]
    
    def format_search_results(self, results: List[Dict]) -> str:
        """
        Format search results for LLM consumption.
        Compatible with Dr. Zero's tool response format.
        
        Args:
            results: List of search results
            
        Returns:
            Formatted string
        """
        if not results:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(results, 1):
            formatted.append(
                f"Document {i} (PMID: {doc['pmid']}, Score: {doc['score']:.3f}):\n"
                f"Title: {doc['title']}\n"
                f"Journal: {doc['journal']} ({doc['pub_date']})\n"
                f"Abstract: {doc['abstract'][:500]}...\n"
            )
        
        return "\n---\n".join(formatted)
    
    def get_document_by_pmid(self, pmid: str) -> Optional[Dict]:
        """Retrieve document by PMID."""
        for doc in self.corpus:
            if doc['pmid'] == pmid:
                return doc
        return None
    
    def rerank_results(
        self,
        query: str,
        results: List[Dict],
        rerank_model: Optional[str] = None
    ) -> List[Dict]:
        """
        Rerank results using cross-encoder (optional).
        
        Args:
            query: Original query
            results: Initial search results
            rerank_model: Cross-encoder model name
            
        Returns:
            Reranked results
        """
        if rerank_model is None:
            return results  # No reranking
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Load reranker
            reranker = CrossEncoder(rerank_model)
            
            # Score query-document pairs
            pairs = [(query, doc['text']) for doc in results]
            scores = reranker.predict(pairs)
            
            # Rerank
            for doc, score in zip(results, scores):
                doc['rerank_score'] = float(score)
            
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
        except ImportError:
            print("⚠️ sentence-transformers not installed, skipping reranking")
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
        
        return results


class BiomedicalSearchAPI:
    """
    Simple API wrapper for compatibility with Dr. Zero's tool calling format.
    """
    
    def __init__(self, retriever: BiomedicalRetrieverServer):
        self.retriever = retriever
    
    def search(self, query_list: List[str]) -> str:
        """
        Search API compatible with Dr. Zero.
        
        Args:
            query_list: List of search queries
            
        Returns:
            Formatted search results as string
        """
        all_results = []
        
        for query in query_list:
            results = self.retriever.search(query)
            formatted = self.retriever.format_search_results(results)
            all_results.append(f"Query: {query}\n{formatted}")
        
        return "\n\n".join(all_results)


def build_biomedical_index(
    corpus_path: str,
    index_path: str,
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
):
    """
    Standalone function to build FAISS index.
    
    Usage:
        build_biomedical_index(
            corpus_path="./corpus/pubmed-corpus.jsonl",
            index_path="./corpus/pubmedbert_index.faiss"
        )
    """
    retriever = BiomedicalRetrieverServer(
        corpus_path=corpus_path,
        index_path=index_path,
        model_name=model_name
    )
    print("✅ Index built successfully!")
    return retriever


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        # Build index from command line
        corpus_path = sys.argv[1]
        index_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        build_biomedical_index(corpus_path, index_path)
    else:
        # Test retriever
        retriever = BiomedicalRetrieverServer(
            corpus_path="./corpus/pubmed/pubmed-corpus.jsonl",
            index_path="./corpus/pubmed/pubmedbert_index.faiss",
            topk=3
        )
        
        # Test search
        query = "TP53 mutations in breast cancer drug resistance"
        results = retriever.search(query)
        
        print("\n🔍 Search Results:")
        print(retriever.format_search_results(results))
