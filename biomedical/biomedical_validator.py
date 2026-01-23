"""
Validator for biomedical entities, claims, and citations.
"""

import re
import requests
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json


class BiomedicalValidator:
    """Validates biomedical questions and answers for scientific accuracy."""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Args:
            cache_dir: Directory to cache API responses
        """
        self.cache_dir = cache_dir
        self.gene_cache = {}
        self.pmid_cache = {}
        
    def validate_question(self, question: str, document: str) -> Tuple[bool, float, str]:
        """
        Validate if a question is scientifically meaningful.
        
        Args:
            question: The generated question
            document: Source document (PubMed abstract)
            
        Returns:
            Tuple of (is_valid, confidence_score, explanation)
        """
        issues = []
        score = 1.0
        
        # Check 1: Question contains biomedical entities
        entities = self._extract_biomedical_entities(question)
        if not entities:
            issues.append("No biomedical entities detected")
            score *= 0.5
        
        # Check 2: Question references the source document
        pmid_in_doc = self._extract_pmid(document)
        if not pmid_in_doc:
            issues.append("Source document has no PMID")
            score *= 0.8
        
        # Check 3: Question is not trivial (avoid title/abstract only questions)
        if self._is_trivial_question(question, document):
            issues.append("Question appears trivial (answerable from title/abstract)")
            score *= 0.6
        
        # Check 4: Gene names are valid (if present)
        genes = self._extract_gene_names(question)
        for gene in genes:
            if not self._is_valid_gene(gene):
                issues.append(f"Invalid or unrecognized gene name: {gene}")
                score *= 0.7
        
        # Check 5: Question implies causal/mechanistic reasoning
        if not self._has_reasoning_indicators(question):
            issues.append("Question lacks mechanistic/causal reasoning indicators")
            score *= 0.8
        
        is_valid = score >= 0.4  # Threshold for validity
        explanation = "; ".join(issues) if issues else "Question passes validation"
        
        return is_valid, score, explanation
    
    def validate_answer(
        self,
        answer: str,
        question: str,
        ground_truth: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Validate answer quality and scientific accuracy.
        
        Args:
            answer: Generated answer
            question: Original question
            ground_truth: Gold answer (if available)
            
        Returns:
            Tuple of (score, details_dict)
        """
        details = {}
        score = 0.0
        
        # Check 1: PMID citations present and valid
        pmids = self._extract_pmids(answer)
        valid_pmids = [pmid for pmid in pmids if self._verify_pmid_exists(pmid)]
        
        citation_score = len(valid_pmids) / max(len(pmids), 1) if pmids else 0.5
        details["valid_citations"] = len(valid_pmids)
        details["invalid_citations"] = len(pmids) - len(valid_pmids)
        details["citation_score"] = citation_score
        score += 0.3 * citation_score
        
        # Check 2: Answer contains relevant biomedical entities
        answer_entities = self._extract_biomedical_entities(answer)
        question_entities = self._extract_biomedical_entities(question)
        
        entity_overlap = len(set(answer_entities) & set(question_entities))
        entity_score = entity_overlap / max(len(question_entities), 1)
        details["entity_overlap"] = entity_overlap
        details["entity_score"] = entity_score
        score += 0.2 * entity_score
        
        # Check 3: Answer length (not too short, not too long)
        answer_words = len(answer.split())
        length_score = 1.0 if 10 <= answer_words <= 150 else 0.5
        details["answer_length"] = answer_words
        details["length_score"] = length_score
        score += 0.1 * length_score
        
        # Check 4: No obvious hallucinations
        hallucination_score = self._check_hallucinations(answer)
        details["hallucination_score"] = hallucination_score
        score += 0.2 * hallucination_score
        
        # Check 5: Semantic similarity to ground truth (if available)
        if ground_truth:
            similarity_score = self._semantic_similarity(answer, ground_truth)
            details["semantic_similarity"] = similarity_score
            score += 0.2 * similarity_score
        else:
            score += 0.2 * 0.5  # Neutral score if no ground truth
        
        return score, details
    
    def _extract_biomedical_entities(self, text: str) -> List[str]:
        """Extract biomedical entities (genes, diseases, drugs) from text."""
        entities = []
        
        # Gene patterns (uppercase, 2-10 chars)
        genes = re.findall(r'\b[A-Z][A-Z0-9]{1,9}\b', text)
        entities.extend(genes)
        
        # Common disease patterns
        disease_keywords = ['cancer', 'carcinoma', 'disease', 'syndrome', 'disorder']
        for keyword in disease_keywords:
            matches = re.findall(rf'\b\w+\s+{keyword}\b', text, re.IGNORECASE)
            entities.extend(matches)
        
        # Drug patterns (often end in -mab, -nib, -cept, etc.)
        drug_suffixes = ['mab', 'nib', 'cept', 'stat', 'pril']
        for suffix in drug_suffixes:
            matches = re.findall(rf'\b\w+{suffix}\b', text, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))
    
    def _extract_pmid(self, text: str) -> Optional[str]:
        """Extract PMID from text."""
        match = re.search(r'PMID[:\s]*(\d+)', text, re.IGNORECASE)
        return match.group(1) if match else None
    
    def _extract_pmids(self, text: str) -> List[str]:
        """Extract all PMIDs from text."""
        return re.findall(r'PMID[:\s]*(\d+)', text, re.IGNORECASE)
    
    def _extract_gene_names(self, text: str) -> List[str]:
        """Extract likely gene names (uppercase, 2-10 chars)."""
        # Common gene patterns
        genes = re.findall(r'\b([A-Z][A-Z0-9]{1,9})\b', text)
        
        # Filter out common English words
        english_words = {'DNA', 'RNA', 'USA', 'FDA', 'WHO', 'NIH', 'UK', 'US'}
        genes = [g for g in genes if g not in english_words]
        
        return genes
    
    def _is_valid_gene(self, gene: str) -> bool:
        """
        Check if gene name is valid using HGNC (cached).
        For simplicity, we'll use a basic heuristic here.
        In production, query HGNC API: https://www.genenames.org/
        """
        # Check cache
        if gene in self.gene_cache:
            return self.gene_cache[gene]
        
        # Basic validation heuristic
        # Valid genes are typically 2-10 uppercase characters
        is_valid = bool(re.match(r'^[A-Z][A-Z0-9]{1,9}$', gene))
        
        # TODO: Add actual HGNC API call in production
        # try:
        #     response = requests.get(f"https://rest.genenames.org/fetch/symbol/{gene}")
        #     is_valid = response.status_code == 200
        # except:
        #     is_valid = False
        
        self.gene_cache[gene] = is_valid
        return is_valid
    
    def _verify_pmid_exists(self, pmid: str) -> bool:
        """
        Verify PMID exists in PubMed (cached).
        """
        # Check cache
        if pmid in self.pmid_cache:
            return self.pmid_cache[pmid]
        
        # Query PubMed API
        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "json"
            }
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                exists = "result" in data and pmid in data["result"]
            else:
                exists = False
            
            self.pmid_cache[pmid] = exists
            return exists
            
        except Exception as e:
            print(f"⚠️ Error verifying PMID {pmid}: {e}")
            return False
    
    def _is_trivial_question(self, question: str, document: str) -> bool:
        """Check if question is answerable directly from document title/abstract."""
        # Extract key phrases from question
        question_lower = question.lower()
        document_lower = document.lower()
        
        # If question asks for something explicitly stated in document, it's trivial
        question_words = set(question_lower.split())
        document_words = set(document_lower.split())
        
        overlap = len(question_words & document_words) / len(question_words)
        
        # If >70% of question words appear in document, likely trivial
        return overlap > 0.7
    
    def _has_reasoning_indicators(self, question: str) -> bool:
        """Check if question requires mechanistic/causal reasoning."""
        reasoning_keywords = [
            'how', 'why', 'mechanism', 'pathway', 'regulate', 'affect',
            'influence', 'cause', 'lead to', 'result in', 'mediate',
            'modulate', 'interact', 'relationship', 'association'
        ]
        
        question_lower = question.lower()
        return any(kw in question_lower for kw in reasoning_keywords)
    
    def _check_hallucinations(self, answer: str) -> float:
        """
        Check for obvious hallucinations (placeholder function).
        Returns score from 0 (hallucination) to 1 (no hallucination).
        """
        # Basic heuristics
        hallucination_patterns = [
            r'I cannot',
            r'I do not know',
            r'As an AI',
            r'I apologize',
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return 0.3  # Low score for refusal patterns
        
        # Check for nonsensical gene combinations
        genes = self._extract_gene_names(answer)
        if len(genes) > 20:  # Suspiciously many genes
            return 0.5
        
        return 1.0  # Pass by default
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity (simple word overlap).
        In production, use sentence-transformers with PubMedBERT.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def validate_question_answer_pair(
        self,
        question: str,
        answer: str,
        document: str
    ) -> Dict:
        """
        Comprehensive validation of QA pair.
        
        Returns:
            Dictionary with validation results
        """
        # Validate question
        q_valid, q_score, q_explanation = self.validate_question(question, document)
        
        # Validate answer
        a_score, a_details = self.validate_answer(answer, question)
        
        # Combined score
        combined_score = 0.4 * q_score + 0.6 * a_score
        
        return {
            "question_valid": q_valid,
            "question_score": q_score,
            "question_explanation": q_explanation,
            "answer_score": a_score,
            "answer_details": a_details,
            "combined_score": combined_score,
            "passed": combined_score >= 0.5
        }


if __name__ == "__main__":
    # Example usage
    validator = BiomedicalValidator()
    
    # Test question validation
    question = "What is the mechanism by which TP53 mutations lead to resistance to platinum-based chemotherapy in ovarian cancer?"
    document = "(PMID: 12345678, Title: 'TP53 mutations in ovarian cancer')\nTP53 is frequently mutated in ovarian cancer..."
    
    is_valid, score, explanation = validator.validate_question(question, document)
    print(f"Question valid: {is_valid} (score: {score:.2f})")
    print(f"Explanation: {explanation}")
    
    # Test answer validation
    answer = "TP53 mutations impair DNA damage response pathways, leading to reduced apoptosis in response to platinum agents (PMID: 12345678)."
    a_score, a_details = validator.validate_answer(answer, question)
    print(f"\nAnswer score: {a_score:.2f}")
    print(f"Details: {a_details}")
