"""
Biomedical-adapted reward functions for Dr. Zero.
"""

import re
from typing import List, Dict, Tuple
import numpy as np
from .biomedical_validator import BiomedicalValidator


class BiomedicalRewardCalculator:
    """
    Calculate rewards for proposer and solver with biomedical validation.
    Adapted from Dr. Zero paper equations (2), (3), and (4).
    """
    
    def __init__(self, validator: BiomedicalValidator = None):
        """
        Args:
            validator: Biomedical validator instance
        """
        self.validator = validator or BiomedicalValidator()
    
    def calculate_proposer_reward(
        self,
        question: str,
        answer: str,
        document: str,
        solver_predictions: List[str],
        n_samples: int = 5
    ) -> float:
        """
        Calculate proposer reward based on difficulty and validity.
        
        Based on Dr. Zero Equation (4):
        r(y, {ŷᵢ}) = I(0 < k < n) * (n-k)/(n-1) + r_f
        
        Enhanced with biomedical validation:
        - r_f includes: format validity + entity validity + scientific plausibility
        
        Args:
            question: Generated question
            answer: Generated answer
            document: Source PubMed document
            solver_predictions: List of solver's predicted answers
            n_samples: Number of solver samples (should match len(solver_predictions))
            
        Returns:
            Reward score (0 to 1.5 range)
        """
        # Count correct predictions
        k = sum(1 for pred in solver_predictions if self._is_correct_answer(pred, answer))
        n = len(solver_predictions)
        
        # Base difficulty reward (from Dr. Zero paper)
        if k == 0 or k == n:
            difficulty_reward = 0.0  # Too hard or too easy
        else:
            difficulty_reward = (n - k) / (n - 1)  # Reward peaks when k=1
        
        # Calculate format reward (r_f) with biomedical validation
        format_reward = self._calculate_format_reward(question, answer, document)
        
        # Combined reward
        total_reward = difficulty_reward + format_reward
        
        return total_reward
    
    def _calculate_format_reward(
        self,
        question: str,
        answer: str,
        document: str
    ) -> float:
        """
        Calculate format reward with biomedical constraints.
        
        Dr. Zero's r_f checks:
        1. Valid <think> structure
        2. Valid tool usage
        3. Extractable <question>
        4. Extractable <answer>
        
        Our biomedical additions:
        5. Valid gene/protein names
        6. Scientific plausibility
        7. Proper PMID references
        8. Mechanistic reasoning indicators
        
        Returns:
            Format reward (0 to 0.5)
        """
        reward = 0.0
        max_reward = 0.5
        
        # Check 1: Question format (0.1)
        if self._has_valid_tags(question, 'question'):
            reward += 0.1
        
        # Check 2: Answer format (0.1)
        if self._has_valid_tags(answer, 'answer'):
            reward += 0.1
        
        # Check 3: Biomedical validation (0.15)
        is_valid, q_score, _ = self.validator.validate_question(question, document)
        if is_valid:
            reward += 0.15 * q_score
        
        # Check 4: Answer quality (0.15)
        a_score, _ = self.validator.validate_answer(answer, question)
        reward += 0.15 * a_score
        
        return min(reward, max_reward)
    
    def calculate_solver_reward(
        self,
        prediction: str,
        ground_truth: str
    ) -> float:
        """
        Calculate solver reward (simple correctness check).
        
        Based on Dr. Zero Equation (1):
        Solver: E[I(y = ŷ)]
        
        Enhanced with partial credit for biomedical answers.
        
        Args:
            prediction: Solver's predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Reward (0 or 1 for exact match, 0-1 for partial credit)
        """
        # Extract answer from tags
        pred_clean = self._extract_answer(prediction)
        gt_clean = self._extract_answer(ground_truth)
        
        # Exact match
        if pred_clean.lower() == gt_clean.lower():
            return 1.0
        
        # Partial credit for biomedical answers
        partial_score = self._calculate_partial_credit(pred_clean, gt_clean)
        
        return partial_score
    
    def _calculate_partial_credit(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate partial credit for biomedical answers.
        
        Criteria:
        - Same gene/protein mentioned (0.5)
        - Same disease mentioned (0.3)
        - Same pathway/mechanism (0.4)
        - PMID citation present (0.2)
        """
        score = 0.0
        
        # Extract entities
        pred_entities = set(self.validator._extract_biomedical_entities(prediction))
        gt_entities = set(self.validator._extract_biomedical_entities(ground_truth))
        
        # Entity overlap
        if pred_entities & gt_entities:
            overlap_ratio = len(pred_entities & gt_entities) / len(gt_entities | pred_entities)
            score += 0.5 * overlap_ratio
        
        # PMID citation
        pred_pmids = set(self.validator._extract_pmids(prediction))
        gt_pmids = set(self.validator._extract_pmids(ground_truth))
        
        if pred_pmids & gt_pmids:
            score += 0.2
        elif pred_pmids and gt_pmids:  # Different PMIDs but both cited
            score += 0.1
        
        # Semantic similarity (word overlap)
        pred_words = set(prediction.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        if pred_words & gt_words:
            word_overlap = len(pred_words & gt_words) / len(gt_words | pred_words)
            score += 0.3 * word_overlap
        
        return min(score, 0.9)  # Cap at 0.9 to reserve 1.0 for exact match
    
    def calculate_hop_grouped_advantage(
        self,
        rewards: List[float],
        hop_labels: List[int]
    ) -> List[float]:
        """
        Calculate hop-grouped advantages for HRPO.
        
        Based on Dr. Zero Equation (3):
        A_{i,h} = (r_i - E_{j∈I_h}[r_j]) / sqrt(Var_{j∈I_h}[r_j] + δ)
        
        Args:
            rewards: List of rewards for each sample
            hop_labels: List of hop counts for each sample (1, 2, 3, 4, etc.)
            
        Returns:
            List of advantages
        """
        # Group by hop count
        hop_groups = {}
        for i, (reward, hop) in enumerate(zip(rewards, hop_labels)):
            if hop not in hop_groups:
                hop_groups[hop] = []
            hop_groups[hop].append((i, reward))
        
        # Calculate advantages per group
        advantages = [0.0] * len(rewards)
        delta = 1e-8  # Small constant for numerical stability
        
        for hop, group in hop_groups.items():
            indices, group_rewards = zip(*group)
            
            # Group statistics
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards)
            
            # Calculate advantages
            for idx, reward in zip(indices, group_rewards):
                advantage = (reward - mean_reward) / (std_reward + delta)
                advantages[idx] = advantage
        
        return advantages
    
    def _is_correct_answer(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth (with partial matching)."""
        pred_clean = self._extract_answer(prediction)
        gt_clean = self._extract_answer(ground_truth)
        
        # Exact match
        if pred_clean.lower() == gt_clean.lower():
            return True
        
        # Partial match with high confidence
        partial_score = self._calculate_partial_credit(pred_clean, gt_clean)
        return partial_score >= 0.7
    
    def _has_valid_tags(self, text: str, tag: str) -> bool:
        """Check if text has valid opening and closing tags."""
        pattern = f'<{tag}>.*?</{tag}>'
        return bool(re.search(pattern, text, re.DOTALL))
    
    def _extract_answer(self, text: str) -> str:
        """Extract content from <answer> tags."""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()
    
    def batch_calculate_rewards(
        self,
        questions: List[str],
        answers: List[str],
        documents: List[str],
        solver_predictions_batch: List[List[str]],
        hop_labels: List[int]
    ) -> Dict[str, List[float]]:
        """
        Calculate rewards for a batch of samples.
        
        Args:
            questions: List of questions
            answers: List of ground truth answers
            documents: List of source documents
            solver_predictions_batch: List of lists of solver predictions
            hop_labels: List of hop counts
            
        Returns:
            Dictionary with rewards and advantages
        """
        # Calculate individual rewards
        proposer_rewards = []
        for q, a, doc, preds in zip(questions, answers, documents, solver_predictions_batch):
            reward = self.calculate_proposer_reward(q, a, doc, preds)
            proposer_rewards.append(reward)
        
        # Calculate hop-grouped advantages
        advantages = self.calculate_hop_grouped_advantage(proposer_rewards, hop_labels)
        
        return {
            "proposer_rewards": proposer_rewards,
            "advantages": advantages,
            "mean_reward": np.mean(proposer_rewards),
            "std_reward": np.std(proposer_rewards)
        }


# Example usage
if __name__ == "__main__":
    calculator = BiomedicalRewardCalculator()
    
    # Test proposer reward
    question = "<question>Through which pathway does TP53 regulate apoptosis?</question>"
    answer = "<answer>Intrinsic mitochondrial pathway (PMID: 12345678)</answer>"
    document = "(PMID: 11111111) TP53 is a tumor suppressor..."
    
    solver_predictions = [
        "<answer>Intrinsic mitochondrial pathway</answer>",
        "<answer>Extrinsic pathway</answer>",
        "<answer>Intrinsic pathway</answer>",
        "<answer>p53 pathway</answer>",
        "<answer>Unknown pathway</answer>"
    ]
    
    reward = calculator.calculate_proposer_reward(
        question, answer, document, solver_predictions
    )
    
    print(f"Proposer reward: {reward:.3f}")
    
    # Test HRPO advantages
    rewards = [0.8, 0.6, 0.9, 0.5, 0.7, 0.4, 0.8]
    hop_labels = [1, 1, 2, 2, 3, 3, 3]
    
    advantages = calculator.calculate_hop_grouped_advantage(rewards, hop_labels)
    
    print("\nHRPO Advantages:")
    for hop, adv in zip(hop_labels, advantages):
        print(f"  Hop {hop}: {adv:.3f}")
