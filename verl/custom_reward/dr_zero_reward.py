"""
Dr. Zero Reward Functions (arXiv:2601.07055)

This module implements the exact reward functions from the Dr. Zero paper:
1. Difficulty-guided reward (Equation 4) for proposer training
2. Format reward (r^f) with 4 components worth 0.5 total
3. Outcome reward for solver training

Reference:
- Paper: Dr. Zero: Self-Evolving Search Agents without Training Data
- arXiv: https://arxiv.org/abs/2601.07055
"""

import re
import json
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


# ============================================================================
# FORMAT REWARD (r^f) - Paper Section A, Page 13
# ============================================================================

def compute_format_reward(text: str) -> float:
    """
    Computes r^f (Format Reward) with max value 0.5.

    Paper Section A (Page 13):
    "We define four requirements:
    (1) adherence to the <think>...</think> structure
    (2) valid tool usage, including correct tool call and arguments
    (3) an extractable question enclosed in <question>...</question> tags
    (4) an extractable answer within <answer>...</answer> tags"

    Each component: 0.125, total max: 0.5

    Args:
        text: The full response from the proposer model

    Returns:
        Format reward score between 0.0 and 0.5
    """
    score = 0.0

    # Component 1: <think> structure (0.125)
    if "<think>" in text and "</think>" in text:
        # Verify it's properly formatted (not empty)
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match and len(think_match.group(1).strip()) > 0:
            score += 0.125

    # Component 2: Valid tool call syntax (0.125)
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_matches = re.findall(tool_call_pattern, text, re.DOTALL)
    if tool_matches:
        # Validate at least one tool call has valid JSON with proper structure
        for match in tool_matches:
            try:
                parsed = json.loads(match.strip())
                if (isinstance(parsed, dict) and
                    "name" in parsed and
                    isinstance(parsed.get("arguments"), dict)):
                    score += 0.125
                    break  # Only need one valid tool call
            except json.JSONDecodeError:
                continue

    # Component 3: <question> tags (0.125)
    if "<question>" in text and "</question>" in text:
        question_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
        if question_match and len(question_match.group(1).strip()) > 0:
            score += 0.125

    # Component 4: <answer> tags (0.125)
    if "<answer>" in text and "</answer>" in text:
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match and len(answer_match.group(1).strip()) > 0:
            score += 0.125

    return score


def compute_format_reward_detailed(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Detailed version that returns breakdown of format reward components.

    Args:
        text: The full response from the proposer model

    Returns:
        Tuple of (total_score, component_breakdown_dict)
    """
    components = {
        "think_structure": 0.0,
        "tool_call": 0.0,
        "question_tags": 0.0,
        "answer_tags": 0.0
    }

    # Component 1: <think> structure
    if "<think>" in text and "</think>" in text:
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match and len(think_match.group(1).strip()) > 0:
            components["think_structure"] = 0.125

    # Component 2: Valid tool call syntax
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_matches = re.findall(tool_call_pattern, text, re.DOTALL)
    if tool_matches:
        for match in tool_matches:
            try:
                parsed = json.loads(match.strip())
                if (isinstance(parsed, dict) and
                    "name" in parsed and
                    isinstance(parsed.get("arguments"), dict)):
                    components["tool_call"] = 0.125
                    break
            except json.JSONDecodeError:
                continue

    # Component 3: <question> tags
    if "<question>" in text and "</question>" in text:
        question_match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
        if question_match and len(question_match.group(1).strip()) > 0:
            components["question_tags"] = 0.125

    # Component 4: <answer> tags
    if "<answer>" in text and "</answer>" in text:
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match and len(answer_match.group(1).strip()) > 0:
            components["answer_tags"] = 0.125

    total = sum(components.values())
    return total, components


# ============================================================================
# DIFFICULTY REWARD - Paper Equation 4
# ============================================================================

def compute_difficulty_score(binary_list: List[int]) -> float:
    """
    Compute difficulty score from binary correctness list.

    Paper Equation 4 (difficulty component):
    I(0 < k < n) × (n-k)/(n-1)

    Where:
    - k = number of correct solver responses (sum of binary_list)
    - n = total number of solver rollouts (len of binary_list)

    This rewards questions that are:
    - Not too easy (k < n, some solvers fail)
    - Not too hard (k > 0, some solvers succeed)
    - Maximized when k = 1 (only one solver gets it right)

    Args:
        binary_list: List of 0/1 values indicating solver success

    Returns:
        Difficulty score between 0.0 and 1.0
    """
    if len(binary_list) <= 1:
        logger.warning("Difficulty score requires at least 2 rollouts")
        return 0.0

    n = len(binary_list)
    k = sum(binary_list)  # Number of correct responses

    # Indicator function: I(0 < k < n)
    if k == 0 or k == n:
        return 0.0

    # (n - k) / (n - 1)
    return (n - k) / (n - 1)


def compute_difficulty_reward(
    proposer_response: str,
    ground_truth_answer: str,
    solver_outputs: List[str],
    n: int = 5,
    match_fn: Optional[callable] = None
) -> float:
    """
    Dr. Zero Proposer Reward (Paper Equation 4).

    r(y, {ŷ_i}^n_{i=1}) = I(0 < k < n) × (n-k)/(n-1) + r^f

    Where:
    - k = number of correct solver responses out of n attempts
    - n = number of solver rollouts (default 5)
    - r^f = format reward (max 0.5)

    Args:
        proposer_response: Full response from proposer (for format reward)
        ground_truth_answer: The answer generated by proposer
        solver_outputs: List of n solver response strings
        n: Expected number of solver rollouts (default 5)
        match_fn: Optional custom matching function (defaults to substring match)

    Returns:
        Total reward (difficulty + format), range [0, 1.5]
    """
    if len(solver_outputs) != n:
        logger.warning(f"Expected {n} solver outputs, got {len(solver_outputs)}")
        n = len(solver_outputs)

    if n == 0:
        # No solver outputs, return format reward only
        return compute_format_reward(proposer_response)

    # Default matching function: case-insensitive substring match
    if match_fn is None:
        def match_fn(prediction: str, answer: str) -> bool:
            pred_norm = prediction.strip().lower()
            ans_norm = answer.strip().lower()
            return ans_norm in pred_norm

    # Calculate k: number of correct solver answers
    binary_results = [1 if match_fn(output, ground_truth_answer) else 0
                      for output in solver_outputs]
    k = sum(binary_results)

    # Difficulty Score (Equation 4 main term)
    if len(binary_results) > 1:
        r_difficulty = compute_difficulty_score(binary_results)
    else:
        r_difficulty = 0.0

    # Format Score (computed on PROPOSER's output)
    r_format = compute_format_reward(proposer_response)

    total_reward = r_difficulty + r_format

    logger.debug(
        f"Difficulty reward: k={k}/{n}, "
        f"r_diff={r_difficulty:.4f}, r_format={r_format:.4f}, "
        f"total={total_reward:.4f}"
    )

    return total_reward


# ============================================================================
# OUTCOME REWARD - Paper Section 3.3 (Solver Training)
# ============================================================================

def compute_outcome_reward(
    ground_truth_answer: str,
    solver_output: str,
    match_fn: Optional[callable] = None
) -> float:
    """
    Solver Outcome Reward (Paper Section 3.3).

    Simple indicator function: I(y = ŷ)

    Used for training the solver model with GRPO.

    Args:
        ground_truth_answer: The correct answer
        solver_output: The solver's prediction
        match_fn: Optional custom matching function

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    if match_fn is None:
        # Default: case-insensitive substring match
        ground_truth_normalized = ground_truth_answer.strip().lower()
        solver_normalized = solver_output.strip().lower()
        return 1.0 if ground_truth_normalized in solver_normalized else 0.0
    else:
        return 1.0 if match_fn(solver_output, ground_truth_answer) else 0.0


# ============================================================================
# BATCH REWARD FUNCTIONS (for veRL integration)
# ============================================================================

def compute_difficulty_reward_batch(
    proposer_responses: List[str],
    ground_truth_answers: List[str],
    solver_outputs_batch: List[List[str]],
    n: int = 5
) -> List[float]:
    """
    Batch version of compute_difficulty_reward.

    Args:
        proposer_responses: List of proposer responses
        ground_truth_answers: List of ground truth answers
        solver_outputs_batch: List of lists, where each inner list contains
                             n solver outputs for the corresponding proposer
        n: Number of solver rollouts per proposer (default 5)

    Returns:
        List of reward scores for each proposer
    """
    rewards = []
    for proposer_resp, gt_answer, solver_outputs in zip(
        proposer_responses, ground_truth_answers, solver_outputs_batch
    ):
        reward = compute_difficulty_reward(
            proposer_response=proposer_resp,
            ground_truth_answer=gt_answer,
            solver_outputs=solver_outputs,
            n=n
        )
        rewards.append(reward)
    return rewards


def compute_outcome_reward_batch(
    ground_truth_answers: List[str],
    solver_outputs: List[str]
) -> List[float]:
    """
    Batch version of compute_outcome_reward.

    Args:
        ground_truth_answers: List of correct answers
        solver_outputs: List of solver predictions

    Returns:
        List of reward scores (0.0 or 1.0)
    """
    return [
        compute_outcome_reward(gt, pred)
        for gt, pred in zip(ground_truth_answers, solver_outputs)
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_question(text: str) -> Optional[str]:
    """Extract question from <question> tags."""
    match = re.search(r'<question>(.*?)</question>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer(text: str) -> Optional[str]:
    """Extract answer from <answer> tags."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else None


def normalize_answer(s: str) -> str:
    """
    Normalize answer for comparison (from original Dr. Zero code).

    Removes articles, punctuation, extra whitespace, and lowercases.
    """
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: Any) -> int:
    """
    Exact match check (from original Dr. Zero code).

    Args:
        prediction: Model's prediction
        golden_answers: Ground truth (string or list of strings)

    Returns:
        1 if exact match found, 0 otherwise
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    normalized_prediction = normalize_answer(prediction)

    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0
