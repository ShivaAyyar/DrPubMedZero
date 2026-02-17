"""
HRPO (Hop-grouped Relative Policy Optimization) Advantage Estimator

This module implements the hop-grouped advantage estimation from the Dr. Zero paper
(arXiv:2601.07055, Equations 2-3).

Key insight from paper (Section 3.2):
"For optimal proposer performance and training efficiency, we adopt a strictly
on-policy framework and omit ratio clipping."

HRPO differs from standard GRPO by:
1. Grouping questions by hop count for advantage normalization
2. NOT using ratio clipping (strictly on-policy)
3. Using KL coefficient = 0

Reference:
- Paper: Dr. Zero: Self-Evolving Search Agents without Training Data
- arXiv: https://arxiv.org/abs/2601.07055
- Equations 2-3, Section 3.2
"""

import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Union

logger = logging.getLogger(__name__)


def compute_hrpo_advantages(
    rewards: torch.Tensor,
    hop_counts: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Paper Equation 3: Standardize rewards over all h-hop questions in the batch.

    A_{i,h} = (r_i - E[r_j for j in I_h]) / sqrt(Var[r_j for j in I_h] + delta)

    Where I_h is the set of indices for all h-hop questions in the batch.

    This is the core innovation of HRPO: instead of normalizing across all samples
    or within a single prompt's responses (like GRPO), HRPO normalizes across
    structurally similar questions (grouped by hop count).

    Args:
        rewards: Tensor of shape (batch_size,) containing reward values
        hop_counts: Tensor of shape (batch_size,) containing hop count for each sample
        epsilon: Small value for numerical stability (delta in paper)

    Returns:
        advantages: Tensor of shape (batch_size,) with hop-grouped normalized advantages
    """
    if rewards.dim() != 1:
        raise ValueError(f"Expected 1D rewards tensor, got shape {rewards.shape}")
    if hop_counts.dim() != 1:
        raise ValueError(f"Expected 1D hop_counts tensor, got shape {hop_counts.shape}")
    if rewards.shape != hop_counts.shape:
        raise ValueError(f"Shape mismatch: rewards {rewards.shape} vs hop_counts {hop_counts.shape}")

    advantages = torch.zeros_like(rewards, dtype=torch.float32)

    # Get unique hop counts present in the batch (e.g., 1, 2, 3, 4)
    unique_hops = torch.unique(hop_counts)

    for h in unique_hops:
        # Create a mask for all queries in the batch with 'h' hops
        mask = (hop_counts == h)
        indices = torch.where(mask)[0]

        # Select rewards for this specific hop group
        group_rewards = rewards[mask]

        if len(group_rewards) > 1:
            # Paper Eq 3: Normalize within hop group
            mean = group_rewards.mean()
            std = group_rewards.std(unbiased=False)
            normalized = (group_rewards - mean) / (std + epsilon)
            advantages[indices] = normalized
        elif len(group_rewards) == 1:
            # Single sample in group: advantage = 0
            # Paper doesn't specify, but 0 is the safe default
            # (no relative comparison possible)
            advantages[indices] = 0.0
            logger.debug(f"Only 1 sample for hop={h.item()}, setting advantage to 0")

    return advantages


def compute_hrpo_advantages_numpy(
    rewards: np.ndarray,
    hop_counts: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    NumPy version of HRPO advantage computation.

    Args:
        rewards: Array of shape (batch_size,) containing reward values
        hop_counts: Array of shape (batch_size,) containing hop count for each sample
        epsilon: Small value for numerical stability

    Returns:
        advantages: Array of shape (batch_size,) with hop-grouped normalized advantages
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)

    unique_hops = np.unique(hop_counts)

    for h in unique_hops:
        mask = (hop_counts == h)
        group_rewards = rewards[mask]

        if len(group_rewards) > 1:
            mean = group_rewards.mean()
            std = group_rewards.std(ddof=0)  # Population std (unbiased=False)
            advantages[mask] = (group_rewards - mean) / (std + epsilon)
        else:
            advantages[mask] = 0.0

    return advantages


def compute_hrpo_advantages_with_baseline(
    rewards: torch.Tensor,
    hop_counts: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    HRPO advantage with optional baseline subtraction.

    This variant first subtracts a baseline (e.g., value function estimate)
    before performing hop-grouped normalization.

    A_{i,h} = ((r_i - b_i) - E[(r_j - b_j) for j in I_h]) / sqrt(Var + delta)

    Args:
        rewards: Tensor of shape (batch_size,) containing reward values
        hop_counts: Tensor of shape (batch_size,) containing hop count for each sample
        baseline: Optional tensor of shape (batch_size,) with baseline values
        epsilon: Small value for numerical stability

    Returns:
        advantages: Tensor of shape (batch_size,) with hop-grouped normalized advantages
    """
    if baseline is not None:
        adjusted_rewards = rewards - baseline
    else:
        adjusted_rewards = rewards

    return compute_hrpo_advantages(adjusted_rewards, hop_counts, epsilon)


# ============================================================================
# HRPO LOSS FUNCTIONS (for veRL integration)
# ============================================================================

def compute_hrpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    use_ratio_clip: bool = False,  # Paper: "omit ratio clipping"
    clip_ratio: float = 0.2
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute HRPO policy loss.

    Key difference from PPO/GRPO: Paper Section 3.2 states
    "we adopt a strictly on-policy framework and omit ratio clipping"

    This means HRPO uses simple policy gradient without clipping:
    L = -E[advantage * log_prob]

    Args:
        log_probs: Current policy log probabilities, shape (batch, seq_len)
        old_log_probs: Old policy log probabilities, shape (batch, seq_len)
        advantages: HRPO-computed advantages, shape (batch,)
        response_mask: Mask for response tokens, shape (batch, seq_len)
        use_ratio_clip: Whether to use ratio clipping (False for HRPO)
        clip_ratio: Clip ratio epsilon (only used if use_ratio_clip=True)

    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Sum log probs over response tokens
    response_log_probs = (log_probs * response_mask).sum(dim=-1)
    old_response_log_probs = (old_log_probs * response_mask).sum(dim=-1)

    # Compute ratio
    log_ratio = response_log_probs - old_response_log_probs
    ratio = torch.exp(log_ratio)

    # Expand advantages to match batch dimension
    if advantages.dim() == 1:
        adv = advantages.unsqueeze(-1) if response_log_probs.dim() > 1 else advantages
    else:
        adv = advantages

    if use_ratio_clip:
        # Standard PPO/GRPO clipped objective (NOT used for HRPO proposer)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss_unclipped = -adv * ratio
        policy_loss_clipped = -adv * clipped_ratio
        policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

        # Metrics
        clip_fraction = ((ratio - 1).abs() > clip_ratio).float().mean().item()
    else:
        # HRPO: Simple policy gradient without clipping
        policy_loss = -(adv * ratio).mean()
        clip_fraction = 0.0

    metrics = {
        "policy_loss": policy_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantage_mean": adv.mean().item(),
        "advantage_std": adv.std().item(),
        "clip_fraction": clip_fraction,
    }

    return policy_loss, metrics


# ============================================================================
# HOP DISTRIBUTION UTILITIES
# ============================================================================

def create_hop_weighted_sampler(
    hop_counts: List[int],
    target_ratio: Dict[int, float] = None
) -> List[float]:
    """
    Create sampling weights to achieve target hop distribution.

    Paper default ratio is 4:3:2:1 for 1-/2-/3-/4-hop questions.

    Args:
        hop_counts: List of hop counts for each sample in dataset
        target_ratio: Dict mapping hop count to target ratio
                     Default: {1: 4, 2: 3, 3: 2, 4: 1}

    Returns:
        List of sampling weights (same length as hop_counts)
    """
    if target_ratio is None:
        # Paper default: 4:3:2:1 for 1/2/3/4-hop
        target_ratio = {1: 4, 2: 3, 3: 2, 4: 1}

    # Count actual distribution
    from collections import Counter
    actual_counts = Counter(hop_counts)

    # Compute weights to achieve target distribution
    total_target = sum(target_ratio.values())
    weights = []

    for h in hop_counts:
        if h in target_ratio and actual_counts[h] > 0:
            # Weight to oversample/undersample to match target
            target_prop = target_ratio[h] / total_target
            actual_prop = actual_counts[h] / len(hop_counts)
            weight = target_prop / actual_prop
        else:
            weight = 1.0  # Default weight for unknown hop counts
        weights.append(weight)

    return weights


def verify_hop_distribution(
    hop_counts: Union[List[int], torch.Tensor, np.ndarray],
    target_ratio: Dict[int, float] = None
) -> Dict[str, any]:
    """
    Verify that hop counts match expected distribution.

    Args:
        hop_counts: Hop counts to verify
        target_ratio: Expected ratio (default: 4:3:2:1)

    Returns:
        Dict with distribution stats and warnings
    """
    if target_ratio is None:
        target_ratio = {1: 4, 2: 3, 3: 2, 4: 1}

    if isinstance(hop_counts, torch.Tensor):
        hop_counts = hop_counts.cpu().numpy()
    elif isinstance(hop_counts, list):
        hop_counts = np.array(hop_counts)

    from collections import Counter
    counts = Counter(hop_counts.tolist())
    total = len(hop_counts)

    # Compute actual ratios
    actual_ratios = {h: c / total for h, c in counts.items()}

    # Compute target ratios (normalized)
    total_target = sum(target_ratio.values())
    expected_ratios = {h: r / total_target for h, r in target_ratio.items()}

    # Check for deviations
    warnings = []
    for h in expected_ratios:
        expected = expected_ratios.get(h, 0)
        actual = actual_ratios.get(h, 0)
        deviation = abs(expected - actual) / (expected + 1e-8)
        if deviation > 0.2:  # More than 20% deviation
            warnings.append(
                f"Hop {h}: expected {expected:.2%}, got {actual:.2%} "
                f"(deviation: {deviation:.2%})"
            )

    return {
        "counts": dict(counts),
        "actual_ratios": actual_ratios,
        "expected_ratios": expected_ratios,
        "total_samples": total,
        "warnings": warnings,
        "is_valid": len(warnings) == 0
    }


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

class HRPOAdvantageEstimator:
    """
    Callable class for HRPO advantage estimation.

    This can be used as a drop-in replacement for veRL's advantage estimator
    in the proposer training loop.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        use_baseline: bool = False,
        target_hop_ratio: Dict[int, float] = None
    ):
        """
        Initialize HRPO advantage estimator.

        Args:
            epsilon: Numerical stability constant
            use_baseline: Whether to use baseline subtraction
            target_hop_ratio: Target distribution ratio (default 4:3:2:1)
        """
        self.epsilon = epsilon
        self.use_baseline = use_baseline
        self.target_hop_ratio = target_hop_ratio or {1: 4, 2: 3, 3: 2, 4: 1}

    def __call__(
        self,
        rewards: torch.Tensor,
        hop_counts: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute HRPO advantages.

        Args:
            rewards: Reward tensor
            hop_counts: Hop count tensor
            baseline: Optional baseline tensor

        Returns:
            Advantage tensor
        """
        if self.use_baseline and baseline is not None:
            return compute_hrpo_advantages_with_baseline(
                rewards, hop_counts, baseline, self.epsilon
            )
        else:
            return compute_hrpo_advantages(rewards, hop_counts, self.epsilon)

    def verify_distribution(self, hop_counts: torch.Tensor) -> Dict:
        """Verify hop distribution matches target ratio."""
        return verify_hop_distribution(hop_counts, self.target_hop_ratio)
