# Dr. Zero Custom Reward Functions
# These modules can be imported independently without ray/verl dependencies

from .dr_zero_reward import (
    compute_format_reward,
    compute_format_reward_detailed,
    compute_difficulty_score,
    compute_difficulty_reward,
    compute_outcome_reward,
    compute_difficulty_reward_batch,
    compute_outcome_reward_batch,
    extract_question,
    extract_answer,
    normalize_answer,
    em_check,
)

from .hrpo_advantage import (
    compute_hrpo_advantages,
    compute_hrpo_advantages_numpy,
    compute_hrpo_advantages_with_baseline,
    compute_hrpo_policy_loss,
    create_hop_weighted_sampler,
    verify_hop_distribution,
    HRPOAdvantageEstimator,
)

from .hop_counter import (
    count_hops,
    count_hops_detailed,
    count_hops_batch,
    extract_hop_from_data_source,
    extract_hops_from_data_sources,
    generate_hop_distribution,
    assign_hop_to_seeds,
    hops_to_tensor,
    hops_to_numpy,
    extract_hops_from_batch,
    HopCounter,
)

__all__ = [
    # dr_zero_reward
    'compute_format_reward',
    'compute_format_reward_detailed',
    'compute_difficulty_score',
    'compute_difficulty_reward',
    'compute_outcome_reward',
    'compute_difficulty_reward_batch',
    'compute_outcome_reward_batch',
    'extract_question',
    'extract_answer',
    'normalize_answer',
    'em_check',
    # hrpo_advantage
    'compute_hrpo_advantages',
    'compute_hrpo_advantages_numpy',
    'compute_hrpo_advantages_with_baseline',
    'compute_hrpo_policy_loss',
    'create_hop_weighted_sampler',
    'verify_hop_distribution',
    'HRPOAdvantageEstimator',
    # hop_counter
    'count_hops',
    'count_hops_detailed',
    'count_hops_batch',
    'extract_hop_from_data_source',
    'extract_hops_from_data_sources',
    'generate_hop_distribution',
    'assign_hop_to_seeds',
    'hops_to_tensor',
    'hops_to_numpy',
    'extract_hops_from_batch',
    'HopCounter',
]
