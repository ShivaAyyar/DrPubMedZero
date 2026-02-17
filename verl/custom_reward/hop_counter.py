"""
Hop Counter Utility for Dr. Zero HRPO Training

This module extracts hop counts from proposer responses to enable hop-grouped
advantage estimation in HRPO.

Hop count definition from Dr. Zero paper:
- 1-hop: Question answerable from a single document
- 2-hop: Requires reasoning across 2 documents/facts
- 3-hop: Requires reasoning across 3 documents/facts
- 4-hop: Requires reasoning across 4+ documents/facts

Implementation: Hop count = number of search tool calls + 1
(Each search advances to the next hop, starting from the initial document)

Reference:
- Paper: Dr. Zero: Self-Evolving Search Agents without Training Data
- arXiv: https://arxiv.org/abs/2601.07055
"""

import re
import json
import logging
from typing import List, Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)


# ============================================================================
# CORE HOP COUNTING FUNCTIONS
# ============================================================================

def count_hops(proposer_response: str) -> int:
    """
    Count the number of hops in a proposer's response.

    Hop count = number of search tool calls + 1
    (Each search advances to the next hop, starting from initial document)

    Paper uses 4:3:2:1 ratio for 1/2/3/4-hop questions.

    Args:
        proposer_response: The full response from the proposer model

    Returns:
        Hop count (minimum 1, maximum 4 for standard training)
    """
    # Pattern to match tool calls
    tool_call_pattern = r'<tool_call>\s*\{[^}]*"name"\s*:\s*"search"[^}]*\}\s*</tool_call>'

    # Count search tool calls
    tool_calls = re.findall(tool_call_pattern, proposer_response, re.DOTALL)
    num_searches = len(tool_calls)

    # Hops = searches + 1 (initial hop from document)
    hop_count = num_searches + 1

    # Clamp to reasonable range (paper uses 1-4 hops)
    return min(max(hop_count, 1), 4)


def count_hops_detailed(proposer_response: str) -> Tuple[int, Dict[str, Any]]:
    """
    Count hops with detailed breakdown.

    Args:
        proposer_response: The full response from the proposer model

    Returns:
        Tuple of (hop_count, details_dict)
    """
    details = {
        "search_calls": 0,
        "tool_responses": 0,
        "reasoning_sections": 0,
        "raw_searches": [],
    }

    # Count search tool calls
    tool_call_pattern = r'<tool_call>\s*(\{[^}]*\})\s*</tool_call>'
    tool_matches = re.findall(tool_call_pattern, proposer_response, re.DOTALL)

    for match in tool_matches:
        try:
            parsed = json.loads(match.strip())
            if parsed.get("name") == "search":
                details["search_calls"] += 1
                if "arguments" in parsed and "query_list" in parsed["arguments"]:
                    details["raw_searches"].append(parsed["arguments"]["query_list"])
        except json.JSONDecodeError:
            continue

    # Count tool responses (search results returned)
    tool_response_pattern = r'<tool_response>.*?</tool_response>'
    tool_responses = re.findall(tool_response_pattern, proposer_response, re.DOTALL)
    details["tool_responses"] = len(tool_responses)

    # Count reasoning sections
    think_pattern = r'<think>.*?</think>'
    think_sections = re.findall(think_pattern, proposer_response, re.DOTALL)
    details["reasoning_sections"] = len(think_sections)

    # Compute hop count
    hop_count = details["search_calls"] + 1
    hop_count = min(max(hop_count, 1), 4)

    return hop_count, details


def count_hops_batch(responses: List[str]) -> List[int]:
    """
    Batch version of hop counting.

    Args:
        responses: List of proposer responses

    Returns:
        List of hop counts
    """
    return [count_hops(resp) for resp in responses]


# ============================================================================
# HOP EXTRACTION FROM DATA SOURCE TAGS
# ============================================================================

def extract_hop_from_data_source(data_source: str) -> int:
    """
    Extract hop count from data_source field.

    The existing codebase uses data_source like "search_zero_2" where
    the last number indicates hop count.

    Args:
        data_source: Data source string (e.g., "search_zero_2", "search_biomedical_3")

    Returns:
        Hop count extracted from data source
    """
    try:
        # Pattern: extract last number after underscore
        match = re.search(r'_(\d+)$', data_source)
        if match:
            hop = int(match.group(1))
            return min(max(hop, 1), 4)
    except (ValueError, AttributeError):
        pass

    # Default to 1 if extraction fails
    logger.warning(f"Could not extract hop from data_source: {data_source}, defaulting to 1")
    return 1


def extract_hops_from_data_sources(data_sources: List[str]) -> List[int]:
    """
    Batch version of hop extraction from data sources.

    Args:
        data_sources: List of data source strings

    Returns:
        List of hop counts
    """
    return [extract_hop_from_data_source(ds) for ds in data_sources]


# ============================================================================
# HOP DISTRIBUTION GENERATION
# ============================================================================

def generate_hop_distribution(
    n_samples: int,
    ratio: Dict[int, int] = None
) -> List[int]:
    """
    Generate a list of hop counts following the target distribution.

    Paper default ratio is 4:3:2:1 for 1-/2-/3-/4-hop questions.

    Args:
        n_samples: Total number of samples to generate
        ratio: Dict mapping hop count to relative frequency
               Default: {1: 4, 2: 3, 3: 2, 4: 1}

    Returns:
        List of hop counts following the distribution
    """
    if ratio is None:
        ratio = {1: 4, 2: 3, 3: 2, 4: 1}

    # Normalize ratios
    total_ratio = sum(ratio.values())
    normalized = {h: r / total_ratio for h, r in ratio.items()}

    # Generate counts for each hop
    hop_counts = []
    remaining = n_samples

    sorted_hops = sorted(ratio.keys())
    for i, h in enumerate(sorted_hops):
        if i == len(sorted_hops) - 1:
            # Last hop gets remaining samples
            count = remaining
        else:
            count = int(n_samples * normalized[h])
            remaining -= count
        hop_counts.extend([h] * count)

    # Shuffle to avoid ordering bias
    import random
    random.shuffle(hop_counts)

    return hop_counts


def assign_hop_to_seeds(
    seeds: List[Dict],
    ratio: Dict[int, int] = None
) -> List[Dict]:
    """
    Assign hop counts to training seeds following target distribution.

    Args:
        seeds: List of seed dictionaries
        ratio: Target hop distribution ratio

    Returns:
        Seeds with 'hop' field added
    """
    hop_counts = generate_hop_distribution(len(seeds), ratio)

    for seed, hop in zip(seeds, hop_counts):
        seed['hop'] = hop
        # Also update data_source if present
        if 'data_source' in seed:
            base = seed['data_source'].rsplit('_', 1)[0]
            seed['data_source'] = f"{base}_{hop}"

    return seeds


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================

def hops_to_tensor(hop_counts: List[int]) -> "torch.Tensor":
    """
    Convert hop counts list to PyTorch tensor.

    Args:
        hop_counts: List of hop counts

    Returns:
        Tensor of hop counts (int64)
    """
    import torch
    return torch.tensor(hop_counts, dtype=torch.int64)


def hops_to_numpy(hop_counts: List[int]) -> "np.ndarray":
    """
    Convert hop counts list to NumPy array.

    Args:
        hop_counts: List of hop counts

    Returns:
        NumPy array of hop counts
    """
    import numpy as np
    return np.array(hop_counts, dtype=np.int64)


# ============================================================================
# INTEGRATION WITH VERL DATA STRUCTURES
# ============================================================================

def extract_hops_from_batch(batch: Any) -> List[int]:
    """
    Extract hop counts from a veRL DataProto batch.

    Tries multiple extraction methods:
    1. From 'hop' field in non_tensor_batch
    2. From 'data_source' field parsing
    3. From response content analysis

    Args:
        batch: veRL DataProto batch object

    Returns:
        List of hop counts
    """
    batch_size = len(batch)

    # Method 1: Direct hop field
    if hasattr(batch, 'non_tensor_batch') and 'hop' in batch.non_tensor_batch:
        hops = batch.non_tensor_batch['hop']
        if hasattr(hops, 'tolist'):
            return hops.tolist()
        return list(hops)

    # Method 2: Extract from data_source
    if hasattr(batch, 'non_tensor_batch') and 'data_source' in batch.non_tensor_batch:
        data_sources = batch.non_tensor_batch['data_source']
        if hasattr(data_sources, 'tolist'):
            data_sources = data_sources.tolist()
        return extract_hops_from_data_sources(data_sources)

    # Method 3: Count from responses (requires processing_class)
    logger.warning("Could not extract hops from batch metadata, defaulting to 1")
    return [1] * batch_size


class HopCounter:
    """
    Callable class for hop counting.

    Provides consistent hop counting interface for veRL integration.
    """

    def __init__(self, default_hop: int = 1, max_hop: int = 4):
        """
        Initialize HopCounter.

        Args:
            default_hop: Default hop count if extraction fails
            max_hop: Maximum hop count to return
        """
        self.default_hop = default_hop
        self.max_hop = max_hop

    def __call__(self, response: str) -> int:
        """Count hops in a single response."""
        hop = count_hops(response)
        return min(hop, self.max_hop)

    def count_batch(self, responses: List[str]) -> List[int]:
        """Count hops in a batch of responses."""
        return [self(resp) for resp in responses]

    def from_data_source(self, data_source: str) -> int:
        """Extract hop from data_source string."""
        return extract_hop_from_data_source(data_source)

    def from_data_sources(self, data_sources: List[str]) -> List[int]:
        """Extract hops from list of data_source strings."""
        return extract_hops_from_data_sources(data_sources)

    def from_batch(self, batch: Any) -> List[int]:
        """Extract hops from veRL batch."""
        return extract_hops_from_batch(batch)
