"""
Biomedical adaptation of Dr. Zero for PubMed literature search.

This module provides:
- PubMed corpus download and preprocessing
- Biomedical entity validation
- Domain-specific prompts and rewards
- BiomedQA evaluation datasets
- Google Colab training utilities
"""

__version__ = "0.1.0"

from .pubmed_corpus import PubMedCorpusManager
from .biomedical_validator import BiomedicalValidator
from .biomedical_retriever import BiomedicalRetrieverServer, build_biomedical_index
from .biomedical_prompts import BiomedicalPrompts
from .biomedical_rewards import BiomedicalRewardCalculator
from .biomedical_datasets import BiomedicalDatasets, download_sample_datasets

__all__ = [
    'PubMedCorpusManager',
    'BiomedicalValidator',
    'BiomedicalRetrieverServer',
    'build_biomedical_index',
    'BiomedicalPrompts',
    'BiomedicalRewardCalculator',
    'BiomedicalDatasets',
    'download_sample_datasets',
]

# Colab-specific helper
def setup_for_colab():
    """
    Setup biomedical module for Google Colab environment.

    This function:
    - Verifies GPU availability
    - Sets up logging
    - Configures paths for Colab

    Returns:
        bool: True if setup successful
    """
    import os
    import sys

    print("Setting up biomedical module for Google Colab...")

    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False
        print("  Warning: Not running in Google Colab")

    # Verify GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ GPU: {gpu_name}")
        else:
            print("  ⚠️ No GPU detected")
            return False
    except ImportError:
        print("  ⚠️ PyTorch not installed")
        return False

    # Setup logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("  ✓ Biomedical module ready for Colab")
    return True
