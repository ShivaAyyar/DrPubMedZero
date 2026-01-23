# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Extended to support both Wikipedia and PubMed corpus downloads.

import argparse
import os
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Download corpus files for Dr. Zero training.")
parser.add_argument("--corpus_type", type=str, default="wikipedia", choices=["wikipedia", "pubmed"],
                    help="Type of corpus to download: wikipedia (default) or pubmed")
parser.add_argument("--repo_id", type=str, default="PeterJinGo/wiki-18-e5-index",
                    help="Hugging Face repository ID (for wikipedia)")
parser.add_argument("--save_path", type=str, required=True, help="Local directory to save files")
parser.add_argument("--email", type=str, default=None,
                    help="Email for NCBI Entrez API (required for pubmed)")
parser.add_argument("--query", type=str,
                    default="(cancer OR drug resistance) AND (gene OR protein OR pathway)",
                    help="PubMed search query (for pubmed corpus type)")
parser.add_argument("--max_results", type=int, default=50000,
                    help="Maximum number of PubMed abstracts to download")
parser.add_argument("--date_start", type=str, default="2019/01/01",
                    help="Start date for PubMed search (YYYY/MM/DD)")
parser.add_argument("--date_end", type=str, default="2024/12/31",
                    help="End date for PubMed search (YYYY/MM/DD)")

args = parser.parse_args()

if args.corpus_type == "wikipedia":
    # Original Wikipedia download code
    print("Downloading Wikipedia corpus...")

    repo_id = "PeterJinGo/wiki-18-e5-index"
    for file in ["part_aa", "part_ab"]:
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type="dataset",
            local_dir=args.save_path,
        )

    repo_id = "PeterJinGo/wiki-18-corpus"
    hf_hub_download(
            repo_id=repo_id,
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            local_dir=args.save_path,
    )

    print(f"Wikipedia corpus downloaded to {args.save_path}")

elif args.corpus_type == "pubmed":
    # PubMed download using biomedical module
    print("Downloading PubMed corpus...")

    if args.email is None:
        raise ValueError("Email is required for PubMed downloads. Use --email your@email.com")

    # Import biomedical module
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from biomedical import PubMedCorpusManager

    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    # Initialize manager and download
    manager = PubMedCorpusManager(save_path=args.save_path, email=args.email)

    print(f"  Query: {args.query}")
    print(f"  Date range: {args.date_start} to {args.date_end}")
    print(f"  Max results: {args.max_results}")

    articles = manager.download_pubmed_abstracts(
        query=args.query,
        max_results=args.max_results,
        date_range=(args.date_start, args.date_end)
    )

    # Save corpus
    corpus_path = manager.save_corpus(articles)

    print(f"PubMed corpus downloaded to {corpus_path}")
    print(f"   Total articles: {len(articles)}")
