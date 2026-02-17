#!/usr/bin/env python3
"""
Dr. Zero Full 3-Iteration Training Script

This script runs the complete Dr. Zero training pipeline:
1. Downloads PubMed corpus (if needed)
2. Builds FAISS index (if needed)
3. Prepares training seeds with hop distribution
4. Runs 3 iterations of proposer-solver co-evolution

Usage:
    python train_drzero_full.py --email your@email.com [--corpus_size 200000] [--iterations 3]

Reference: Dr. Zero paper (arXiv:2601.07055)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    # Model
    'model_name': 'Qwen/Qwen2.5-3B-Instruct',

    # Corpus
    'corpus_size': 200000,
    'training_seeds': 5000,
    'pubmed_query': '(cancer OR diabetes OR alzheimer OR cardiovascular) AND (gene OR protein OR mutation)',
    'date_range': ('2020/01/01', '2024/12/31'),

    # Training (from Paper Tables 5-6)
    'micro_batch_size': 32,
    'gradient_accumulation': 8,
    'proposer_lr': 1e-6,
    'solver_lr': 1e-6,
    'steps_per_iteration': 50,
    'num_iterations': 3,
    'save_freq': 25,

    # HRPO/GRPO parameters
    'proposer_group_size': 1,
    'solver_group_size': 5,
    'reward_rollout_n': 5,
    'proposer_kl_coef': 0.0,
    'solver_kl_coef': 0.001,
    'solver_clip_ratio': 0.2,
    'hop_ratio': {1: 4, 2: 3, 3: 2, 4: 1},

    # Sequence lengths
    'proposer_max_seq_len': 4096,
    'solver_max_seq_len': 3072,
    'max_turns': 5,

    # Server ports
    'retrieval_port': 8000,
    'solver_port': 8001,
}

# =============================================================================
# UTILITIES
# =============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_step(step_num, total, description):
    print(f"\n[{step_num}/{total}] {description}")
    print("-" * 50)

def wait_for_server(url, timeout=300, check_interval=5):
    """Wait for a server to become ready."""
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(check_interval)
    return False

def kill_process(pid):
    """Kill a process by PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        os.kill(pid, signal.SIGKILL)
    except:
        pass

# =============================================================================
# PHASE 1: DATA PREPARATION
# =============================================================================

def download_corpus(config):
    """Download PubMed corpus if needed."""
    from biomedical import PubMedCorpusManager

    corpus_file = Path(config['corpus_file'])

    if corpus_file.exists():
        with open(corpus_file) as f:
            n_papers = sum(1 for _ in f)
        if n_papers >= config['corpus_size'] * 0.9:
            print(f"  Corpus exists: {n_papers:,} papers. Skipping download.")
            return True
        print(f"  Corpus incomplete: {n_papers:,} papers. Downloading more...")

    print(f"  Downloading {config['corpus_size']:,} PubMed papers...")
    print(f"  Query: {config['pubmed_query'][:50]}...")

    manager = PubMedCorpusManager(
        save_path=str(corpus_file.parent),
        email=config['email']
    )

    articles = manager.download_pubmed_abstracts(
        query=config['pubmed_query'],
        max_results=config['corpus_size'],
        date_range=config['date_range']
    )

    if articles:
        manager.save_corpus(articles)
        print(f"  Downloaded {len(articles):,} papers")
        return True
    else:
        print("  ERROR: Download failed")
        return False

def build_index(config):
    """Build FAISS index if needed."""
    import torch
    from biomedical.biomedical_retriever import BiomedicalRetrieverServer

    index_file = Path(config['index_file'])
    corpus_file = Path(config['corpus_file'])

    if index_file.exists():
        print(f"  Index exists: {index_file}. Skipping build.")
        return True

    print(f"  Building PubMedBERT FAISS index...")
    print(f"  This may take 1-2 hours for large corpora.")

    retriever = BiomedicalRetrieverServer(
        corpus_path=str(corpus_file),
        index_path=str(index_file),
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        device="cuda" if torch.cuda.is_available() else "cpu",
        topk=3
    )

    # Test search
    results = retriever.search("BRCA1 mutation breast cancer")
    print(f"  Test search returned {len(results)} results")

    del retriever
    torch.cuda.empty_cache()

    size_gb = os.path.getsize(index_file) / 1e9
    print(f"  Index built: {size_gb:.2f} GB")
    return True

def prepare_training_seeds(config):
    """Prepare training seeds with hop distribution."""
    import pandas as pd

    parquet_file = Path(config['train_parquet'])
    corpus_file = Path(config['corpus_file'])

    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        print(f"  Training data exists: {len(df)} examples")
        hops = df['data_source'].apply(lambda x: int(x.split('_')[-1]) if '_' in x else 1)
        print(f"  Hop distribution: {hops.value_counts().sort_index().to_dict()}")
        return True

    print(f"  Preparing {config['training_seeds']} training seeds...")
    print(f"  Target hop ratio: {config['hop_ratio']}")

    # Load corpus
    corpus = []
    with open(corpus_file) as f:
        for line in f:
            corpus.append(json.loads(line))
    print(f"  Loaded {len(corpus):,} papers")

    # Filter for substantial abstracts
    substantial = [p for p in corpus if len(p.get('abstract', '').split()) > 100]
    print(f"  {len(substantial):,} have >100 word abstracts")

    # Sample seeds
    source = substantial if len(substantial) >= config['training_seeds'] else corpus
    seeds = random.sample(source, min(config['training_seeds'], len(source)))

    # Generate hop distribution
    hop_ratio = config['hop_ratio']
    total_ratio = sum(hop_ratio.values())
    hop_counts = []
    for h, r in sorted(hop_ratio.items()):
        count = int(len(seeds) * r / total_ratio)
        hop_counts.extend([h] * count)
    while len(hop_counts) < len(seeds):
        hop_counts.append(1)
    random.shuffle(hop_counts)

    # Convert to veRL format
    data = []
    for idx, (seed, hop) in enumerate(zip(seeds, hop_counts)):
        title = seed.get('title', '')
        abstract = seed.get('abstract', seed.get('text', ''))[:800]
        pmid = seed.get('pmid', str(idx))

        user_msg = f"Generate a challenging {hop}-hop biomedical question from this paper:\n\nTitle: {title}\n\nAbstract: {abstract}"

        data.append({
            'prompt': [{'role': 'user', 'content': user_msg}],
            'data_source': f'search_biomedical_{hop}',
            'extra_info': {
                'index': idx,
                'pmid': pmid,
                'hop': hop,
                'tools_kwargs': {},
                'interaction_kwargs': {}
            }
        })

    df = pd.DataFrame(data)
    parquet_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_file)

    # Report distribution
    hop_dist = Counter(hop_counts)
    print(f"  Hop distribution:")
    for h in sorted(hop_dist.keys()):
        pct = hop_dist[h] / len(hop_counts) * 100
        print(f"    {h}-hop: {hop_dist[h]} ({pct:.1f}%)")

    print(f"  Saved {len(df)} training seeds")
    return True

# =============================================================================
# PHASE 2: SERVER MANAGEMENT
# =============================================================================

def start_retrieval_server(config):
    """Start the retrieval server."""
    print("  Starting retrieval server on port 8000...")

    repo_dir = Path(config['repo_dir'])

    cmd = [
        sys.executable, str(repo_dir / "search" / "retrieval_server.py"),
        "--mode=biomedical",
        f"--index_path={config['index_file']}",
        f"--corpus_path={config['corpus_file']}",
        "--retriever_model=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "--topk=3",
        f"--port={config['retrieval_port']}"
    ]

    log_file = open('/tmp/retrieval_server.log', 'w')
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(repo_dir))

    print(f"  PID: {proc.pid}")
    print("  Waiting for server to be ready...")

    if wait_for_server(f"http://127.0.0.1:{config['retrieval_port']}/health", timeout=180):
        print("  Retrieval server ready!")
        return proc.pid, log_file
    else:
        print("  WARNING: Server may not be ready. Check /tmp/retrieval_server.log")
        return proc.pid, log_file

def start_solver_server(config, model_path=None):
    """Start the solver server."""
    model = model_path or config['model_name']
    print(f"  Starting solver server on port 8001...")
    print(f"  Model: {model}")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        f"--model-path={model}",
        f"--port={config['solver_port']}",
        "--tool-call-parser=qwen25",
        "--mem-fraction-static=0.35",
        "--tp-size=1",
        "--dp-size=1",
    ]

    log_file = open('/tmp/solver_server.log', 'w')
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    print(f"  PID: {proc.pid}")
    print("  Waiting for server to be ready (3-5 minutes)...")

    if wait_for_server(f"http://127.0.0.1:{config['solver_port']}/health", timeout=360):
        print("  Solver server ready!")
        return proc.pid, log_file
    else:
        print("  WARNING: Server may not be ready. Check /tmp/solver_server.log")
        return proc.pid, log_file

def stop_server(pid, log_file):
    """Stop a server."""
    kill_process(pid)
    if log_file:
        log_file.close()

# =============================================================================
# PHASE 3: TRAINING
# =============================================================================

def train_proposer(config, iteration, solver_model=None):
    """Train proposer with HRPO."""
    print(f"\n  Training Proposer (HRPO) - Iteration {iteration}")
    print(f"  Algorithm: HRPO (no ratio clipping, KL=0)")
    print(f"  Steps: {config['steps_per_iteration']}")

    repo_dir = Path(config['repo_dir'])
    checkpoint_dir = Path(config['checkpoints_dir']) / f'iter{iteration}' / 'proposer'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path", str(repo_dir / "config"),
        "--config-name", "search_multiturn_grpo",

        # Data
        f"data.train_files={config['train_parquet']}",
        f"data.train_batch_size={config['micro_batch_size']}",
        f"trainer.gradient_accumulation_steps={config['gradient_accumulation']}",
        "data.max_prompt_length=1536",
        "data.max_response_length=2560",
        f"data.max_seq_length={config['proposer_max_seq_len']}",

        # Model
        f"actor_rollout_ref.model.path={config['model_name']}",
        f"actor_rollout_ref.actor.optim.lr={config['proposer_lr']}",
        "actor_rollout_ref.actor.grad_clip=0.1",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03",
        "actor_rollout_ref.actor.optim.weight_decay=0.01",

        # HRPO: No ratio clipping, KL=0
        "algorithm.use_kl_in_reward=False",
        "algorithm.adv_estimator=grpo_batch",
        f"actor_rollout_ref.rollout.n={config['proposer_group_size']}",
        "actor_rollout_ref.actor.use_kl_loss=False",

        # Single GPU
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",

        # Memory optimization
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",

        # Multi-turn tool use
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={repo_dir}/config/search_tool_config.yaml",
        f"actor_rollout_ref.rollout.max_turns={config['max_turns']}",

        # Reward function
        "reward_model.reward_manager=batch",
        "custom_reward_function.name=compute_biomedical_challenger_score_batch",
        "custom_reward_function.path=verl/custom_reward/reward_function.py",
        f"custom_reward_function.reward_kwargs.model_name={config['model_name']}",
        f"custom_reward_function.reward_kwargs.base_url=http://127.0.0.1:{config['solver_port']}",
        f"custom_reward_function.reward_kwargs.reward_rollout_n={config['reward_rollout_n']}",

        # Training schedule
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={config['steps_per_iteration']}",
        f"trainer.save_freq={config['save_freq']}",
        f"trainer.default_hdfs_dir={checkpoint_dir}",
        f"trainer.project_name=drzero-iter{iteration}",
        f"trainer.experiment_name=proposer_iter{iteration}",
        "trainer.logger=[\"console\"]",
    ]

    result = subprocess.run(cmd, cwd=str(repo_dir))

    if result.returncode == 0:
        ckpt = checkpoint_dir / f'global_step_{config["steps_per_iteration"]}'
        print(f"  Proposer training complete: {ckpt}")
        return str(ckpt)
    else:
        print(f"  ERROR: Proposer training failed")
        return None

def generate_data(config, iteration, proposer_ckpt):
    """Generate QA data using trained proposer."""
    print(f"\n  Generating data with Iteration {iteration} Proposer")

    repo_dir = Path(config['repo_dir'])
    output_parquet = Path(config['checkpoints_dir']).parent / 'data' / f'iter{iteration}_generated.parquet'
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "verl.trainer.main_generation",
        "--config-path", str(repo_dir / "config"),
        "--config-name", "search_multiturn_grpo",

        f"+ckpt_path={proposer_ckpt}",
        f"+data.path={config['train_parquet']}",
        f"+data.output_path={output_parquet}",
        "+data.batch_size=64",

        f"actor_rollout_ref.rollout.n={config['solver_group_size']}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.8",
    ]

    result = subprocess.run(cmd, cwd=str(repo_dir))

    if result.returncode == 0:
        print(f"  Data generation complete: {output_parquet}")
        return str(output_parquet)
    else:
        print(f"  ERROR: Data generation failed")
        return None

def train_solver(config, iteration, generated_data):
    """Train solver with GRPO."""
    print(f"\n  Training Solver (GRPO) - Iteration {iteration}")
    print(f"  Algorithm: GRPO (ratio clipping ε=0.2, KL=0.001)")
    print(f"  Steps: {config['steps_per_iteration']}")

    repo_dir = Path(config['repo_dir'])
    checkpoint_dir = Path(config['checkpoints_dir']) / f'iter{iteration}' / 'solver'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path", str(repo_dir / "config"),
        "--config-name", "search_multiturn_grpo",

        # Data
        f"data.train_files={generated_data}",
        f"data.train_batch_size={config['micro_batch_size']}",
        f"trainer.gradient_accumulation_steps={config['gradient_accumulation']}",
        "data.max_prompt_length=512",
        f"data.max_seq_length={config['solver_max_seq_len']}",

        # Model
        f"actor_rollout_ref.model.path={config['model_name']}",
        f"actor_rollout_ref.actor.optim.lr={config['solver_lr']}",
        "actor_rollout_ref.actor.grad_clip=0.1",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03",
        "actor_rollout_ref.actor.optim.weight_decay=0.01",

        # GRPO: With ratio clipping
        "algorithm.adv_estimator=grpo",
        f"actor_rollout_ref.rollout.n={config['solver_group_size']}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={config['solver_kl_coef']}",
        f"actor_rollout_ref.actor.clip_ratio={config['solver_clip_ratio']}",

        # Single GPU
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",

        # Memory optimization
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",

        # Multi-turn
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={repo_dir}/config/search_tool_config.yaml",
        f"actor_rollout_ref.rollout.max_turns={config['max_turns']}",

        # Training schedule
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={config['steps_per_iteration']}",
        f"trainer.save_freq={config['save_freq']}",
        f"trainer.default_hdfs_dir={checkpoint_dir}",
        f"trainer.project_name=drzero-iter{iteration}",
        f"trainer.experiment_name=solver_iter{iteration}",
        "trainer.logger=[\"console\"]",
    ]

    result = subprocess.run(cmd, cwd=str(repo_dir))

    if result.returncode == 0:
        ckpt = checkpoint_dir / f'global_step_{config["steps_per_iteration"]}'
        print(f"  Solver training complete: {ckpt}")
        return str(ckpt)
    else:
        print(f"  ERROR: Solver training failed")
        return None

def convert_to_hf(config, iteration, solver_ckpt):
    """Convert solver checkpoint to HuggingFace format."""
    print(f"\n  Converting Solver Iteration {iteration} to HuggingFace format")

    output_dir = Path(config['checkpoints_dir']) / f'iter{iteration}' / 'solver' / f'solver_iter{iteration}_hf'

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", f"{solver_ckpt}/actor",
        "--target_dir", str(output_dir),
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"  Conversion complete: {output_dir}")
        return str(output_dir)
    else:
        print(f"  ERROR: Conversion failed")
        return None

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def find_resume_point(config):
    """Find the last completed checkpoint to resume from."""
    checkpoints_dir = Path(config['checkpoints_dir'])

    # Check each iteration in reverse order
    for iteration in range(config['num_iterations'], 0, -1):
        solver_hf = checkpoints_dir / f'iter{iteration}' / 'solver' / f'solver_iter{iteration}_hf' / 'config.json'
        if solver_hf.exists():
            print(f"  Found completed iteration {iteration}")
            return iteration, 'complete'

        solver_ckpt = checkpoints_dir / f'iter{iteration}' / 'solver'
        if list(solver_ckpt.glob('global_step_*')):
            print(f"  Found solver checkpoint at iteration {iteration}")
            return iteration, 'solver'

        generated_data = checkpoints_dir.parent / 'data' / f'iter{iteration}_generated.parquet'
        if generated_data.exists():
            print(f"  Found generated data at iteration {iteration}")
            return iteration, 'data'

        proposer_ckpt = checkpoints_dir / f'iter{iteration}' / 'proposer'
        if list(proposer_ckpt.glob('global_step_*')):
            print(f"  Found proposer checkpoint at iteration {iteration}")
            return iteration, 'proposer'

    return 0, 'start'

def run_training(config):
    """Run the full 3-iteration training pipeline."""

    print_header("DR. ZERO FULL TRAINING PIPELINE")
    print(f"Iterations: {config['num_iterations']}")
    print(f"Steps per iteration: {config['steps_per_iteration']}")
    print(f"Corpus size: {config['corpus_size']:,}")
    print(f"Training seeds: {config['training_seeds']:,}")
    print(f"Effective batch size: {config['micro_batch_size'] * config['gradient_accumulation']}")

    # Check for resume point
    print("\nChecking for existing progress...")
    resume_iter, resume_stage = find_resume_point(config)
    if resume_iter > 0:
        print(f"  Resuming from iteration {resume_iter}, stage: {resume_stage}")
    else:
        print("  Starting fresh training")

    total_steps = 7 + (config['num_iterations'] * 4)  # Prep + iterations
    current_step = 0

    # =========================================================================
    # PHASE 1: DATA PREPARATION
    # =========================================================================

    print_header("PHASE 1: DATA PREPARATION")

    current_step += 1
    print_step(current_step, total_steps, "Downloading PubMed corpus")
    if not download_corpus(config):
        return False

    current_step += 1
    print_step(current_step, total_steps, "Building FAISS index")
    if not build_index(config):
        return False

    current_step += 1
    print_step(current_step, total_steps, "Preparing training seeds")
    if not prepare_training_seeds(config):
        return False

    # =========================================================================
    # PHASE 2: START SERVERS
    # =========================================================================

    print_header("PHASE 2: STARTING SERVERS")

    current_step += 1
    print_step(current_step, total_steps, "Starting retrieval server")
    retrieval_pid, retrieval_log = start_retrieval_server(config)

    current_step += 1
    print_step(current_step, total_steps, "Starting solver server (base model)")
    solver_pid, solver_log = start_solver_server(config)

    # Track server PIDs for cleanup
    server_pids = [(retrieval_pid, retrieval_log), (solver_pid, solver_log)]

    try:
        # =====================================================================
        # PHASE 3: TRAINING ITERATIONS
        # =====================================================================

        solver_hf_path = None  # Will be set after each iteration

        # Find solver HF path from previous iterations if resuming
        if resume_iter > 0:
            for i in range(resume_iter, 0, -1):
                prev_hf = Path(config['checkpoints_dir']) / f'iter{i}' / 'solver' / f'solver_iter{i}_hf'
                if (prev_hf / 'config.json').exists():
                    solver_hf_path = str(prev_hf)
                    print(f"\n  Using solver from iteration {i}: {solver_hf_path}")
                    break

        start_iteration = 1
        if resume_iter > 0 and resume_stage == 'complete':
            start_iteration = resume_iter + 1

        for iteration in range(start_iteration, config['num_iterations'] + 1):
            print_header(f"ITERATION {iteration} of {config['num_iterations']}")

            # Determine what stage to start at for this iteration
            skip_proposer = False
            skip_data = False
            skip_solver = False

            if iteration == resume_iter:
                if resume_stage == 'solver':
                    skip_proposer = True
                    skip_data = True
                elif resume_stage == 'data':
                    skip_proposer = True
                elif resume_stage == 'proposer':
                    pass  # Start from proposer

            # If not first iteration, restart solver server with trained model
            if iteration > 1 and solver_hf_path:
                print("\n  Restarting solver server with trained model...")
                stop_server(solver_pid, solver_log)
                time.sleep(5)
                solver_pid, solver_log = start_solver_server(config, solver_hf_path)
                server_pids[-1] = (solver_pid, solver_log)

            # Train proposer
            current_step += 1
            if skip_proposer:
                print_step(current_step, total_steps, f"Iteration {iteration}: Train Proposer (SKIPPED - already done)")
                ckpt_dir = Path(config['checkpoints_dir']) / f'iter{iteration}' / 'proposer'
                ckpts = sorted(ckpt_dir.glob('global_step_*'))
                proposer_ckpt = str(ckpts[-1]) if ckpts else None
            else:
                print_step(current_step, total_steps, f"Iteration {iteration}: Train Proposer (HRPO)")
                proposer_ckpt = train_proposer(config, iteration)
                if not proposer_ckpt:
                    print("ERROR: Proposer training failed. Stopping.")
                    break

            # Generate data
            current_step += 1
            if skip_data:
                print_step(current_step, total_steps, f"Iteration {iteration}: Generate Data (SKIPPED - already done)")
                generated_data = str(Path(config['checkpoints_dir']).parent / 'data' / f'iter{iteration}_generated.parquet')
            else:
                print_step(current_step, total_steps, f"Iteration {iteration}: Generate Data")
                generated_data = generate_data(config, iteration, proposer_ckpt)
                if not generated_data:
                    print("ERROR: Data generation failed. Stopping.")
                    break

            # Train solver
            current_step += 1
            if skip_solver:
                print_step(current_step, total_steps, f"Iteration {iteration}: Train Solver (SKIPPED - already done)")
                ckpt_dir = Path(config['checkpoints_dir']) / f'iter{iteration}' / 'solver'
                ckpts = sorted(ckpt_dir.glob('global_step_*'))
                solver_ckpt = str(ckpts[-1]) if ckpts else None
            else:
                print_step(current_step, total_steps, f"Iteration {iteration}: Train Solver (GRPO)")
                solver_ckpt = train_solver(config, iteration, generated_data)
                if not solver_ckpt:
                    print("ERROR: Solver training failed. Stopping.")
                    break

            # Convert to HF format
            current_step += 1
            print_step(current_step, total_steps, f"Iteration {iteration}: Convert to HuggingFace")
            solver_hf_path = convert_to_hf(config, iteration, solver_ckpt)
            if not solver_hf_path:
                print("WARNING: Conversion failed, but continuing...")

            print(f"\n  Iteration {iteration} complete!")

        # =====================================================================
        # PHASE 4: CLEANUP & SUMMARY
        # =====================================================================

        print_header("TRAINING COMPLETE!")

        print("\nFinal checkpoints:")
        for i in range(1, config['num_iterations'] + 1):
            hf_path = Path(config['checkpoints_dir']) / f'iter{i}' / 'solver' / f'solver_iter{i}_hf'
            if hf_path.exists():
                print(f"  Iteration {i}: {hf_path}")

        print("\nStorage usage:")
        import shutil
        drive_path = Path(config['checkpoints_dir']).parent
        total, used, free = shutil.disk_usage(drive_path)
        print(f"  Total: {total / 1e9:.1f} GB")
        print(f"  Used: {used / 1e9:.1f} GB")
        print(f"  Free: {free / 1e9:.1f} GB")

        return True

    finally:
        # Cleanup servers
        print("\nStopping servers...")
        for pid, log in server_pids:
            stop_server(pid, log)
        print("Servers stopped.")

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dr. Zero Full Training")
    parser.add_argument("--email", required=True, help="Email for NCBI API")
    parser.add_argument("--corpus_size", type=int, default=200000, help="Number of papers to download")
    parser.add_argument("--training_seeds", type=int, default=5000, help="Number of training seeds")
    parser.add_argument("--iterations", type=int, default=3, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=50, help="Steps per iteration")
    parser.add_argument("--drive_path", default="/content/drive/MyDrive/drzero_full", help="Google Drive path")
    parser.add_argument("--repo_dir", default="/content/DrPubMedZero", help="Repository directory")

    args = parser.parse_args()

    # Build configuration
    config = DEFAULT_CONFIG.copy()
    config['email'] = args.email
    config['corpus_size'] = args.corpus_size
    config['training_seeds'] = args.training_seeds
    config['num_iterations'] = args.iterations
    config['steps_per_iteration'] = args.steps

    # Set paths
    drive_base = Path(args.drive_path)
    config['corpus_file'] = str(drive_base / 'corpus' / 'pubmed-corpus.jsonl')
    config['index_file'] = str(drive_base / 'corpus' / 'pubmedbert_index.faiss')
    config['train_parquet'] = str(drive_base / 'data' / 'training_seeds.parquet')
    config['checkpoints_dir'] = str(drive_base / 'checkpoints')
    config['repo_dir'] = args.repo_dir

    # Create directories
    for subdir in ['corpus', 'data', 'checkpoints', 'logs']:
        (drive_base / subdir).mkdir(parents=True, exist_ok=True)

    # Run training
    success = run_training(config)

    if success:
        print("\n" + "=" * 70)
        print(" TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print(" TRAINING FAILED")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
