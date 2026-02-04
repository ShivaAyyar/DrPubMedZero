# Dr. Zero: Self-Evolving Search Agents without Training Data

This repository contains the code for [**Dr. Zero: Self-Evolving Search Agents without Training Data**](https://arxiv.org/abs/2601.07055), with adaptations for biomedical literature search using PubMed.

## 🚀 Overview

Dr. Zero is a framework enabling search agents to effectively self-evolve without any training data. The framework uses a self-evolution feedback loop where a proposer generates diverse questions to train a solver initialized from the same base model. As the solver evolves, it incentivizes the proposer to produce increasingly difficult yet solvable tasks, establishing an automated curriculum to refine both agents.

### Core Components

*   **Proposer:** A question generation agent that creates hard yet solvable questions, driving solver improvement.
*   **Solver:** The primary search agent trained with synthetic data from the proposer to answer challenging questions using the search tool.
*   **Zero-Data Initialization:** The process starts with zero training data and relies solely on an external search engine.

<img src=verl/intro.png width=1000>

## 🧬 Biomedical Adaptation

This repository includes adaptations for biomedical literature search using PubMed abstracts instead of Wikipedia. Key modifications:

- **Corpus**: PubMed abstracts (configurable size, e.g., 10K-50K papers)
- **Embeddings**: PubMedBERT for domain-specific semantic search
- **Validation**: Biomedical entity validation (genes, PMIDs, mechanisms)
- **Reward Function**: `compute_biomedical_challenger_score_batch` with scientific validity scoring
- **Training**: Full GRPO algorithm with multi-turn tool use

### Quick Start (Google Colab)

For a proof-of-concept on a single GPU, see [COLAB_QUICKSTART.md](COLAB_QUICKSTART.md) for a complete walkthrough using the `DrZero_Biomedical_Training.ipynb` notebook.

**Key Features of Colab Version:**
- Single A100 GPU (80GB)
- 10K PubMed papers corpus
- Full veRL GRPO training pipeline
- Retrieval + solver server architecture
- ~12-16 hours total runtime
- <15 GB Google Drive storage required

## 🛠️ Setup & Installation

### 1. Environment

Install required dependencies:

```bash
pip install torch transformers faiss-gpu datasets
pip install verl==0.5.0
pip install sglang[all]
pip install pandas pyarrow biopython
```

Additional dependencies: [verl requirements](https://github.com/volcengine/verl/blob/v0.5.0/requirements.txt) and [sglang requirements](https://github.com/volcengine/verl/blob/v0.5.0/requirements_sglang.txt)

### 2. Search Engine Setup

The framework requires a local retrieval server with a corpus and search index.

#### For Wikipedia (Original Dr. Zero)

```bash
save_path=./corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

#### For PubMed (Biomedical Version)

The biomedical notebook handles this automatically, or manually:

```python
from biomedical import PubMedCorpusManager

manager = PubMedCorpusManager(
    save_path="./corpus/pubmed",
    email="your_email@example.com"
)

# Download abstracts
articles = manager.download_pubmed_abstracts(
    query="cancer OR diabetes OR alzheimers",
    max_results=10000,
    date_range=("2020/01/01", "2024/12/31")
)

# Save corpus
manager.save_corpus(articles)

# Build index with PubMedBERT
from biomedical import build_biomedical_index
retriever = build_biomedical_index(
    corpus_path="./corpus/pubmed/pubmed-corpus.jsonl",
    index_path="./corpus/pubmed/pubmedbert_index.faiss",
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
```

### 3. Launch Servers

The training requires two background servers:

**Retrieval Server** (port 8000):
```bash
python search/retrieval_server.py \
    --mode=biomedical \
    --index_path=./corpus/pubmed/pubmedbert_index.faiss \
    --corpus_path=./corpus/pubmed/pubmed-corpus.jsonl \
    --retriever_model=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --faiss_gpu \
    --topk=3
```

**Solver Server** (port 8001):
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --port 8001 \
    --mem-fraction-static 0.35 \
    --tp 1
```

## 🏃 Training Workflow

### Original Dr. Zero (Wikipedia)

The training process proceeds in iterations (Iter 1, Iter 2, Iter 3...).

#### Phase 0: Initial Data Preparation

```bash
python process_train.py --local_dir ./data
python process_test.py --local_dir ./data
```

#### Iteration 1

**1. Train Proposer:**
```bash
bash iter1_challenger.sh
```

**2. Synthesize Data:**
```bash
bash iter1_gen_data.sh
```

**3. Train Solver:**
```bash
bash iter1_solver.sh
```

**4. Convert Solver to HF Format:**
```bash
bash convert.sh
```

#### Subsequent Iterations

Repeat with `iter2_challenger.sh`, `iter2_gen_data.sh`, `iter2_solver.sh`, etc.

### Biomedical Version (PubMed)

For the biomedical adaptation, use the Jupyter notebook `DrZero_Biomedical_Training.ipynb` which orchestrates:

1. **Corpus Preparation**: Download and index PubMed abstracts
2. **Seed Generation**: Convert papers to training seeds in parquet format
3. **Server Launch**: Start retrieval and solver servers
4. **Proposer Training**: Full veRL GRPO training with biomedical reward function
5. **Evaluation**: Generate and validate biomedical QA pairs

The notebook uses the same GRPO algorithm and reward function as the paper, adapted only for single-GPU constraints.

## 📊 Biomedical Reward Function

The biomedical adaptation uses `compute_biomedical_challenger_score_batch` from [verl/custom_reward/reward_function.py](verl/custom_reward/reward_function.py):

**Reward Components (50/50 split):**

1. **Format Validation (50%)**:
   - XML structure (`<think>`, `<question>`, `<answer>`)
   - Tool call presence and validity
   - Answer length constraints
   - PMID citation format

2. **Difficulty Scoring (50%)**:
   - Solver rollout (3 attempts per question)
   - Variance-based difficulty measurement
   - Partial credit for biomedical entities
   - Gene name and mechanism validation

**Key Features:**
- Rewards questions where solver predictions vary (not too easy/hard)
- Validates scientific correctness (genes, PMIDs, mechanisms)
- Encourages multi-turn tool use for reasoning
- Penalizes format violations and hallucinations

## 💾 System Requirements

### Minimum (Colab PoC)
- **GPU**: 1x A100 (80GB)
- **RAM**: 80 GB
- **Storage**: 15 GB
- **Time**: 12-16 hours

### Recommended (Full Training)
- **GPU**: 8x A100 (80GB)
- **RAM**: 512 GB
- **Storage**: 200 GB
- **Time**: 24-72 hours

## 📚 Citation

If you find Dr. Zero useful, please cite:

```bibtex
@article{yue2026dr,
  title={Dr. Zero: Self-Evolving Search Agents without Training Data},
  author={Yue, Zhenrui and Upasani, Kartikeya and Yang, Xianjun and Ge, Suyu and Nie, Shaoliang and Mao, Yuning and Liu, Zhe and Wang, Dong},
  journal={arXiv preprint arXiv:2601.07055},
  year={2026}
}
```

## 📄 License

The code is released under a non-commercial license. See [LICENSE](LICENSE.md) for details.

## 🙏 Acknowledgements

This implementation builds upon:
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - Multi-turn search agent framework
- [VeRL](https://github.com/volcengine/verl) - Distributed RL training infrastructure
- [PubMedBERT](https://github.com/ncbi-nlp/bluebert) - Biomedical language model
- [SGLang](https://github.com/sgl-project/sglang) - Efficient LLM serving

## 🔗 Related Resources

- [Original Dr. Zero Paper](https://arxiv.org/abs/2601.07055)
- [PubMedBERT Paper](https://arxiv.org/abs/2007.15779)
- [GRPO Algorithm](https://github.com/volcengine/verl)
- [Colab Quickstart Guide](COLAB_QUICKSTART.md)
