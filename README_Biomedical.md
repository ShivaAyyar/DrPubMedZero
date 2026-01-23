# Dr. Zero: Biomedical Literature Search Adaptation

Adaptation of [Dr. Zero](https://arxiv.org/abs/2601.07055) for biomedical literature search using PubMed.

## 🎯 Overview

This project adapts Meta's Dr. Zero self-evolving search agent framework from Wikipedia to PubMed, enabling data-free training of biomedical question answering systems.

### Key Features

- **PubMed Corpus Management**: Download and preprocess biomedical literature
- **Biomedical Embeddings**: Uses PubMedBERT for domain-specific retrieval
- **Scientific Validation**: Validates genes, PMIDs, and mechanistic reasoning
- **Domain-Adapted Prompts**: Proposer and solver prompts optimized for biomedicine
- **Biomedical Rewards**: Reward functions that incentivize scientific validity
- **Evaluation Datasets**: PubMedQA, BioASQ, MedQA, COVID-QA support

## 📁 File Structure

```
drzero/                          # Original Dr. Zero repository
├── biomedical/                  # NEW: Biomedical adaptations
│   ├── __init__.py
│   ├── pubmed_corpus.py         # PubMed download & preprocessing
│   ├── biomedical_validator.py  # Scientific validation
│   ├── biomedical_retriever.py  # PubMedBERT search
│   ├── biomedical_prompts.py    # Domain-adapted prompts
│   ├── biomedical_rewards.py    # Scientific reward functions
│   └── biomedical_datasets.py   # Evaluation datasets
├── main.py                      # Orchestration script (Colab-ready)
└── README.md                    # This file
```

## 🚀 Quick Start (Google Colab)

### Option 1: Run in Colab (Recommended)

1. **Open Colab**: [Open in Colab](https://colab.research.google.com/)

2. **Clone repository**:
```python
!git clone https://github.com/facebookresearch/drzero.git
%cd drzero
```

3. **Copy biomedical adaptation files**:
```bash
# Copy the biomedical/ folder and main.py to drzero/
# (Upload files or copy from this repository)
```

4. **Run main.py sections**:
```python
# Copy and run each section from main.py
# Start with SECTION 1: Setup and Installation
```

### Option 2: Local Installation

```bash
# Clone Dr. Zero
git clone https://github.com/facebookresearch/drzero.git
cd drzero

# Copy biomedical adaptation files
cp -r /path/to/biomedical ./
cp /path/to/main.py ./

# Install dependencies
pip install torch transformers faiss-gpu datasets biopython sentence-transformers

# Run
python main.py
```

## 📖 Usage Guide

### Step 1: Download PubMed Corpus

```python
from biomedical import PubMedCorpusManager

manager = PubMedCorpusManager(
    save_path="./corpus/pubmed",
    email="your_email@example.com"  # Required by NCBI
)

# Download abstracts
articles = manager.download_pubmed_abstracts(
    query="breast cancer drug resistance",
    max_results=5000,
    date_range=("2020/01/01", "2024/12/31")
)

# Save corpus
manager.save_corpus(articles)
```

### Step 2: Build Search Index

```python
from biomedical import build_biomedical_index

retriever = build_biomedical_index(
    corpus_path="./corpus/pubmed/pubmed-corpus.jsonl",
    index_path="./corpus/pubmed/pubmedbert_index.faiss",
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

# Test search
results = retriever.search("TP53 mutations in cancer")
```

### Step 3: Generate Training Data

```python
from biomedical import BiomedicalPrompts

prompts = BiomedicalPrompts()

# Create proposer prompt
prompt = prompts.format_document_for_proposer(article, hop=2)

# Generate QA pair (using LLM)
# See main.py Section 6 for full example
```

### Step 4: Validate Questions/Answers

```python
from biomedical import BiomedicalValidator

validator = BiomedicalValidator()

# Validate question
is_valid, score, explanation = validator.validate_question(
    question="How does TP53 regulate apoptosis?",
    document=pubmed_abstract
)

# Validate answer
score, details = validator.validate_answer(
    answer="TP53 activates BAX pathway (PMID: 12345678)",
    question="How does TP53 regulate apoptosis?"
)
```

### Step 5: Calculate Rewards

```python
from biomedical import BiomedicalRewardCalculator

reward_calc = BiomedicalRewardCalculator()

# Calculate proposer reward (HRPO)
reward = reward_calc.calculate_proposer_reward(
    question=question,
    answer=answer,
    document=document,
    solver_predictions=[pred1, pred2, pred3, pred4, pred5]
)
```

### Step 6: Evaluate

```python
from biomedical import BiomedicalDatasets

datasets = BiomedicalDatasets()

# Load PubMedQA
pubmedqa = datasets.load_pubmedqa("test")

# Evaluate model
# See main.py Section 7 for full evaluation pipeline
```

## 🔧 Integration with Dr. Zero Training

To integrate with the full Dr. Zero training pipeline:

### 1. Update Corpus Configuration

Edit `scripts/download.py`:
```python
# Replace Wikipedia download with PubMed
from biomedical import PubMedCorpusManager
manager = PubMedCorpusManager(save_path=args.save_path)
articles = manager.download_pubmed_abstracts(...)
```

### 2. Update Retriever Server

Edit `search/retriever_server.py`:
```python
# Replace E5 with PubMedBERT
from biomedical import BiomedicalRetrieverServer
server = BiomedicalRetrieverServer(
    corpus_path="./corpus/pubmed-corpus.jsonl",
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
```

### 3. Update Proposer Prompts

Edit `verl/` training scripts:
```python
# Replace default prompts
from biomedical import BiomedicalPrompts
prompts = BiomedicalPrompts()
system_prompt = prompts.get_proposer_system_prompt()
```

### 4. Update Reward Calculation

Edit `verl/` reward functions:
```python
# Add biomedical validation
from biomedical import BiomedicalRewardCalculator
reward_calc = BiomedicalRewardCalculator()
reward = reward_calc.calculate_proposer_reward(...)
```

### 5. Run Training

```bash
# Train proposer (Iteration 1)
bash iter1_challenger.sh

# Generate synthetic data
bash iter1_gen_data.sh

# Train solver
bash iter1_solver.sh

# Continue for iterations 2, 3, etc.
```

## 📊 Datasets

### Included Evaluation Datasets

1. **PubMedQA**: Yes/no/maybe questions about PubMed abstracts
2. **BioASQ**: Biomedical semantic indexing and QA challenge
3. **MedQA**: Medical exam questions (USMLE, MCMLE)
4. **COVID-QA**: COVID-19 research questions

### Download Datasets

```python
from biomedical import download_sample_datasets
download_sample_datasets()  # Downloads PubMedQA and COVID-QA samples
```

## 🧪 Testing

Run tests for each component:

```bash
# Test corpus manager
python biomedical/pubmed_corpus.py

# Test retriever
python biomedical/biomedical_retriever.py

# Test validator
python biomedical/biomedical_validator.py

# Test prompts
python biomedical/biomedical_prompts.py

# Test rewards
python biomedical/biomedical_rewards.py

# Test datasets
python biomedical/biomedical_datasets.py
```

## 💾 System Requirements

### Minimum Requirements
- **CPU**: 8 cores
- **RAM**: 32 GB
- **Storage**: 50 GB
- **GPU**: Not required (but slow without it)

### Recommended for Training
- **CPU**: 16+ cores
- **RAM**: 64 GB
- **Storage**: 200 GB (for full PubMed corpus)
- **GPU**: 1x A100 (40GB) or 2x A6000 (48GB)

### Full Dr. Zero Training
- **GPU**: 8x A100 (80GB)
- **Time**: 24-72 hours

## 🐛 Troubleshooting

### Issue: NCBI API Rate Limiting

**Solution**: Get an NCBI API key
```python
import os
os.environ["NCBI_API_KEY"] = "your_key_here"
```
Get key at: https://www.ncbi.nlm.nih.gov/account/settings/

### Issue: FAISS index building fails

**Solution**: Use CPU version
```bash
pip uninstall faiss-gpu
pip install faiss-cpu
```

### Issue: Out of memory during indexing

**Solution**: Batch the encoding
```python
# Edit biomedical_retriever.py
# Reduce batch_size from 32 to 8
embeddings = self._encode_batch(texts, batch_size=8)
```

### Issue: PubMed download is slow

**Solution**: Reduce batch size or use pre-downloaded corpus
```python
# Reduce batch_size
articles = manager.download_pubmed_abstracts(
    query="...",
    max_results=5000,
    batch_size=100  # Default: 500
)
```

## 📚 References

### Papers
- [Dr. Zero: Self-Evolving Search Agents without Training Data](https://arxiv.org/abs/2601.07055)
- [PubMedBERT: Domain-specific language model for biomedical NLP](https://arxiv.org/abs/2007.15779)

### Code
- [Dr. Zero Official Repository](https://github.com/facebookresearch/drzero)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [veRL Framework](https://github.com/volcengine/verl)

### Datasets
- [PubMedQA](https://pubmedqa.github.io/)
- [BioASQ](http://bioasq.org/)
- [MedQA](https://github.com/jind11/MedQA)

## 📄 License

This project adapts Dr. Zero (Meta, non-commercial license). See [LICENSE](LICENSE.md) for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

### Areas for Contribution
- Additional biomedical datasets
- Improved entity recognition
- Cross-encoder reranking
- Clinical trial integration
- Drug-drug interaction detection

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- Meta AI Research for Dr. Zero
- Microsoft Research for PubMedBERT
- NCBI for PubMed API access
- BioASQ team for evaluation datasets

---

**Note**: This is a research prototype. For clinical use, always consult domain experts and validate results.
