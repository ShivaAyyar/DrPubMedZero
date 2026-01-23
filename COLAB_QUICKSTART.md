# Google Colab Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**

### Step 2: Clone and Setup

```python
# Cell 1: Clone repository and upload files
!git clone https://github.com/facebookresearch/drzero.git
%cd drzero

# Upload the biomedical/ folder and main.py from this project
# Method 1: Use Colab's file upload
# Method 2: Upload to Google Drive and copy
```

### Step 3: Install Dependencies

```python
# Cell 2: Install dependencies
!pip install -q torch transformers faiss-gpu datasets biopython sentence-transformers

# Verify installation
import torch
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
```

### Step 4: Run Main Script Sections

```python
# Cell 3: Import and setup
import sys
sys.path.insert(0, '/content/drzero')

# Copy sections from main.py and run them one by one
# Start with SECTION 1: Setup and Installation
```

## 📋 Section-by-Section Guide

### SECTION 1: Setup and Installation ⚙️

**Purpose**: Install packages and create directories

```python
# Copy and run from main.py lines 1-50
import subprocess
import sys

def install_packages():
    packages = ["torch", "transformers", "faiss-gpu", "datasets", "biopython"]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_packages()

import os
os.makedirs("./corpus/pubmed", exist_ok=True)
os.makedirs("./data/biomedical", exist_ok=True)
os.makedirs("./outputs", exist_ok=True)
```

**Expected Output**: ✓ Packages installed, ✓ Directories created

---

### SECTION 2: Download PubMed Corpus 📥

**Purpose**: Download biomedical abstracts from PubMed

```python
# Copy and run from main.py lines 52-120
from biomedical.pubmed_corpus import PubMedCorpusManager

manager = PubMedCorpusManager(
    save_path="./corpus/pubmed",
    email="YOUR_EMAIL@example.com"  # ⚠️ CHANGE THIS!
)

articles = manager.download_pubmed_abstracts(
    query="breast cancer drug resistance",
    max_results=1000,  # Start small (1000), increase later (5000-10000)
    date_range=("2020/01/01", "2024/12/31")
)

manager.save_corpus(articles)
```

**Expected Time**: 5-15 minutes for 1000 papers  
**Expected Output**: ✓ Downloaded 1000 articles

**⚠️ Important**: You MUST provide a valid email for NCBI API access

---

### SECTION 3: Build Search Index 🔨

**Purpose**: Create FAISS index with PubMedBERT embeddings

```python
# Copy and run from main.py lines 122-180
from biomedical.biomedical_retriever import build_biomedical_index

retriever = build_biomedical_index(
    corpus_path="./corpus/pubmed/pubmed-corpus.jsonl",
    index_path="./corpus/pubmed/pubmedbert_index.faiss",
    model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

# Test search
results = retriever.search("TP53 mutations in breast cancer")
print(retriever.format_search_results(results))
```

**Expected Time**: 10-30 minutes depending on corpus size  
**Expected Output**: ✓ Index built, search results displayed

**💡 Tip**: This downloads ~400MB PubMedBERT model on first run

---

### SECTION 4: Download Evaluation Datasets 📊

**Purpose**: Get biomedical QA benchmarks

```python
# Copy and run from main.py lines 182-200
from biomedical.biomedical_datasets import download_sample_datasets

download_sample_datasets()
```

**Expected Time**: 2-5 minutes  
**Expected Output**: ✓ PubMedQA and COVID-QA downloaded

---

### SECTION 5: Test Components 🧪

**Purpose**: Verify everything works

```python
# Copy and run from main.py lines 202-280
from biomedical import (
    BiomedicalValidator,
    BiomedicalPrompts,
    BiomedicalRewardCalculator
)

# Test validation
validator = BiomedicalValidator()
question = "How does TP53 regulate apoptosis?"
document = "(PMID: 12345678) TP53 is a tumor suppressor..."

is_valid, score, explanation = validator.validate_question(question, document)
print(f"Valid: {is_valid}, Score: {score:.2f}")

# Test prompts
prompts = BiomedicalPrompts()
test_article = {
    "pmid": "12345678",
    "title": "TP53 in cancer",
    "abstract": "TP53 mutations are common..."
}

proposer_prompt = prompts.format_document_for_proposer(test_article, hop=2)
print(f"Prompt length: {len(proposer_prompt)}")
```

**Expected Output**: ✓ All components working

---

### SECTION 6: Generate Synthetic QA Pairs 🤖

**Purpose**: Create training data using proposer

```python
# Copy and run from main.py lines 282-350
# ⚠️ This requires GPU and takes longer

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate one QA pair as test
# See main.py for full batch generation
```

**Expected Time**: 30-60 minutes for 100 pairs  
**Required**: GPU (T4, A100, etc.)

---

### SECTION 7: Evaluate Model 📈

**Purpose**: Test on biomedical QA benchmarks

```python
# Copy and run from main.py lines 352-450
from biomedical import BiomedicalDatasets

datasets = BiomedicalDatasets()
pubmedqa = datasets.load_pubmedqa("test")

# Run evaluation
# See main.py for complete evaluation pipeline
```

**Expected Time**: 20-40 minutes for 50 samples  
**Expected Output**: Accuracy metrics

---

## 💾 Saving Results to Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r ./outputs /content/drive/MyDrive/drzero_biomedical_outputs

# Or download as zip
!zip -r outputs.zip ./outputs
from google.colab import files
files.download('outputs.zip')
```

---

## 🐛 Common Issues and Solutions

### Issue 1: "Email is required by NCBI"

**Solution**:
```python
manager = PubMedCorpusManager(
    save_path="./corpus/pubmed",
    email="your.actual.email@domain.com"  # Use real email!
)
```

### Issue 2: Out of Memory during indexing

**Solution**: Reduce batch size
```python
# In biomedical_retriever.py, line ~150
embeddings = self._encode_batch(texts, batch_size=8)  # Default: 32
```

### Issue 3: Slow PubMed downloads

**Solution 1**: Get NCBI API key (increases rate limit)
```python
import os
os.environ["NCBI_API_KEY"] = "your_key_here"
# Get at: https://www.ncbi.nlm.nih.gov/account/settings/
```

**Solution 2**: Download fewer papers
```python
articles = manager.download_pubmed_abstracts(
    max_results=500  # Start smaller
)
```

### Issue 4: CUDA out of memory

**Solution**: Use smaller model or CPU
```python
# Option 1: Use CPU
retriever = BiomedicalRetrieverServer(
    corpus_path="...",
    device="cpu"
)

# Option 2: Smaller batch size
model.generate(..., batch_size=1)
```

---

## ⏱️ Time Estimates

### Quick Test (Sections 1-5)
- **Time**: 30-45 minutes
- **Resources**: CPU okay, GPU better
- **Storage**: ~5 GB

### Full Pipeline (Sections 1-7)
- **Time**: 2-4 hours
- **Resources**: GPU required (T4 minimum)
- **Storage**: ~20 GB

### Full Dr. Zero Training
- **Time**: 24-72 hours
- **Resources**: 8x A100 GPUs
- **Storage**: ~200 GB

---

## 📊 Expected Results

### PubMedQA Accuracy
- **Baseline (Qwen 3B-Instruct)**: ~40-50%
- **After Dr. Zero training**: ~60-70%
- **Best supervised models**: ~70-80%

### Generation Quality
- **Question diversity**: 1000+ unique questions
- **Hop distribution**: 40% 1-hop, 30% 2-hop, 20% 3-hop, 10% 4-hop
- **Valid PMIDs**: 80-90% citation accuracy

---

## 🎯 Next Steps After Quick Start

1. **Increase corpus size**: Download 5000-10000 papers
2. **Try different queries**: Cancer types, drug classes, pathways
3. **Fine-tune models**: Use generated QA pairs for supervised training
4. **Custom evaluation**: Create domain-specific test sets
5. **Full Dr. Zero training**: Set up multi-GPU environment

---

## 📚 Additional Resources

- **PubMed Search Help**: https://pubmed.ncbi.nlm.nih.gov/help/
- **PubMedBERT Paper**: https://arxiv.org/abs/2007.15779
- **Dr. Zero Paper**: https://arxiv.org/abs/2601.07055
- **BioASQ Challenge**: http://bioasq.org/

---

## 💬 Getting Help

1. Check error messages carefully
2. Review README.md
3. Check GitHub issues
4. Search for similar errors in Dr. Zero/Search-R1 repos

---

**🎉 You're ready to go! Start with Section 1 and work your way through.**
