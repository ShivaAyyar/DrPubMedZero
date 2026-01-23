# Quick Start: Dr. Zero Biomedical Training on Google Colab

## 🎯 Overview

Train Dr. Zero on PubMed biomedical literature using Google Colab - **no file uploads needed!** Everything clones directly from your GitHub repository.

## ✅ What You Need

1. **Google Colab Pro/Pro+** ($10-50/month)
   - Get at: https://colab.research.google.com/signup
   - Need for A100 GPU access

2. **Weights & Biases Account** (free)
   - Sign up: https://wandb.ai
   - Get API key: https://wandb.ai/authorize

3. **~50 GB Google Drive Space**
   - **Option A**: Single account with 50+ GB
   - **Option B**: Use Dual-Drive Setup (see below) ⭐ **RECOMMENDED**

4. **Your email** (for NCBI PubMed API)

### 💡 Don't Have 50 GB? Use Dual-Drive Setup!

If your Colab Pro account only has 15 GB storage:

**✨ Solution**: Mount two Google Drives in one session!

```
Your Setup:
- Account A (Colab Pro): A100 GPU + 15 GB storage
- Account B (Any account): 80 GB storage

The notebook will:
- Use Account A for compute (A100 GPU)
- Use Account B for storage (large files)
- Mount both drives automatically in Cell 1
```

**No extra costs, no file transfers needed!**

📖 See [DUAL_DRIVE_SETUP.md](DUAL_DRIVE_SETUP.md) for full instructions.

## 🚀 Super Simple Setup (3 Steps!)

### Step 1: Upload Notebook to Colab

1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload `DrZero_Biomedical_Training.ipynb` (from this repository)

### Step 2: Set Runtime to A100

1. Click "Runtime" → "Change runtime type"
2. Select:
   - **Hardware accelerator**: GPU
   - **GPU type**: A100
3. Click "Save"

### Step 3: Run All Cells

Just execute cells in order. That's it!

**Cell 1**: Mount Google Drive
**Cell 2**: Install dependencies (~10 min)
**Cell 3**: **Clone your GitHub repo** (no uploads needed!)
**Cell 4**: Enter W&B key + configuration
**Cell 5**: Download PubMed corpus (~1-2 hours)
**Cells 6+**: Training pipeline

## 💡 Key Advantage: GitHub Integration

**Cell 3 automatically:**
- Clones `https://github.com/ShivaAyyar/DrPubMedZero`
- Loads all biomedical modules
- Verifies all files present
- No manual file uploads!

**This means:**
- ✅ Always using latest code from your repo
- ✅ Easy to update (just `git pull`)
- ✅ No file management hassle
- ✅ Reproducible setup

## 📊 What Happens During Training

### Timeline

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Setup | 15-30 min | Install dependencies, clone repo |
| Corpus | 1-2 hours | Download 50K PubMed papers |
| Index | 30-60 min | Build PubMedBERT search index |
| Iter 1 | 8-10 hours | Train base proposer + solver |
| Iter 2 | 8-10 hours | Train with improved solver |
| Iter 3 | 8-10 hours | Final training iteration |
| Eval | 2-3 hours | Test on biomedical benchmarks |
| **Total** | **30-40 hours** | Complete pipeline |

### Storage (Google Drive)

- Corpus: ~10 GB
- Checkpoints: ~30 GB
- Data: ~5 GB
- Logs: ~2 GB
- **Total: ~50 GB**

## 🔄 Auto-Recovery Features

The notebook includes:

**✅ Automatic Checkpointing**
- Saves to Google Drive every 25 steps
- Can resume from any point

**✅ Disconnect Recovery**
- Just re-run the training cell
- Auto-detects last checkpoint
- Continues where it left off

**✅ Progress Monitoring**
- W&B dashboard shows real-time metrics
- Loss curves, rewards, GPU usage
- Sample generated questions

## 📈 Expected Results

### After Full Training

**Model Performance:**
- PubMedQA Accuracy: 60-70% (vs 45% baseline)
- BioASQ: Competitive with supervised methods
- Citation Quality: 80-90% valid PMIDs

**Generated Questions:**
- Mechanistic reasoning (how/why)
- Multi-hop (2-4 reasoning steps)
- Proper gene nomenclature
- Valid PMID citations

**Example Output:**
```
Q: Through what molecular mechanism do TP53 mutations confer
   platinum resistance in ovarian cancer?

A: TP53 mutations lead to loss of BAX pathway activation, preventing
   mitochondrial outer membrane permeabilization following platinum-
   induced DNA damage. This allows cells to evade apoptosis via
   alternative survival pathways such as PI3K/AKT signaling.
   (PMID: 12345678, PMID: 87654321)
```

## 🐛 Quick Troubleshooting

### "No GPU available"
→ Runtime → Change runtime type → A100 GPU

### "Out of Memory"
→ In Cell 4, reduce batch size:
```python
CONFIG['batch_size'] = 32  # from 64
```

### "Corpus download slow"
→ Get NCBI API key (10x faster):
```python
import os
os.environ['NCBI_API_KEY'] = 'your_key'
# Get at: https://www.ncbi.nlm.nih.gov/account/settings/
```

### "Session disconnected"
→ Just re-run the training cell
→ Auto-resumes from last checkpoint in Drive

### "Import error"
→ Re-run Cell 3 (installs missing dependencies)

## 📚 Full Documentation

For detailed information, see:

- **[COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)** - Complete guide with all details
- **[README_Biomedical.md](README_Biomedical.md)** - Biomedical adaptation overview
- **[COLAB_QUICKSTART.md](COLAB_QUICKSTART.md)** - Original quickstart guide

## 💰 Cost Estimate

**Google Colab Pro+:**
- $50/month subscription
- ~30 hours GPU time
- **Total cost: ~$50** for full training

**Alternative: Colab Pro**
- $10/month
- V100 GPU (slower)
- ~60 hours needed
- **Total cost: ~$10-20**

## 🎓 What You'll Learn

This pipeline teaches you:
- Self-supervised learning
- Reinforcement learning with GRPO
- Biomedical NLP
- Tool-augmented language models
- Multi-hop reasoning
- Scientific literature mining

## 🤝 Next Steps

After training completes:

1. **Download Models**
   ```python
   !zip -r final_models.zip ./checkpoints/iter3_*
   from google.colab import files
   files.download('final_models.zip')
   ```

2. **Deploy for Inference**
   - Use for biomedical Q&A
   - Literature search
   - Hypothesis generation

3. **Fine-tune Further**
   - Add domain-specific data
   - Customize for your research area

4. **Share Results**
   - Publish on Hugging Face Hub
   - Contribute to biomedical AI

## 📞 Support

**Issues?**
- Check [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) troubleshooting section
- Open GitHub issue with error details
- Include cell number and GPU type

**Questions about the paper?**
- Dr. Zero: https://arxiv.org/abs/2601.07055
- GitHub: https://github.com/facebookresearch/drzero

---

**Ready to start?** Upload `DrZero_Biomedical_Training.ipynb` to Colab and run Cell 1! 🚀
