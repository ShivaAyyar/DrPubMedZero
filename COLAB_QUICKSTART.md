# Quick Start: Dr. Zero Biomedical Training on Google Colab

**Lightweight Version** - Optimized for single 15GB Google Drive, <20 GPU hours

## What You'll Get

- Trained biomedical question generator (proposer model)
- 10,000 PubMed papers corpus with semantic search
- Single-iteration training (~10-15 hours on A100)
- Total storage: <15 GB on Google Drive

## Requirements

1. **Google Colab Pro/Pro+** ($10-50/month)
   - Get at: https://colab.research.google.com/signup
   - Need for A100 GPU access

2. **Google Drive** (15 GB free tier is sufficient)

3. **Weights & Biases Account** (free, optional)
   - Sign up: https://wandb.ai
   - Get API key: https://wandb.ai/authorize

4. **Your email** (for NCBI PubMed API)

## Setup (3 Steps)

### Step 1: Upload Notebook

1. Go to https://colab.research.google.com/
2. Click "File" → "Upload notebook"
3. Upload `DrZero_Biomedical_Training.ipynb` from this repository

### Step 2: Set Runtime to A100

1. Click "Runtime" → "Change runtime type"
2. Select:
   - **Hardware accelerator**: GPU
   - **GPU type**: A100
3. Click "Save"

### Step 3: Run All Cells

Execute cells 1-10 in order:

| Cell | What It Does | Time |
|------|--------------|------|
| 1 | Mount Google Drive, create directories | 1 min |
| 2 | Install dependencies (PyTorch, veRL, etc.) | 10 min |
| 3 | Clone DrPubMedZero repository from GitHub | 1 min |
| 4 | Configure training (enter W&B key) | 1 min |
| 5 | Download 10K PubMed papers | 15-20 min |
| 6 | Build PubMedBERT search index | 10-15 min |
| 7 | Prepare 500 training seeds | 1 min |
| 8 | Launch retrieval & solver servers | 3-5 min |
| 9 | **Train proposer model** | **10-15 hours** |
| 10 | Check storage usage | 1 min |

**Total time**: ~12-16 hours

## Storage Strategy

The notebook uses a hybrid storage approach to stay under 15 GB:

**Google Drive (persistent)**:
- Corpus: ~2 GB
- FAISS index: ~0.5 GB
- Final checkpoint: ~8-10 GB
- **Total**: ~12 GB

**Colab VM temp storage** (deleted on disconnect):
- Intermediate checkpoints: ~15-20 GB
- Working files: ~2 GB
- **Auto-deleted** when session ends

**Key feature**: Only the final trained model is saved to Drive. Intermediate checkpoints stay in temp storage to save space.

## After Training

Once Cell 9 completes, you have a trained proposer that can:
- Generate biomedical research questions
- Use PubMed literature for multi-hop reasoning
- Cite valid PMIDs in responses

### Download Your Model

```python
!zip -r model.zip /content/drive/MyDrive/drzero_lite/final_checkpoint
from google.colab import files
files.download('model.zip')
```

### Test Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/content/drive/MyDrive/drzero_lite/final_checkpoint/step_100"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Generate a biomedical question
prompt = "Generate a research question about breast cancer and TP53:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### "No GPU available"
- Runtime → Change runtime type → A100 GPU
- May need to wait if GPUs are busy

### "Out of Memory"
- Restart runtime and re-run from Cell 1
- Reduce batch_size in Cell 4: `CONFIG['batch_size'] = 16`

### "Corpus download slow"
- Get NCBI API key for 10x faster downloads
- https://www.ncbi.nlm.nih.gov/account/settings/
- Add to Cell 4: `os.environ['NCBI_API_KEY'] = 'your_key'`

### "Session disconnected during training"
- Training checkpoints are lost (temp storage)
- But partial progress may be saved - check W&B dashboard
- Re-run Cell 9 to restart training

### "Exceeds 15 GB on Drive"
- Delete old checkpoints: Check `/content/drive/MyDrive/drzero_lite`
- Compress corpus: `!gzip /content/drive/MyDrive/drzero_lite/corpus/*.jsonl`

## Cost Estimate

**Google Colab Pro**: $10/month
- V100 GPU: ~20 hours needed
- **Total cost**: $10

**Google Colab Pro+**: $50/month
- A100 GPU: ~12 hours needed
- Compute units deducted
- **Recommended** for faster training

## Scaling Up

Want full 3-iteration training like the original Dr. Zero paper?

**Option 1: Extend this notebook**
- Increase `max_steps` in Cell 4 (100 → 500)
- Add more training seeds (500 → 2000)
- Storage needed: ~20-25 GB

**Option 2: Get more storage**
- Upgrade Google Drive (100 GB for $2/month)
- Download full 50K paper corpus
- Train for multiple iterations

**Option 3: Use cloud compute**
- Export to AWS/GCP with more storage
- Follow README_Biomedical.md for full setup

## What's Different from Full Version?

This lightweight version trades scale for convenience:

| Aspect | Lightweight | Full Version |
|--------|-------------|--------------|
| Corpus | 10K papers | 50K papers |
| Training | 100 steps | 600+ steps |
| Iterations | 1 | 3 |
| GPU time | 12-15 hours | 30-40 hours |
| Storage | <15 GB | ~50 GB |
| Performance | Good | Better |

The lightweight version is perfect for:
- Testing the pipeline
- Learning how Dr. Zero works
- Generating initial biomedical QA pairs
- Academic projects with limited resources

## Next Steps

After completing training:

1. **Evaluate quality**: Generate sample questions, check if they make sense
2. **Try on benchmarks**: Test on PubMedQA (evaluation code in `biomedical/`)
3. **Fine-tune further**: Add domain-specific data for your research area
4. **Share results**: Publish model on Hugging Face Hub

## Support

**Issues with the notebook?**
- Check [README_Biomedical.md](README_Biomedical.md) for detailed documentation
- Open GitHub issue with error details
- Include cell number and GPU type

**Questions about Dr. Zero?**
- Original paper: https://arxiv.org/abs/2601.07055
- GitHub: https://github.com/facebookresearch/drzero

---

**Ready to start?** Upload the notebook to Colab and run Cell 1!
