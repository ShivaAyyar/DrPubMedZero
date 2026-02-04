# Google Colab Quick Start: Dr. Zero Biomedical Training

**Full veRL GRPO Training** - Proof-of-concept with paper's algorithm on single GPU

## What You'll Get

- Trained biomedical question generator (proposer model) using GRPO algorithm
- 10,000 PubMed papers corpus with PubMedBERT semantic search
- Full veRL training pipeline with retrieval and solver servers
- Biomedical reward function with scientific validation
- Single-iteration training (~12-16 hours on A100)
- Total storage: <15 GB on Google Drive

## Key Features

This implementation uses the **exact same training approach** from the Dr. Zero paper:
- ✅ GRPO (Group Relative Policy Optimization) algorithm
- ✅ `compute_biomedical_challenger_score_batch` reward function
- ✅ Multi-turn tool use with search retrieval
- ✅ Format validation and difficulty scoring
- ✅ Retrieval server (port 8000) for PubMed corpus
- ✅ Solver server (port 8001) via SGLang

**Only hardware parameters are adapted** (GPU count, batch size, etc.)

## Requirements

1. **Google Colab Pro/Pro+** ($10-50/month)
   - Get at: https://colab.research.google.com/signup
   - Need A100 GPU access (80GB VRAM)

2. **Google Drive** (15 GB free tier is sufficient)

3. **Your email** (for NCBI PubMed API)

## Setup Instructions

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

### Step 3: Run the Notebook

Execute cells 1-10 in order:

| Cell | What It Does | Time |
|------|--------------|------|
| **1** | Mount Google Drive, create directories | 1 min |
| **2** | Install dependencies (PyTorch, veRL, SGLang) | 10 min |
| **3** | Clone DrPubMedZero repository | 1 min |
| **4** | Configure training paths | 1 min |
| **5** | Download 10K PubMed papers | 15-20 min |
| **6** | Build PubMedBERT FAISS index | 10-15 min |
| **7** | Prepare 500 training seeds (JSONL) | 1 min |
| **7.5** | Convert seeds to parquet format | 1 min |
| **8.5** | Launch retrieval + solver servers | 3-5 min |
| **9** | **Train proposer with veRL GRPO** | **10-15 hours** |
| **9.5** | Shutdown servers and cleanup | 1 min |
| **10** | Check storage usage | 1 min |

**Total time**: ~12-16 hours

## Training Architecture

The notebook implements the full Dr. Zero pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│ Background Process 1: Retrieval Server (port 8000)         │
│ - PubMedBERT embeddings                                     │
│ - FAISS index search over 10K papers                        │
│ - Returns top-3 papers for each query                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Background Process 2: Solver Server (port 8001)            │
│ - SGLang serving Qwen-3B-Instruct                          │
│ - Used by reward function                                   │
│ - Attempts to answer proposer's questions                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Main Process: veRL GRPO Training                           │
├─────────────────────────────────────────────────────────────┤
│  PROPOSER ──generates──→ Question + Answer                 │
│      ↑                         │                            │
│      │                         ↓                            │
│      │              ┌──────────────────────┐                │
│      │              │ REWARD FUNCTION      │                │
│      │              │ 50% format + 50%     │                │
│      │              │ difficulty           │                │
│      │              └──────────┬───────────┘                │
│      │                         │                            │
│      └───── GRPO update ───────┘                            │
└─────────────────────────────────────────────────────────────┘
```

## Storage Strategy

The notebook uses a hybrid storage approach to stay under 15 GB:

**Google Drive (persistent)**:
- Corpus: ~2 GB
- FAISS index: ~0.5 GB
- Final checkpoint: ~8-10 GB
- **Total**: ~12 GB

**Colab VM temp storage** (deleted on disconnect):
- Intermediate checkpoints: ~15-20 GB
- Solver server cache: ~8 GB
- Working files: ~2 GB
- **Auto-deleted** when session ends

**Key feature**: Only the final trained model is saved to Drive. Intermediate checkpoints stay in temp storage to save space.

## What Makes This Faithful to the Paper

### Preserved (Zero Algorithm Changes)
1. ✅ **GRPO algorithm** - Exact implementation from veRL
2. ✅ **Reward function** - `compute_biomedical_challenger_score_batch` unchanged
3. ✅ **Format validation** - XML tags, tool calls, answer constraints
4. ✅ **Difficulty scoring** - Variance-based solver success measurement
5. ✅ **Multi-turn reasoning** - Tool use with search retrieval
6. ✅ **Biomedical validation** - Gene names, PMID citations, partial credit
7. ✅ **Solver rollout** - Multiple attempts (3x) to measure question difficulty
8. ✅ **Training loop** - veRL's PPO infrastructure

### Adapted (Only for Hardware)
1. ⚙️ **GPU count** - 8 → 1 (requires TP=1, DP=1)
2. ⚙️ **Batch size** - 256 → 32 (memory constraint)
3. ⚙️ **Corpus size** - 50K → 10K papers (storage constraint)
4. ⚙️ **Training duration** - 3 iterations → 1 iteration demo (time constraint)

## Training Metrics

During training (Cell 9), veRL prints real-time metrics:

```
🧬 [BIOMEDICAL] Raw format rewards: Avg 0.65, Max 0.95
🧬 [BIOMEDICAL] Final rewards: Avg 0.58, Max 0.82
🧑‍🔬 Challenger question: What is the role of TP53 mutations in...
🧬 Challenger answer: TP53 mutations confer chemoresistance...
🔬 Solver responses: ['TP53 is a tumor suppressor', ...]
```

**Expected behavior**:
- Format rewards should increase over time (0.4 → 0.7)
- Final rewards should show gradual improvement (0.3 → 0.6)
- Questions should become more structured and cite PMIDs
- Solver difficulty scores should show variance (not all 0.0 or 1.0)

## After Training

Once Cell 9 completes, you have a trained proposer that can:
- Generate biomedical research questions
- Use PubMed literature for multi-hop reasoning
- Cite valid PMIDs in responses
- Ask questions that challenge the solver

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
    "/content/drive/MyDrive/drzero_lite/final_checkpoint"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Generate a biomedical question
prompt = """Generate a challenging biomedical question about breast cancer and TP53:"""
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=400, temperature=0.8)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### "No GPU available"
- Runtime → Change runtime type → A100 GPU
- May need to wait if GPUs are busy
- Consider Colab Pro+ for priority access

### "Out of Memory during training"
- Restart runtime and re-run from Cell 1
- Reduce batch size in Cell 9: Change `data.train_batch_size=32` to `=16`
- Enable more aggressive FSDP offloading

### "Corpus download slow"
- Get NCBI API key for 10x faster downloads
- https://www.ncbi.nlm.nih.gov/account/settings/
- Add to Cell 4: `os.environ['NCBI_API_KEY'] = 'your_key'`

### "Server startup failed"
- Check Cell 8.5 logs: `!tail /tmp/retrieval_server.log`
- Verify ports are free: `!lsof -i :8000` and `!lsof -i :8001`
- Restart runtime if ports are blocked

### "Training hangs at step 0"
- Ensure both servers are running (green checkmarks in Cell 8.5)
- Check server health: `!curl http://127.0.0.1:8000/health`
- Verify parquet file was created: `!ls -lh /tmp/drzero_data/training_seeds.parquet`

### "Session disconnected during training"
- Training progress is lost (checkpoints in temp storage)
- Check W&B dashboard for partial metrics
- Re-run Cell 8.5 and Cell 9 to restart training
- Consider upgrading to Colab Pro+ for longer runtimes

### "Exceeds 15 GB on Drive"
- Delete old checkpoints: Check `/content/drive/MyDrive/drzero_lite/checkpoints/`
- Compress corpus: `!gzip /content/drive/MyDrive/drzero_lite/corpus/*.jsonl`
- Move intermediate files to temp: Only keep final checkpoint on Drive

## Cost Estimate

**Google Colab Pro**: $10/month
- Includes background execution
- V100/T4 GPU (not recommended, too slow)
- A100 access with compute units
- **Expected cost**: ~$10-15 for one training run

**Google Colab Pro+**: $50/month
- Priority A100 access
- Faster training (~12 hours vs 20+ hours)
- Background execution
- **Recommended** for this workload

## What's Different from Full Dr. Zero?

This Colab version maintains algorithm fidelity while adapting to hardware:

| Aspect | Colab PoC | Full Dr. Zero |
|--------|-----------|---------------|
| **Algorithm** | GRPO | GRPO |
| **Reward Function** | compute_biomedical_challenger_score_batch | Same |
| **Multi-turn Tool Use** | Yes (up to 5 rounds) | Yes |
| **Format Validation** | Yes | Yes |
| **Difficulty Scoring** | Yes (solver rollout) | Yes |
| | | |
| **Corpus** | 10K papers | 50K papers |
| **Training** | 50-100 steps | 600+ steps |
| **Iterations** | 1 | 3 |
| **GPU** | 1x A100 | 8x A100 |
| **Batch Size** | 32 | 256 |
| **Time** | 12-15 hours | 30-40 hours |
| **Storage** | <15 GB | ~50 GB |

**Bottom line**: This is the **real Dr. Zero training**, not a simplified approximation. Only scale is reduced.

## Next Steps

After completing training:

1. **Evaluate Quality**: Generate sample questions from test papers
2. **Compare to Base Model**: Check if questions are more structured and challenging
3. **Analyze Metrics**: Review W&B dashboard for reward curves
4. **Test on Benchmarks**: Evaluate on PubMedQA (evaluation code in repo)
5. **Scale Up**: Consider running multiple iterations if you have more time/resources

## Support

**Issues with the notebook?**
- Check [README.md](README.md) for detailed documentation
- Open GitHub issue with error details
- Include cell number, GPU type, and error logs

**Questions about Dr. Zero?**
- Original paper: https://arxiv.org/abs/2601.07055
- GitHub: https://github.com/volcengine/verl
- VeRL docs: https://verl.readthedocs.io/

---

**Ready to start?** Upload the notebook to Colab and run Cell 1!
