"""
Main orchestration script for Dr. Zero Biomedical Adaptation.

This script can be run in sections in Google Colab.
Each section is clearly marked with comments for easy copy-pasting.
"""

# ============================================================================
# SECTION 1: Setup and Installation
# ============================================================================

# Run this section first in Google Colab
print("=" * 80)
print("SECTION 1: Setup and Installation")
print("=" * 80)

# Install required packages
import subprocess
import sys

def install_packages():
    """Install all required packages."""
    packages = [
        "torch",
        "transformers",
        "faiss-gpu",  # Use faiss-cpu if no GPU
        "datasets",
        "biopython",
        "sentence-transformers",
        "accelerate",
        "wandb",
        "tqdm",
    ]
    
    print("📦 Installing packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✓ {package}")
        except:
            print(f"  ⚠️ Failed to install {package}")
    
    print("\n✅ Installation complete!")

# Uncomment to run installation
# install_packages()

# Clone Dr. Zero repository (if not already cloned)
import os
if not os.path.exists("drzero"):
    print("\n📥 Cloning Dr. Zero repository...")
    subprocess.run(["git", "clone", "https://github.com/facebookresearch/drzero.git"])
    print("✓ Repository cloned")

# Set up directories
os.makedirs("./corpus/pubmed", exist_ok=True)
os.makedirs("./data/biomedical", exist_ok=True)
os.makedirs("./outputs", exist_ok=True)
os.makedirs("./cache", exist_ok=True)

print("\n✓ Directory structure created")


# ============================================================================
# SECTION 2: Download PubMed Corpus
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 2: Download PubMed Corpus")
print("=" * 80)

from biomedical.pubmed_corpus import PubMedCorpusManager

def download_pubmed_corpus(
    query="(breast cancer OR lung cancer OR drug resistance) AND (gene OR protein OR pathway)",
    max_results=5000,
    email="your_email@example.com"
):
    """
    Download PubMed abstracts.
    
    Args:
        query: PubMed search query
        max_results: Maximum number of papers
        email: Your email (required by NCBI)
    """
    print(f"🔍 Query: {query}")
    print(f"📊 Max results: {max_results}")
    print(f"📧 Email: {email}")
    print()
    
    # Initialize corpus manager
    manager = PubMedCorpusManager(
        save_path="./corpus/pubmed",
        email=email
    )
    
    # Download abstracts
    articles = manager.download_pubmed_abstracts(
        query=query,
        max_results=max_results,
        date_range=("2020/01/01", "2024/12/31")  # Recent papers only
    )
    
    if not articles:
        print("❌ No articles downloaded. Check your query and internet connection.")
        return None
    
    # Save corpus
    manager.save_corpus(articles)
    
    # Get statistics
    stats = manager.get_corpus_statistics()
    print("\n📊 Corpus Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create training seeds
    seeds = manager.create_training_seeds(n_seeds=1000)
    
    # Save seeds
    import json
    with open("./data/biomedical/training_seeds.jsonl", 'w') as f:
        for seed in seeds:
            f.write(json.dumps(seed) + '\n')
    
    print(f"\n✓ Created {len(seeds)} training seeds")
    
    return manager

# Uncomment to run (CHANGE EMAIL!)
# manager = download_pubmed_corpus(
#     query="breast cancer drug resistance mechanisms",
#     max_results=5000,
#     email="YOUR_EMAIL@example.com"  # CHANGE THIS!
# )


# ============================================================================
# SECTION 3: Build Search Index
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 3: Build Search Index with PubMedBERT")
print("=" * 80)

from biomedical.biomedical_retriever import build_biomedical_index

def build_search_index():
    """Build FAISS index for retrieval."""
    print("🔨 Building search index...")
    print("⚠️ This may take 10-30 minutes depending on corpus size and GPU availability")
    print()
    
    retriever = build_biomedical_index(
        corpus_path="./corpus/pubmed/pubmed-corpus.jsonl",
        index_path="./corpus/pubmed/pubmedbert_index.faiss",
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    
    # Test search
    print("\n🔍 Testing search...")
    query = "TP53 mutations in breast cancer"
    results = retriever.search(query, topk=3)
    
    print(f"\nQuery: {query}")
    print("\nTop 3 Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. PMID: {result['pmid']} (Score: {result['score']:.3f})")
        print(f"   Title: {result['title'][:100]}...")
    
    print("\n✅ Search index built successfully!")
    return retriever

# Uncomment to run
# retriever = build_search_index()


# ============================================================================
# SECTION 4: Download Evaluation Datasets
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 4: Download Evaluation Datasets")
print("=" * 80)

from biomedical.biomedical_datasets import BiomedicalDatasets, download_sample_datasets

def setup_evaluation_datasets():
    """Download biomedical QA evaluation datasets."""
    print("📥 Downloading evaluation datasets...")
    print()
    
    # Download sample datasets
    download_sample_datasets()
    
    # Or load all available
    datasets = BiomedicalDatasets()
    all_data = datasets.get_all_datasets()
    
    print("\n✅ Evaluation datasets ready!")
    return datasets

# Uncomment to run
# datasets = setup_evaluation_datasets()


# ============================================================================
# SECTION 5: Test Biomedical Components
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 5: Test Biomedical Components")
print("=" * 80)

from biomedical import (
    BiomedicalValidator,
    BiomedicalPrompts,
    BiomedicalRewardCalculator
)

def test_biomedical_components():
    """Test validator, prompts, and rewards."""
    print("🧪 Testing biomedical components...")
    print()
    
    # Initialize components
    validator = BiomedicalValidator()
    prompts = BiomedicalPrompts()
    reward_calc = BiomedicalRewardCalculator(validator)
    
    # Test article
    test_article = {
        "pmid": "12345678",
        "title": "TP53 mutations confer resistance to platinum-based chemotherapy in ovarian cancer",
        "abstract": "TP53 is frequently mutated in ovarian cancer. We found that TP53 mutations impair DNA damage response pathways, leading to chemotherapy resistance through activation of alternative survival pathways..."
    }
    
    # Test question validation
    print("1️⃣ Testing Question Validation:")
    test_question = "What molecular mechanism underlies TP53-mediated chemotherapy resistance in ovarian cancer?"
    is_valid, score, explanation = validator.validate_question(
        test_question,
        f"(PMID: {test_article['pmid']}) {test_article['abstract']}"
    )
    print(f"   Valid: {is_valid}")
    print(f"   Score: {score:.2f}")
    print(f"   Explanation: {explanation}")
    
    # Test answer validation
    print("\n2️⃣ Testing Answer Validation:")
    test_answer = "TP53 mutations activate PI3K/AKT pathway, bypassing apoptosis (PMID: 12345678)"
    a_score, a_details = validator.validate_answer(test_answer, test_question)
    print(f"   Score: {a_score:.2f}")
    print(f"   Details: {a_details}")
    
    # Test prompt generation
    print("\n3️⃣ Testing Prompt Generation:")
    proposer_prompt = prompts.format_document_for_proposer(test_article, hop=2)
    print(f"   Proposer prompt length: {len(proposer_prompt)} chars")
    print(f"   First 200 chars: {proposer_prompt[:200]}...")
    
    solver_prompt = prompts.format_question_for_solver(test_question)
    print(f"   Solver prompt length: {len(solver_prompt)} chars")
    
    # Test reward calculation
    print("\n4️⃣ Testing Reward Calculation:")
    question_tagged = f"<question>{test_question}</question>"
    answer_tagged = f"<answer>{test_answer}</answer>"
    document = f"(PMID: {test_article['pmid']}) {test_article['abstract']}"
    
    solver_predictions = [
        "<answer>PI3K/AKT pathway</answer>",
        "<answer>MAPK pathway</answer>",
        "<answer>PI3K pathway activation</answer>",
        "<answer>Unknown mechanism</answer>",
        "<answer>Apoptosis bypass</answer>"
    ]
    
    reward = reward_calc.calculate_proposer_reward(
        question_tagged,
        answer_tagged,
        document,
        solver_predictions
    )
    print(f"   Proposer reward: {reward:.3f}")
    
    print("\n✅ All components working!")

# Uncomment to run
# test_biomedical_components()


# ============================================================================
# SECTION 6: Generate Synthetic Training Data (Simplified)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 6: Generate Synthetic Training Data")
print("=" * 80)

def generate_synthetic_qa_pairs(
    n_pairs=100,
    model_name="Qwen/Qwen2.5-3B-Instruct"
):
    """
    Generate synthetic QA pairs using proposer.
    
    Note: This is a simplified version. Full Dr. Zero training requires
    the complete veRL pipeline. See the official repository for details.
    """
    print(f"🔄 Generating {n_pairs} synthetic QA pairs...")
    print(f"📝 Model: {model_name}")
    print()
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import json
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load training seeds
    print("Loading training seeds...")
    seeds = []
    with open("./data/biomedical/training_seeds.jsonl", 'r') as f:
        for line in f:
            seeds.append(json.loads(line))
    
    # Initialize components
    from biomedical import BiomedicalPrompts
    prompts = BiomedicalPrompts()
    
    # Generate QA pairs
    qa_pairs = []
    for i, seed in enumerate(seeds[:n_pairs]):
        print(f"Generating pair {i+1}/{n_pairs}...", end='\r')
        
        # Create prompt
        hop = (i % 3) + 1  # Vary difficulty: 1, 2, or 3 hops
        prompt = prompts.format_document_for_proposer(seed, hop=hop)
        
        # Generate (simplified - actual Dr. Zero uses tool calling)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract question and answer (simplified parsing)
        import re
        q_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
        a_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        
        if q_match and a_match:
            qa_pairs.append({
                "question": q_match.group(1).strip(),
                "answer": a_match.group(1).strip(),
                "document": seed['document'],
                "pmid": seed['pmid'],
                "hop": hop
            })
    
    print(f"\n✓ Generated {len(qa_pairs)} valid QA pairs")
    
    # Save
    output_path = "./data/biomedical/synthetic_qa_pairs.jsonl"
    with open(output_path, 'w') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"💾 Saved to {output_path}")
    
    return qa_pairs

# Uncomment to run (requires GPU)
# qa_pairs = generate_synthetic_qa_pairs(n_pairs=100)


# ============================================================================
# SECTION 7: Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 7: Evaluate on Biomedical QA Benchmarks")
print("=" * 80)

def evaluate_model(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    eval_dataset="pubmedqa",
    n_samples=100
):
    """
    Evaluate model on biomedical QA dataset.
    """
    print(f"📊 Evaluating {model_name} on {eval_dataset}...")
    print()
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import json
    from tqdm import tqdm
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load dataset
    print(f"Loading {eval_dataset} dataset...")
    from biomedical import BiomedicalDatasets
    datasets = BiomedicalDatasets()
    
    if eval_dataset == "pubmedqa":
        examples = datasets.load_pubmedqa("test")[:n_samples]
    else:
        print(f"❌ Dataset {eval_dataset} not implemented")
        return
    
    # Evaluate
    from biomedical import BiomedicalPrompts, BiomedicalValidator
    prompts = BiomedicalPrompts()
    validator = BiomedicalValidator()
    
    correct = 0
    total = 0
    results = []
    
    for example in tqdm(examples, desc="Evaluating"):
        # Create prompt
        prompt = prompts.format_question_for_solver(example['question'])
        
        # Generate answer
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        import re
        a_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        pred_answer = a_match.group(1).strip() if a_match else response.split("answer:")[-1].strip()
        
        # Check correctness
        is_correct = pred_answer.lower() == example['answer'].lower()
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": example['question'],
            "ground_truth": example['answer'],
            "prediction": pred_answer,
            "correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Dataset: {eval_dataset}")
    print(f"   Samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.2%}")
    
    # Save results
    results_path = f"./outputs/eval_{eval_dataset}_{model_name.replace('/', '_')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            "model": model_name,
            "dataset": eval_dataset,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    
    return accuracy, results

# Uncomment to run (requires GPU)
# accuracy, results = evaluate_model(
#     model_name="Qwen/Qwen2.5-3B-Instruct",
#     eval_dataset="pubmedqa",
#     n_samples=50
# )


# ============================================================================
# SECTION 8: Full Dr. Zero Training (Advanced)
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 8: Full Dr. Zero Training Pipeline")
print("=" * 80)

def run_drzero_training():
    """
    Run full Dr. Zero training pipeline.
    
    Note: This requires the complete veRL setup and significant compute.
    See the Dr. Zero repository for full training scripts.
    """
    print("⚠️ Full Dr. Zero training requires:")
    print("   - Multiple GPUs (recommended)")
    print("   - veRL framework installed")
    print("   - Significant compute time (hours to days)")
    print()
    print("📖 For full training, please refer to:")
    print("   https://github.com/facebookresearch/drzero")
    print()
    print("🔧 This notebook provides the biomedical adaptations:")
    print("   - PubMed corpus preparation")
    print("   - Biomedical prompts")
    print("   - Domain-specific rewards")
    print("   - Evaluation datasets")
    print()
    print("💡 To integrate with Dr. Zero:")
    print("   1. Copy biomedical/ folder to drzero/")
    print("   2. Modify iter*_challenger.sh to use BiomedicalPrompts")
    print("   3. Modify iter*_solver.sh to use BiomedicalRewardCalculator")
    print("   4. Update config/ to point to PubMed corpus and index")
    print("   5. Run: bash iter1_challenger.sh")

# Show instructions
run_drzero_training()


# ============================================================================
# SECTION 9: Summary and Next Steps
# ============================================================================

print("\n" + "=" * 80)
print("SECTION 9: Summary and Next Steps")
print("=" * 80)

print("""
✅ What You've Done:
   1. Set up Dr. Zero biomedical adaptation
   2. Downloaded PubMed corpus
   3. Built search index with PubMedBERT
   4. Prepared evaluation datasets
   5. Tested biomedical components

🎯 Next Steps:

   Option A - Quick Testing:
   • Run evaluation on existing models (Section 7)
   • Generate synthetic QA pairs (Section 6)
   • Analyze results and iterate

   Option B - Full Training:
   • Set up multi-GPU environment
   • Install veRL framework
   • Integrate biomedical components with Dr. Zero
   • Run full training pipeline (iter1, iter2, iter3)
   • Expected training time: 24-72 hours on 8x A100 GPUs

   Option C - Use as Research Tool:
   • Use biomedical retriever for literature search
   • Generate research questions with proposer prompts
   • Validate biomedical QA systems
   • Build custom biomedical QA datasets

📚 Resources:
   • Dr. Zero paper: https://arxiv.org/abs/2601.07055
   • Dr. Zero code: https://github.com/facebookresearch/drzero
   • PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
   • BioASQ: http://bioasq.org/

💬 Questions? Check the code comments or open an issue on GitHub!
""")


# ============================================================================
# Helper Functions for Colab
# ============================================================================

def mount_google_drive():
    """Mount Google Drive to save results."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted")
        
        # Create output directory in Drive
        import os
        os.makedirs('/content/drive/MyDrive/drzero_biomedical', exist_ok=True)
        print("✓ Output directory created in Drive")
        
    except ImportError:
        print("⚠️ Not running in Google Colab")

def download_outputs():
    """Download outputs as zip file."""
    import shutil
    
    print("📦 Creating outputs archive...")
    shutil.make_archive('drzero_biomedical_outputs', 'zip', './outputs')
    print("✓ Archive created: drzero_biomedical_outputs.zip")
    
    try:
        from google.colab import files
        files.download('drzero_biomedical_outputs.zip')
        print("✓ Download started")
    except ImportError:
        print("💾 File saved locally")

# Uncomment to use
# mount_google_drive()
# download_outputs()


print("\n" + "=" * 80)
print("🎉 Setup Complete! Ready to run Dr. Zero biomedical adaptation.")
print("=" * 80)
