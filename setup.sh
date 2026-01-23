#!/bin/bash

# Dr. Zero Biomedical Adaptation - Quick Setup Script
# This script automates the initial setup process

set -e  # Exit on error

echo "========================================="
echo "Dr. Zero Biomedical Adaptation Setup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if running in Google Colab
if [ -d "/content" ] && [ -d "/content/sample_data" ]; then
    echo "Running in Google Colab environment"
    COLAB=true
else
    echo "Running in local environment"
    COLAB=false
fi

# Step 1: Check Python version
echo ""
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    print_success "Python $PYTHON_VERSION detected"
else
    print_error "Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

# Step 2: Clone Dr. Zero repository (if not exists)
echo ""
echo "Step 2: Checking Dr. Zero repository..."
if [ ! -d "drzero" ]; then
    echo "Cloning Dr. Zero repository..."
    git clone https://github.com/facebookresearch/drzero.git
    print_success "Repository cloned"
else
    print_success "Repository already exists"
fi

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
echo "This may take 5-10 minutes..."

# Determine if GPU is available
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    print_success "GPU detected, installing CUDA packages"
    FAISS_PACKAGE="faiss-gpu"
else
    print_warning "No GPU detected, installing CPU packages"
    FAISS_PACKAGE="faiss-cpu"
fi

# Install packages
pip install -q torch transformers $FAISS_PACKAGE datasets biopython sentence-transformers accelerate tqdm 2>&1 | grep -v "already satisfied" || true
print_success "Dependencies installed"

# Step 4: Create directory structure
echo ""
echo "Step 4: Creating directory structure..."
mkdir -p corpus/pubmed
mkdir -p data/biomedical
mkdir -p outputs
mkdir -p cache
print_success "Directories created"

# Step 5: Copy biomedical files
echo ""
echo "Step 5: Setting up biomedical adaptation..."
if [ -d "biomedical" ]; then
    cp -r biomedical drzero/
    print_success "Biomedical modules copied to drzero/"
else
    print_warning "biomedical/ folder not found. Please copy manually."
fi

if [ -f "main.py" ]; then
    cp main.py drzero/
    print_success "main.py copied to drzero/"
else
    print_warning "main.py not found. Please copy manually."
fi

# Step 6: Test imports
echo ""
echo "Step 6: Testing imports..."
python3 << EOF
try:
    import torch
    print("✓ PyTorch imported")
    import transformers
    print("✓ Transformers imported")
    import faiss
    print("✓ FAISS imported")
    import datasets
    print("✓ Datasets imported")
    from Bio import Entrez
    print("✓ BioPython imported")
    print("\n✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    print_success "All imports successful"
else
    print_error "Import test failed"
    exit 1
fi

# Step 7: Configuration
echo ""
echo "Step 7: Configuration..."
echo ""
echo "Please provide the following information:"
echo ""

# Get email for NCBI
read -p "Enter your email for NCBI Entrez (required): " NCBI_EMAIL

# Get NCBI API key (optional)
read -p "Enter your NCBI API key (optional, press Enter to skip): " NCBI_API_KEY

# Create config file
cat > config.json << EOF
{
  "ncbi_email": "$NCBI_EMAIL",
  "ncbi_api_key": "$NCBI_API_KEY",
  "corpus_path": "./corpus/pubmed/pubmed-corpus.jsonl",
  "index_path": "./corpus/pubmed/pubmedbert_index.faiss",
  "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
  "cache_dir": "./cache",
  "output_dir": "./outputs"
}
EOF

print_success "Configuration saved to config.json"

# Step 8: Quick test
echo ""
echo "Step 8: Running quick test..."
python3 << EOF
import sys
sys.path.insert(0, 'drzero')

try:
    from biomedical import BiomedicalValidator
    validator = BiomedicalValidator()
    
    # Test validation
    question = "What is the role of TP53 in cancer?"
    document = "(PMID: 12345678) TP53 is a tumor suppressor gene..."
    is_valid, score, explanation = validator.validate_question(question, document)
    
    print(f"✓ Validator test passed (score: {score:.2f})")
    print("\n✅ Setup successful!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "Quick test failed"
    exit 1
fi

# Step 9: Summary
echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "What's next?"
echo ""
echo "1. Download PubMed corpus:"
echo "   python3 drzero/main.py  # Run Section 2"
echo ""
echo "2. Build search index:"
echo "   python3 drzero/main.py  # Run Section 3"
echo ""
echo "3. Test components:"
echo "   python3 drzero/main.py  # Run Section 5"
echo ""
echo "4. For full training, see:"
echo "   https://github.com/facebookresearch/drzero"
echo ""
echo "Configuration saved in: config.json"
echo "Corpus will be saved to: corpus/pubmed/"
echo "Outputs will be saved to: outputs/"
echo ""

if [ "$COLAB" = true ]; then
    echo "Google Colab Tips:"
    echo "• Mount Drive to save results: mount_google_drive()"
    echo "• Use GPU runtime for faster processing"
    echo "• Results will persist across sessions if saved to Drive"
    echo ""
fi

echo "For help, see: README.md"
echo ""
print_success "Setup complete! Happy researching! 🎉"
