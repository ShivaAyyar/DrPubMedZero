# Dual Google Drive Setup for Colab Training

Complete guide for mounting two Google Drive accounts in a single Colab session to overcome storage limitations.

## 🎯 Problem & Solution

### The Challenge

You have two Google accounts:
- **Account A (Colab Pro)**: A100 GPU access, but only **15 GB** Drive storage
- **Account B**: **80 GB** Drive storage, but no Colab Pro

Training Dr. Zero requires ~50 GB:
- PubMed corpus: 10 GB
- Model checkpoints: 30 GB
- Generated data: 5 GB
- Outputs: 5 GB

###Solution: Mount Both Drives! ✨

Google Colab supports mounting **multiple Google Drive accounts simultaneously** in one notebook session.

## 📋 How It Works

```
┌─────────────────────────────────────────────────────────┐
│ Google Colab Session (Account A - Colab Pro)            │
│ GPU: A100                                               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  /content/drive_pro/          /content/drive_storage/   │
│  (Account A - 15GB)           (Account B - 80GB)        │
│  ├── logs/                    ├── corpus/               │
│  ├── configs/                 ├── checkpoints/          │
│  └── notebooks/               ├── data/                 │
│      (<1 GB used)             └── outputs/              │
│                                   (~47 GB used)         │
└─────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- ✅ Use Colab Pro's A100 GPU
- ✅ Store large files on 80GB account
- ✅ No file transfers needed
- ✅ Both drives accessible simultaneously
- ✅ No additional costs

## 🚀 Step-by-Step Setup

### Step 1: Prepare Your Browser

Before running the notebook, make sure:

1. Sign into **both Google accounts** in your browser
   - Open Chrome/Firefox
   - Go to accounts.google.com
   - Sign into Account A (Colab Pro)
   - Click your profile → Add account → Sign into Account B

2. Verify you can switch between accounts:
   - Click your profile picture
   - You should see both accounts listed

### Step 2: Open Notebook in Colab Pro Account

1. Go to https://colab.research.google.com/
2. **Make sure you're signed in as Account A** (Colab Pro)
3. Upload `DrZero_Biomedical_Training.ipynb`
4. Set runtime: Runtime → Change runtime type → A100 GPU

### Step 3: Run Cell 1 (Mount Account A)

Run Cell 1 in the notebook. It will display:

```
====================================================
STEP 1: Mount Account A (Colab Pro Account)
====================================================

📌 Click the link below and authenticate with your COLAB PRO account
```

**What to do:**
1. Click the authentication link that appears
2. Select **Account A** (your Colab Pro account)
3. Click "Allow" to grant permissions
4. Wait for confirmation: "✅ Account A (Colab Pro) mounted successfully!"

### Step 4: Run Cell 1 (Mount Account B)

Cell 1 continues with:

```
====================================================
STEP 2: Mount Account B (80GB Storage Account)
====================================================

📌 Click the link below and:
   1. Click 'Use another account'
   2. Sign in with your 80GB STORAGE account
   3. Authorize access
```

**What to do:**
1. Click the authentication link
2. **IMPORTANT**: Click "Use another account" (don't select Account A!)
3. Enter Account B credentials (your 80GB storage account)
4. Click "Allow" to grant permissions
5. Wait for confirmation: "✅ Account B (Storage) mounted successfully!"

### Step 5: Verify Both Mounts

Cell 1 automatically verifies both drives:

```
====================================================
STEP 3: Verify Both Drives
====================================================

✅ Pro Drive (Account A): 15 items
   Path: /content/drive_pro/MyDrive/
   Sample: ['folder1', 'file1.txt', 'folder2']

✅ Storage Drive (Account B): 8 items
   Path: /content/drive_storage/MyDrive/
   Sample: ['photos', 'backup', 'documents']
```

If you see this, **both drives are successfully mounted**! 🎉

## 📁 Storage Layout

Cell 1 creates this directory structure:

### Account A (Colab Pro - 15GB)
```
/content/drive_pro/MyDrive/drzero_biomedical/
├── logs/           # Training logs (text files)
├── configs/        # Configuration files
└── notebooks/      # Saved notebooks

Expected usage: <1 GB
```

### Account B (Storage - 80GB)
```
/content/drive_storage/MyDrive/drzero_biomedical/
├── corpus/
│   └── pubmed/
│       ├── pubmed-corpus.jsonl        (~8 GB)
│       └── pubmedbert_index.faiss     (~2 GB)
├── checkpoints/
│   ├── iter1_proposer/                (~10 GB)
│   ├── iter2_proposer/                (~10 GB)
│   └── iter3_proposer/                (~10 GB)
├── data/
│   └── biomedical/
│       └── training_seeds.parquet     (~5 GB)
└── outputs/
    └── evaluation_results/            (~2 GB)

Expected usage: ~47 GB
```

## 🐛 Troubleshooting

### Issue 1: "Already mounted to /content/drive"

**Cause**: You previously mounted a drive to the default `/content/drive` path.

**Solution**:
```python
# Use different mount points (already done in Cell 1)
drive.mount('/content/drive_pro')      # Not /content/drive
drive.mount('/content/drive_storage')  # Not /content/drive
```

### Issue 2: "Can't see second account in selection"

**Cause**: Not signed into both accounts in browser.

**Solution**:
1. Open new tab → accounts.google.com
2. Sign into both accounts
3. Return to Colab and re-run Cell 1
4. In auth screen, click "Use another account"

### Issue 3: "Permission denied on drive_storage"

**Cause**: Account B hasn't authorized Colab.

**Solution**:
1. Go to https://myaccount.google.com/permissions
2. Sign in as Account B
3. Check if "Google Drive" has Colab access
4. Re-run Cell 1 and re-authorize

### Issue 4: "Authentication popup blocked"

**Cause**: Browser is blocking popups.

**Solution**:
1. Allow popups for colab.research.google.com
2. Re-run Cell 1
3. Click the authentication links

### Issue 5: "Second mount fails silently"

**Cause**: Already authenticated with Account A in this session.

**Solution**:
```python
# Force re-authentication
drive.mount('/content/drive_storage', force_remount=True)
# Then click "Use another account" and select Account B
```

### Issue 6: "Wrong account mounted to drive_storage"

**Cause**: Accidentally selected Account A for both mounts.

**Solution**:
```python
# Unmount and remount
!fusermount -u /content/drive_storage
drive.mount('/content/drive_storage', force_remount=True)
# This time, be careful to select Account B
```

## ✅ Verification Checklist

Before proceeding with training, verify:

- [ ] Cell 1 shows "✅ Account A (Colab Pro) mounted successfully!"
- [ ] Cell 1 shows "✅ Account B (Storage) mounted successfully!"
- [ ] Can see different contents in each drive
- [ ] Directories created on both drives
- [ ] No error messages

To manually verify:

```python
import os

# Check Pro Drive (Account A)
print("Pro Drive contents:", os.listdir('/content/drive_pro/MyDrive')[:5])

# Check Storage Drive (Account B)
print("Storage Drive contents:", os.listdir('/content/drive_storage/MyDrive')[:5])

# They should show DIFFERENT contents!
```

## 🔒 Security & Privacy

**Is this safe?**

Yes! Each account maintains its own permissions:
- Account A can only access Account A's Drive
- Account B can only access Account B's Drive
- No cross-account access
- Standard Google OAuth security

**Who can see my files?**

- Only you (logged into respective accounts)
- Google (as per their privacy policy)
- No sharing between accounts unless you explicitly share

**Can I revoke access later?**

Yes:
1. Go to https://myaccount.google.com/permissions
2. Sign in to the account
3. Find "Google Colaboratory"
4. Click "Remove access"

## 💡 Tips & Best Practices

### Tip 1: Label Your Drives Clearly

In your code, always use clear variable names:

```python
# Good
PRO_LOGS = '/content/drive_pro/MyDrive/drzero_biomedical/logs'
STORAGE_CORPUS = '/content/drive_storage/MyDrive/drzero_biomedical/corpus'

# Bad (confusing!)
logs = '/content/drive_pro/...'
corpus = '/content/drive_storage/...'
```

### Tip 2: Keep Small Files on Pro Drive

Minimize Account B usage for faster syncing:
- ✅ Text logs → Pro Drive
- ✅ Config files → Pro Drive
- ✅ Small outputs (<100 MB) → Pro Drive
- ❌ Models → Storage Drive
- ❌ Corpus → Storage Drive
- ❌ Checkpoints → Storage Drive

### Tip 3: Periodic Sync Checks

During long training runs, periodically verify both drives:

```python
import os

def check_drives():
    try:
        os.listdir('/content/drive_pro/MyDrive')
        os.listdir('/content/drive_storage/MyDrive')
        print("✅ Both drives accessible")
    except:
        print("⚠️ Drive connection issue - may need remount")

# Run every few hours
check_drives()
```

### Tip 4: Use Symbolic Links (Advanced)

Create shortcuts for easier access:

```python
!ln -s /content/drive_storage/MyDrive/drzero_biomedical/corpus /content/corpus
!ln -s /content/drive_storage/MyDrive/drzero_biomedical/checkpoints /content/checkpoints

# Now you can use shorter paths
corpus_file = '/content/corpus/pubmed/pubmed-corpus.jsonl'
```

### Tip 5: Monitor Storage Usage

Check how much space you're using:

```python
import shutil

def check_storage(path):
    total, used, free = shutil.disk_usage(path)
    print(f"Path: {path}")
    print(f"  Total: {total / (2**30):.1f} GB")
    print(f"  Used: {used / (2**30):.1f} GB")
    print(f"  Free: {free / (2**30):.1f} GB")

# Check Pro Drive
check_storage('/content/drive_pro')

# Check Storage Drive
check_storage('/content/drive_storage')
```

## 📞 Getting Help

If dual-drive setup isn't working:

1. **Check the troubleshooting section above** first
2. **Verify browser setup**: Both accounts signed in
3. **Try incognito mode**: Sometimes helps with auth issues
4. **Restart runtime**: Runtime → Restart runtime
5. **Contact support**:
   - GitHub issues: (your repo)
   - Google Colab help: colab.research.google.com/help

## 🎉 Success!

Once Cell 1 completes successfully, you have:

✅ Two Google Drives mounted in one session
✅ 15 GB Pro account for small files
✅ 80 GB storage account for large files
✅ Total ~95 GB accessible storage
✅ A100 GPU from Colab Pro account
✅ No file transfer overhead

You're ready to train Dr. Zero! Continue to Cell 2.

---

**Next Steps:**
- Run Cell 2: Install dependencies
- Run Cell 3: Clone repository
- Run Cell 4: Configure training
- Run Cell 5+: Start training!

Good luck! 🚀
