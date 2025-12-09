# Quick Deploy Guide 🚀

## Fastest Way to Get Your App Online

### Prerequisites
1. GitHub account
2. Zilliz Cloud account (free): https://cloud.zilliz.com/
3. OpenAI API key

---

## Step-by-Step (15 minutes)

### 1. Set Up Cloud Milvus (Zilliz Cloud) - 5 min

1. Go to https://cloud.zilliz.com/ and sign up (free)
2. Create a new cluster (free tier: 1GB)
3. Wait for cluster to be ready (~2 minutes)
4. Click "Connect" → Copy:
   - **URI** (e.g., `https://xxx.api.gcp-us-west1.vectordb.zillizcloud.com:19530`)
   - **Username** and **Password**

### 2. Update Milvus Connection - 2 min

Edit `vectorstore.py` line 12:

```python
# Replace this:
milvus_client = MilvusClient(uri="http://localhost:19530")

# With this (use your Zilliz credentials):
milvus_client = MilvusClient(
    uri="YOUR_ZILLIZ_URI_HERE",
    token="YOUR_USERNAME:YOUR_PASSWORD"
)
```

### 3. Upload Your Data - 3 min

Run ingestion to upload your documents to cloud Milvus:

```bash
cd /Users/jainamsanghvi/Milvus_RAG_test/new_codes
python3 ingest.py
```

Wait for it to complete.

### 4. Push to GitHub - 2 min

```bash
cd /Users/jainamsanghvi/Milvus_RAG_test
git init
git add new_codes/
git commit -m "RAG app ready for deployment"
# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### 5. Deploy on Streamlit Cloud - 3 min

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Main file: `new_codes/app.py`
6. Click "Deploy"

### 6. Add API Key - 1 min

1. In Streamlit Cloud dashboard → Your app → Settings → Secrets
2. Add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
3. Save (app auto-redeploys)

### 7. Share! 🎉

Your app URL: `https://your-app-name.streamlit.app`

---

## ✅ Done!

Your RAG app is now live and accessible to anyone with the link!

---

## Troubleshooting

**"Milvus connection failed"**
- Double-check Zilliz URI and credentials
- Ensure cluster is running (check Zilliz dashboard)

**"OPENAI_API_KEY not found"**
- Add it to Streamlit Cloud secrets (Settings → Secrets)

**"Module not found"**
- Check `requirements.txt` has all packages
- Streamlit Cloud will auto-install on deploy

---

## Need Help?

See `HOSTING_GUIDE.md` for detailed instructions and alternatives.

