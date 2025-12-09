# Hosting Guide for RAG Streamlit App

This guide will help you deploy your RAG application so others can access it via a link.

## ⚠️ Important: Milvus Connection

Your app currently connects to Milvus at `localhost:19530`. For cloud hosting, you need a cloud-accessible Milvus instance.

### Option 1: Zilliz Cloud (Recommended - Free Tier Available)

1. **Sign up for Zilliz Cloud** (free tier available):
   - Go to https://cloud.zilliz.com/
   - Create a free account
   - Create a new cluster (free tier: 1GB storage)

2. **Get your connection details**:
   - After creating cluster, go to "Connect" tab
   - Copy the connection URI (looks like: `https://xxx.api.gcp-us-west1.vectordb.zillizcloud.com:19530`)
   - Copy your username and password

3. **Update `vectorstore.py`**:
   ```python
   # Change line 12 from:
   milvus_client = MilvusClient(uri="http://localhost:19530")
   
   # To:
   milvus_client = MilvusClient(
       uri="https://xxx.api.gcp-us-west1.vectordb.zillizcloud.com:19530",
       token="username:password"  # Format: "username:password"
   )
   ```

4. **Migrate your data**:
   - Run `ingest.py` again to upload your documents to the cloud Milvus instance

### Option 2: Self-Hosted Milvus on Cloud Server

If you prefer self-hosting:
- Deploy Milvus on AWS/GCP/Azure
- Update the connection URI in `vectorstore.py`
- Ensure the server is publicly accessible (with proper security)

---

## 🚀 Deploy to Streamlit Cloud (Easiest)

### Step 1: Prepare Your Repository

1. **Create a GitHub repository**:
   ```bash
   cd /Users/jainamsanghvi/Milvus_RAG_test
   git init
   git add new_codes/
   git add requirements.txt  # If at root, or move it
   git commit -m "Initial RAG app"
   ```

2. **Push to GitHub**:
   ```bash
   # Create repo on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Set Up Environment Variables

1. **Create `.streamlit/secrets.toml`** (for local testing) or use Streamlit Cloud secrets:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   ```

2. **Update `config.py`** to support Streamlit secrets:
   ```python
   import os
   from dotenv import load_dotenv
   
   # Try Streamlit secrets first (for cloud), then .env file
   if 'OPENAI_API_KEY' in os.environ:
       # Already set (from Streamlit secrets or environment)
       pass
   else:
       # Load from .env file (for local development)
       load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
   
   if "OPENAI_API_KEY" not in os.environ:
       raise ValueError("OPENAI_API_KEY not found in environment variables.")
   ```

### Step 3: Deploy to Streamlit Cloud

1. **Go to https://share.streamlit.io/**
2. **Sign in with GitHub**
3. **Click "New app"**
4. **Fill in the details**:
   - Repository: Select your GitHub repo
   - Branch: `main`
   - Main file path: `new_codes/app.py`
5. **Click "Deploy"**

### Step 4: Add Secrets in Streamlit Cloud

1. **In your Streamlit Cloud dashboard**, click on your app
2. **Go to "Settings" → "Secrets"**
3. **Add your secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. **Save** - The app will automatically redeploy

### Step 5: Share Your App

Once deployed, Streamlit Cloud will give you a URL like:
```
https://your-app-name.streamlit.app
```

Share this link with anyone! 🎉

---

## 🔧 Alternative: Other Hosting Platforms

### Railway.app
1. Connect GitHub repo
2. Set environment variables
3. Deploy

### Heroku
1. Create `Procfile`: `web: streamlit run new_codes/app.py --server.port=$PORT --server.address=0.0.0.0`
2. Deploy via Heroku CLI or GitHub integration

### AWS/GCP/Azure
- Use container services (ECS, Cloud Run, Container Instances)
- Or use EC2/Compute Engine with Streamlit

---

## 📝 Checklist Before Deploying

- [ ] Milvus connection updated to cloud instance (Zilliz Cloud recommended)
- [ ] Data migrated to cloud Milvus
- [ ] `requirements.txt` includes all dependencies
- [ ] `OPENAI_API_KEY` set in Streamlit Cloud secrets
- [ ] Code pushed to GitHub
- [ ] Tested locally with cloud Milvus connection

---

## 🐛 Troubleshooting

**App won't start:**
- Check Streamlit Cloud logs
- Verify all dependencies in `requirements.txt`
- Ensure `app.py` path is correct

**Milvus connection errors:**
- Verify cloud Milvus URI is correct
- Check firewall/security settings
- Ensure credentials are correct

**Missing API key:**
- Add `OPENAI_API_KEY` to Streamlit Cloud secrets
- Restart the app after adding secrets

---

## 💡 Quick Start (Zilliz Cloud + Streamlit Cloud)

1. Sign up for Zilliz Cloud → Create cluster → Get connection URI
2. Update `vectorstore.py` with Zilliz connection
3. Run `ingest.py` to upload data
4. Push code to GitHub
5. Deploy on Streamlit Cloud
6. Add `OPENAI_API_KEY` to secrets
7. Share your link! 🚀

