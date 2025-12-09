# RAG System - Streamlit Frontend

## Quick Start

### 1. Run the Streamlit App

```bash
cd /Users/jainamsanghvi/Milvus_RAG_test/new_codes
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### 2. Share with Others

**Option A: Local Network Sharing**
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Then share your IP address: `http://YOUR_IP:8501`

**Option B: Streamlit Cloud (Free)**
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

**Option C: Other Cloud Platforms**
- Heroku
- Railway
- AWS/GCP/Azure

## Prerequisites

Before running, ensure:
1. ✅ Milvus is running: `docker ps | grep milvus` or check `http://localhost:19530`
2. ✅ Data is ingested: Run `python3 ingest.py` first
3. ✅ `.env` file exists with `OPENAI_API_KEY`
4. ✅ All packages installed (see requirements below)

## Features

- 🔍 Query interface with example questions
- ⚙️ Adjustable settings (top_k, MMR parameters)
- 📊 Pipeline statistics
- 📚 Source citations
- 🎨 Clean, modern UI

## Troubleshooting

**Port already in use:**
```bash
streamlit run app.py --server.port 8502
```

**Can't connect to Milvus:**
- Check if Milvus is running: `docker ps`
- Verify connection: `curl http://localhost:19530/healthz`

**No results:**
- Make sure data is ingested: `python3 ingest.py`
- Check collection name matches in sidebar

