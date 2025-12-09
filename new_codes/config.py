import os
from dotenv import load_dotenv

# Try to load from Streamlit secrets (for cloud deployment)
def _load_streamlit_secrets():
    """Safely try to load secrets from Streamlit."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets:
            if 'OPENAI_API_KEY' in st.secrets:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                return True
    except (ImportError, AttributeError, RuntimeError, KeyError):
        # Streamlit not available, not initialized, or secrets not set
        pass
    return False

# Try Streamlit secrets first
if not _load_streamlit_secrets():
    # Fall back to .env file (for local development)
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Don't raise error at import time - let OpenAI client handle missing key
# This allows Streamlit to initialize first, then secrets will be available