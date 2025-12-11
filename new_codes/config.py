import os
from dotenv import load_dotenv

# Try to load from Streamlit secrets (for cloud deployment)
def _load_streamlit_secrets():
    """Safely try to load secrets from Streamlit."""
    try:
        import streamlit as st
        # Check if we're in a Streamlit context
        if hasattr(st, 'secrets'):
            try:
                # Try to access secrets, but catch errors if not in Streamlit runtime
                if hasattr(st.secrets, 'get') or 'OPENAI_API_KEY' in dir(st.secrets):
                    # Use getattr to safely access without triggering file system checks
                    api_key = getattr(st.secrets, 'OPENAI_API_KEY', None)
                    if api_key:
                        os.environ['OPENAI_API_KEY'] = api_key
                        return True
            except (RuntimeError, AttributeError, KeyError):
                # Not in Streamlit runtime or secrets not available
                pass
    except ImportError:
        # Streamlit not installed
        pass
    return False

# Try Streamlit secrets first
if not _load_streamlit_secrets():
    # Fall back to .env file (for local development)
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Don't raise error at import time - let OpenAI client handle missing key
# This allows Streamlit to initialize first, then secrets will be available