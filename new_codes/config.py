import os
from dotenv import load_dotenv

# Try Streamlit secrets first (for cloud deployment), then .env file (for local development)
try:
    import streamlit as st
    # If running in Streamlit, check secrets first
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
except ImportError:
    # Not running in Streamlit, continue with .env file
    pass

# Load environment variables from .env file (for local development)
# This will look for .env in the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Verify OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file or Streamlit secrets.")