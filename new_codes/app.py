"""
Streamlit Web Interface for RAG System
Quick sharing interface for the RAG query system.
"""

import streamlit as st
from query import query_rag, format_answer
import time
import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(__file__))

# Page configuration
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .answer-box p {
        color: #ffffff;
    }
    .answer-box * {
        color: #ffffff;
    }
    .source-box {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .source-box * {
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🔍 RAG Query System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents and get AI-powered answers with source citations.</p>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    collection_name = st.text_input(
        "Collection Name", 
        value="my_rag_collection",
        help="Name of the Milvus collection to search"
    )
    
    st.subheader("🔎 Search Parameters")
    top_k = st.slider(
        "Top K (chunks to retrieve)", 
        50, 500, 200, 50,
        help="Number of chunks to retrieve from Milvus vector search"
    )
    
    st.subheader("🎯 MMR Parameters")
    top_n = st.slider(
        "Top N (for MMR)", 
        5, 30, 15, 1,
        help="Number of top chunks to consider for diversity selection"
    )
    final_k = st.slider(
        "Final K (diverse chunks)", 
        3, 10, 6, 1,
        help="Final number of diverse chunks to use for answer generation"
    )
    lambda_mmr = st.slider(
        "Lambda (relevance vs diversity)", 
        0.0, 1.0, 0.7, 0.1,
        help="Higher = more weight on relevance, Lower = more weight on diversity"
    )
    
    st.subheader("🔧 Advanced")
    include_embeddings = st.checkbox(
        "Include embeddings", 
        value=True,
        help="Retrieve embeddings from Milvus (faster, no recomputation)"
    )
    show_stats = st.checkbox(
        "Show pipeline statistics", 
        value=True,
        help="Display timing and statistics for each pipeline step"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline Flow")
    st.markdown("""
    1. **Embed** query
    2. **Search** Milvus (top K)
    3. **Rerank** chunks
    4. **MMR** select diverse chunks
    5. **Generate** answer with GPT-4o
    """)

# Main query interface
st.markdown("### 💬 Ask a Question")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is machine learning?",
    key="query_input"
)

# Search button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Process query
if search_button or (query and query.strip()):
    if query.strip():
        with st.spinner("🔄 Processing your query..."):
            # Run the RAG pipeline
            start_time = time.time()
            try:
                result = query_rag(
                    query=query,
                    collection_name=collection_name,
                    top_k_search=top_k,
                    top_n_mmr=top_n,
                    final_k_mmr=final_k,
                    lambda_mmr=lambda_mmr,
                    include_embeddings=include_embeddings,
                    verbose=False  # Don't print to console in Streamlit
                )
                total_time = time.time() - start_time
                
                # Display answer
                st.markdown("---")
                st.markdown("### 📝 Answer")
                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Display sources
                if result.get('sources'):
                    st.markdown("---")
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(result['sources'], 1):
                        st.markdown(f'<div class="source-box">{i}. <strong>{source}</strong></div>', unsafe_allow_html=True)
                
                # Display statistics
                if show_stats and 'pipeline_stats' in result:
                    st.markdown("---")
                    st.markdown("### 📊 Pipeline Statistics")
                    stats = result['pipeline_stats']
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Chunks Found", stats.get('chunks_found', 0))
                    with col2:
                        st.metric("Chunks Reranked", stats.get('chunks_reranked', 0))
                    with col3:
                        st.metric("Chunks Selected", stats.get('chunks_selected', 0))
                    with col4:
                        st.metric("Chunks Used", result.get('chunks_used', 0))
                    
                    # Detailed timing
                    with st.expander("⏱️ Detailed Timing"):
                        timing_cols = st.columns(5)
                        timing_data = [
                            ('Embedding', stats.get('embedding_time', 0)),
                            ('Search', stats.get('search_time', 0)),
                            ('Rerank', stats.get('rerank_time', 0)),
                            ('MMR', stats.get('mmr_time', 0)),
                            ('Answer', stats.get('answer_time', 0))
                        ]
                        for i, (label, time_val) in enumerate(timing_data):
                            with timing_cols[i]:
                                st.metric(label, f"{time_val:.2f}s")
                        
                        st.markdown(f"**Total Time:** {total_time:.2f}s")
                
                # Success message
                st.success("✅ Query processed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.exception(e)
    else:
        st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>RAG Pipeline:</strong> Embed → Search → Rerank → MMR → Answer</p>
    <p>Powered by Milvus, OpenAI, and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Instructions in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter your question in the text box
    2. Adjust settings if needed (optional)
    3. Click **Search** or press Enter
    4. View the answer and sources
    
    **Tip:** Use example queries to get started!
    """)

