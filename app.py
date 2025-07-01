import streamlit as st
import asyncio
from main import run_review


st.set_page_config(page_title="Literature Review Assistant", page_icon="📚", layout="centered")


st.markdown("## 📚 **Literature Review Assistant**")


topic = st.text_input("Research topic", placeholder="e.g., Large Language Models")
num_papers = st.slider("Number of papers", min_value=1, max_value=10, value=5)


if st.button("Search"):
    if not topic.strip():
        st.warning("⚠️ Please enter a research topic.")
    else:
        with st.spinner(" Fetching and summarizing papers..."):
            try:
                result = asyncio.run(run_review(topic, num_papers))
                st.markdown(result) 
            except Exception as e:
                st.error(f" Error: {str(e)}")

