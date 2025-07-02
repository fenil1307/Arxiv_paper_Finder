import streamlit as st
import asyncio
import nest_asyncio
from main import build_team

nest_asyncio.apply()

st.set_page_config(page_title="ArXplore", page_icon="📚", layout="centered")
st.markdown("## 📚 **ArXplore: Literature Review Assistant**")

topic = st.text_input("Research topic", placeholder="e.g., Large Language Models")
num_papers = st.slider("Number of papers", min_value=1, max_value=10, value=5)

async def run_custom_team(topic: str, num_papers: int):
    team = build_team()
    task = f"""Conduct a literature review on the topic '{topic}' and return exactly {num_papers} papers.

For the summarizer: Please format each paper with the following structure:
- **Paper Title:** [Title as a clickable link]
- **Authors:** [List of authors]  
- **Problem Tackled:** [Brief description of the problem addressed]
- **Key Contribution:** [Main contribution of the paper]

Use bullet points for each paper and make it easily readable."""

    summary_result = None

    async for msg in team.run_stream(task=task):
        if hasattr(msg, 'content') and msg.content:
            if hasattr(msg, 'source') and msg.source == 'summarizer_agent':
                summary_result = msg.content

    return summary_result

if st.button("Search"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner(" Fetching and summarizing papers..."):
            try:
                loop = asyncio.get_event_loop()
                summary_result = loop.run_until_complete(run_custom_team(topic, num_papers))

                if summary_result:
                    st.markdown("###  **Literature Review Summary:**")
                    st.markdown(f"""
                    <div style="text-align: justify; text-justify: inter-word; line-height: 1.6; margin-top: 10px;">
                    {summary_result}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No summary generated. Please try again.")
            except Exception as e:
                st.error(f" Error: {str(e)}")
                st.info("Check your OpenAI API key and internet connection.")
