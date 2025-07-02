from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
import os
import arxiv
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


openai_brain = OpenAIChatCompletionClient(
    model='gpt-4o',
    api_key=os.getenv('OPENAI_API_KEY')
)


def arxiv_search(query: str, max_results: int) -> List[Dict]:
    """Return a compact list of arXiv papers matching query."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers: List[Dict] = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "pdf_url": result.pdf_url,
        })
    return papers


arxiv_researcher_agent = AssistantAgent(
    name='arxiv_search_agent',
    description='Creates arXiv queries and retrieves candidate papers',
    model_client=openai_brain,
    tools=[arxiv_search],
    system_message=(
        "Given a user topic, think of the best arXiv query and call the tool. "
        "Always fetch five-times the papers requested so you can down-select the most relevant ones. "
        "Return exactly the number of papers requested in concise JSON format."
    )
)


summarizer_agent = AssistantAgent(
    name='summarizer_agent',
    description='Summarizes the result',
    model_client=openai_brain,
    system_message=(
        "You are an expert researcher. When you receive the JSON list of papers, "
        "write a literature review-style report in Markdown:\n"
        "1. Start with a short introduction.\n"
        "2. Then, one bullet per paper with: title (as Markdown link), authors, problem tackled, and key contribution.\n"
        "3. End with a single takeaway sentence."
    )
)


def build_team() -> RoundRobinGroupChat:
    return RoundRobinGroupChat(
        participants=[arxiv_researcher_agent, summarizer_agent],
        max_turns=2
    )
