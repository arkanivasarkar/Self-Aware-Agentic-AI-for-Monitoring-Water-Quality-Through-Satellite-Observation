from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image
import openai
import os
import numpy as np
from agno.media import Image
from agno.tools.pubmed import PubmedTools
from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat

# openai.api_key=os.getenv("OPENAI_API_KEY")
os.environ['API_KEY'] = 'ag-58BhPxCx8IoiNEX0Yy-hWtKgaYyoh66MszhvhZRM9_g'
os.environ['GROQ_API_KEY'] = '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'
    

# def call_agents():
    # Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions=[
        "Return output in a paragraph and not bullets, as if you are talking to me. Trust the source and confidently state the cause."
    ],
    tools=[DuckDuckGoTools()],
    markdown=True,
)

rewriter_agent = Agent(
    name="Rewriter Agent",
    role="Rewrite paragraph based on given prompt",
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions=[
        "Revise the following paragraph in 30 words to remove any skeptical language. Replace words like 'mostly,' 'probably,' 'most likely,' and similar expressions with more confident and assertive wording. Ensure the tone remains professional and natural while strengthening the claims. Keep the meaning intact. Also, remove any markdown formatting such as bold (**), italics (*), or other markdown tags while keeping the content clear and professional."
    ],
    markdown=True,
)

# Literature search agent
literature_search_agent = Agent(name="Literature Search Agent",
    role="Search literatures on water quslity monitoring using satellite images, machine learning based detection and how to improve degrading water quality",
    instructions=[
        "Focus on water qaulity monitoring using satellite images from SENTINEL and Copernicus.",
        "Prefer machine learning based monitoring",
        "Try to find how to improve the water quality after detection",
        "Recommend methods to improve water quality",
    ],
    model=Groq(id="llama-3.3-70b-specdec"),
    tools=[PubmedTools()],
    markdown=True,
)
    

multi_ai_agent = Agent(
    team=[web_search_agent,rewriter_agent],
    model=Groq(id="llama-3.2-3b-preview"),
    show_tool_calls=True,
    markdown=True,
    add_history_to_messages=True,
    num_history_responses=10,
)

    #return multi_ai_agent

# response: RunResponse = web_search_agent.run("My satellite image analysis from SENTINEL dataset tell me that the water qaulity in my fish farm is going bad.Identify the reasons. Answer in a paragraph in brief.")
# #print(response.content)
# response2: RunResponse = rewriter_agent.run(response.content)
# print(response2.content)
