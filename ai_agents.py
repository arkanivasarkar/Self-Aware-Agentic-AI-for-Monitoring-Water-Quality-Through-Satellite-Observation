from agno.agent import Agent,AgentMemory
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.pubmed import PubmedTools
from agno.media import Image
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder
import os
import openai

os.environ['API_KEY'] = '' #Enter Agno api key
os.environ['GROQ_API_KEY'] = '' #Enter Qroq api key
os.environ['OPENAI_API_KEY'] = '' #Enter openai api key


# web search agent for searching the internet
web_search_agent = Agent(
    name="Web Search Agent",
    description="You are an assistant helping fish farmers to monitor water quality and help them manage their fish farm.",
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions=[
        "Search the web for the information regarding water quality monitoring and management.",
        "Check for reliability and add references."
    ],
    tools=[DuckDuckGoTools()],
    add_history_to_messages=True,
    num_history_responses=3,
    show_tool_calls=True,
    read_chat_history=True,
    markdown=True,
)


# Literature search agent for searching through published literatures
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
    add_history_to_messages=True,
    num_history_responses=3,
    show_tool_calls=True,
    read_chat_history=True,
    markdown=True,
)


# RAG agent to query documents
rag_agent = Agent(
    model=Groq(id="llama-3.3-70b-specdec"),
    description="You are an expert in water quality monitoring for precision aquaculture.",
    instructions=[
        "Search your knowledge base.",
        "Focus on extracting the most relevant and accurate information from the context. ",
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://iris.who.int/bitstream/handle/10665/44584/9789241548151_eng.pdf",
              "https://www.fao.org/fileadmin/templates/SEC/docs/Fishery/Fisheries_Events_2012/Water_Quality_for_Aquaculture_and_Impact_of_Aquaculture_to_Environments.pdf",
              "https://food.ec.europa.eu/system/files/2022-07/aw_platform_plat-conc_guide_farmed-fish_en.pdf",
              "https://www.fao.org/4/e4223e/e4223e.pdf",
              "https://rr-europe.woah.org/app/uploads/2023/11/3_eu_platform_animal_welfare_broberg.pdf"],
        vector_db=LanceDb(
            uri="vectordb/rag_knowledge",
            table_name="knowledge",
            search_type=SearchType.hybrid,
            embedder=SentenceTransformerEmbedder(),
        ),
    ),
    markdown=True
)
rag_agent.knowledge.load(recreate=False) #store in vector database only once


# tool calling agent for running AI models for real time monitoring
real_time_monitoring_agent = Agent(
    name="Tool Identifier Agent",
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions=[
        "Based on the prompt, understand whether the AI model has to be called.",
        "The prompt can be that the user is sceptical about the information.",
        "Return 'run model' if the AI model has to be run, else return 'no'",
    ],
    markdown=True,
)


# Rewriter agent to rephrase everything in simple language
rewriter_agent = Agent(
    name="Rewriter Agent",
    role="Rewrite paragraph based on given prompt",
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions=[
        "Rewrite the paragraph so that in feels like someone is having normal conversation.",
        "Remove technical jargons and use simpler and easy to understand language."
    ],
    markdown=True,
)


# Multi agent call
multi_ai_agent = Agent(
    team=[web_search_agent,
          literature_search_agent,
          rag_agent,
          real_time_monitoring_agent,
          rewriter_agent],
    model=Groq(id="llama-3.3-70b-specdec"),
    instructions = [
        "Search the web for the information regarding water quality monitoring and management.",
        "Check for reliability and add references.",
        "Analyze the query to identify key aspects and prioritize the most relevant sources.",
        "Cross-verify the information from multiple sources to ensure accuracy.",
        "Focus on water quality monitoring using satellite images from SENTINEL and Copernicus.",
        "Prefer machine learning-based monitoring.",
        "Try to find how to improve the water quality after detection.",
        "Recommend methods to improve water quality.",
        "Critically evaluate the relevance of the literature based on the query context.",
        "Summarize findings in a way that links the problem, approach, and potential solutions.",
        "Search your knowledge base and focus on extracting the most relevant and accurate information from the context.",
        "Identify patterns in the data to better understand the relationship between the input and the required response.",
        "Formulate responses that connect the user's query to the retrieved knowledge with clear reasoning.",
        "Based on the prompt, understand whether the AI model has to be called.",
        "The prompt can be that the user is sceptical about the information.",
        "Return 'run model' if the AI model has to be run, else return 'no'.",
        "Evaluate the input to determine if the task requires additional computation or verification.",
        "Break down the user's request to assess the type of reasoning required and whether a model's output will provide meaningful value.",
        "Rewrite the paragraph so that it feels like someone is having a normal conversation.",
        "Remove technical jargons and use simpler and easy-to-understand language.",
        "Ensure the rewritten response maintains the original intent while being more approachable.",
        "Think about how to make the content relatable and engaging for the intended audience.",
        "When simplifying, avoid oversimplifying critical details to retain key information."
    ],
    memory=AgentMemory(
        db=LanceDb(table_name="agent_memory", uri="vectordb/agentic_memory",), create_user_memories=True, create_session_summary=True
    ),
    reasoning=True, 
    structured_outputs=True,
    show_tool_calls=True,
    markdown=True,
    add_history_to_messages=True,
    num_history_responses=10,
)



# response: RunResponse = web_search_agent.run("My satellite image analysis from SENTINEL dataset tell me that the water qaulity in my fish farm is going bad.Identify the reasons. Answer in a paragraph in brief.")
# #print(response.content)
# response2: RunResponse = rewriter_agent.run(response.content)
# print(response2.content)
