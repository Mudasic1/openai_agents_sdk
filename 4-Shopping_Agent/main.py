from agents import Agent, RunConfig, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
import requests
from dotenv import load_dotenv
import os
import chainlit as cl
import asyncio

load_dotenv()

# The API endpoint
url = "https://template6-six.vercel.app/api/products"

response = requests.get(url)
data = response.json()

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not GEMINI_API_KEY:
    ValueError("Can't found API key")


external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=BASE_URL
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
async def getItem():
    return f"The product {data} is available"

agent = Agent(name="ShopMate AI agent", instructions="You are ShopMate AI, a friendly shopping assistant using the OpenAI Agents Framework. Your job is to help users find, compare, and recommend products using API data.",tools=[getItem])

@cl.on_chat_start
async def chat_start():
    await cl.Message(content="Hello I'm ShopMate AI. How can I help you?").send()

@cl.on_message
async def message(message: cl.Message):
    user_query = message.content
    res = await Runner.run(starting_agent=agent, input=user_query, run_config=config)
    await cl.Message(content=res.final_output).send()



