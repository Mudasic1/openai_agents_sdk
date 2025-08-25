from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import chainlit as cl



load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

# External Client
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

# Lyric Poetry
Lyric_Agent = Agent(name="Lyric Poetry Agent", instructions="You are a Lyric Poetry agent you will tell about poetry that are about feelings and thoughts, like songs or poem about being sad or happy", model=model)

# Narrative Poetry Agent
Narrative_Agent = Agent(name="Narrative Poetry Agent", instructions="You are a Narrative Poetry Agent you will tells poetry that about a story with characters and events just like a regular story but written in poem form with rhymes or special rhythm")

# Dramatic Poetry Agent
Dramatic_Agent = Agent(name="Dramatic Poetry Agent", instructions="You are a Dramatic Agent that tell poetry is meant to be performed out loud, where someone acts like a character and speaks their thoughts and feelings to an audience (acting in a theatre)", model=model)


# Triage Agent/ Orchestrator / Parent Agent
Triage_Agent = Agent(name="Triage Agent", instructions="You are a triage agent that will take a user's input and decide which agent to use to generate a response", handoffs=[Lyric_Agent, Narrative_Agent, Dramatic_Agent], model=model)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
    content="Hello 👋, I'm your Poetry Agent. I can create Poetry 🌷 for you such as:\n\n"
            "- Lyric Poetry 🎵✨💭\n"
            "- Narrative Poetry 📖🛤️👤\n"
            "- Dramatic Poetry 🎭🎤🔥"
            ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    user_query = msg.content
    res = await Runner.run(starting_agent=Triage_Agent, input=user_query, run_config=config)
    await cl.Message(content=res.final_output).send()