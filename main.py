from dotenv import load_dotenv
import os
from agents import Agent, RunConfig, OpenAIChatCompletionsModel, Runner, AsyncOpenAI

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


agent = Agent(
    name="Translation Agent",
    instructions="Translate the given text from English to french."
)

response = Runner.run_sync(
    agent,
    input="my name is safwan, i am student at Governor house sindh karachi",
    run_config=config,
)

print(response)