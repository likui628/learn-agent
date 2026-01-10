from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai import Agent
from dotenv import load_dotenv
import os

load_dotenv()


model = OpenRouterModel(
    'anthropic/claude-haiku-4.5',
    provider=OpenRouterProvider(api_key=os.getenv('OPENROUTER_API_KEY')),
)

agent = Agent(
    model,
    instructions='Be concise, reply with one sentence.',
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
