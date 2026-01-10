import os
from dotenv import load_dotenv
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


model = OpenRouterModel(
    'openai/gpt-4o-mini',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/pydantic-ai-examples',
        app_title='Pydantic AI examples',
    )
)

agent = Agent(
    model=model,
    output_type=[Box, str],
    system_prompt=(
        "Extract me the dimensions of a box, "
        "if you can't extract all data, ask the user to try again."
    ),
)

result = agent.run_sync('The box is 10x20x30')
print(result.output)
# > Please provide the units for the dimensions (e.g., cm, in, m).

result = agent.run_sync('The box is 10x20x30 cm')
print(result.output)
# > width=10 height=20 depth=30 units='cm'
