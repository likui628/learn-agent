import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

"""The xiaomi/mimo-v2-flash:free model previously triggered ModelHTTPError (404) when using output_type 
because its free endpoint did not support tool_choice for structured outputs. The google/gemini-2.5-flash-lite, 
despite supporting tool calling, often produces incorrect outputs (e.g., returning True even when the tool result is "loser") 
due to weaker reasoning and less strict adherence to tool results in structured bool scenarios compared to GPT-4o-mini.
"""
model = OpenRouterModel(
    'openai/gpt-4o-mini',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/pydantic-ai-examples',
        app_title='Pydantic AI examples',),
)


roulette_agent = Agent(
    model,
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18
result = roulette_agent.run_sync(
    'Put my money on square eighteen', deps=success_number)
print(result.output)
# > True

result = roulette_agent.run_sync(
    'I bet five is the winner', deps=success_number)
print(result.output)
# > False
