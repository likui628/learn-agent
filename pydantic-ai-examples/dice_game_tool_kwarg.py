import os
import random

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext, Tool

from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

load_dotenv()

model = OpenRouterModel(
    'mistralai/devstral-2512:free',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/pydantic-ai-examples',
        app_title='Pydantic AI examples',
    )
)

system_prompt = """\
You're a dice game, you should roll the die and see if the number
you get back matches the user's guess. If so, tell them they're a winner.
Use the player's name in the response.
"""


def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))


def get_player_name(ctx: RunContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


agent_a = Agent(
    model=model,
    deps_type=str,
    # the function signature is inspected to determine if the tool takes RunContext.
    tools=[roll_dice, get_player_name],
    system_prompt=system_prompt,
)
agent_b = Agent(
    model=model,
    deps_type=str,
    tools=[
        Tool(roll_dice, takes_ctx=False),
        Tool(get_player_name, takes_ctx=True),
    ],
    system_prompt=system_prompt,
)

dice_result = {}
dice_result['a'] = agent_a.run_sync('My guess is 6', deps='Yashar')
dice_result['b'] = agent_b.run_sync('My guess is 4', deps='Anne')
print(dice_result['a'].output)
# > Tough luck, Yashar, you rolled a 4. Better luck next time.
print(dice_result['b'].output)
# > Congratulations Anne, you guessed correctly! You're a winner!
