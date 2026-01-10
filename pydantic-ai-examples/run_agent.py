import asyncio
import os
from dotenv import load_dotenv
from pydantic_ai import Agent
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

agent = Agent(
    model=model,
    model_settings={
        'temperature': 0.7,
        'max_tokens': 1024
    },
)

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
# > The capital of Italy is Rome.


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    # > The capital of France is Paris.

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)
            # > The capital of
            # > The capital of the UK is
            # > The capital of the UK is London.

    async for event in agent.run_stream_events('What is the capital of Mexico?'):

        print(
            event.part.content if hasattr(event, 'part') else
            event.delta.content_delta if hasattr(event, 'delta') else
            '', end=''
        )
    print()
asyncio.run(main())
