import asyncio
from datetime import date
import os
from typing import AsyncIterable
from dotenv import load_dotenv
from pydantic_ai import Agent, AgentStreamEvent, FinalResultEvent, FinalResultEvent, FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent, RunContext, TextPartDelta, ThinkingPartDelta, ToolCallPartDelta
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

weather_agent = Agent(model=model)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext,
    location: str,
    forecast_date: date,
) -> str:
    return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'

output_messages: list[str] = []


async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        output_messages.append(
            f'[Request] Starting part {event.index}: {event.part!r}')
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            output_messages.append(
                f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(
                f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ToolCallPartDelta):
            output_messages.append(
                f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
    elif isinstance(event, FunctionToolCallEvent):
        output_messages.append(
            f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
        )
    elif isinstance(event, FunctionToolResultEvent):
        output_messages.append(
            f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
    elif isinstance(event, FinalResultEvent):
        output_messages.append(
            f'[Result] The model starting producing a final result (tool_name={event.tool_name})')


async def event_stream_handler(ctx: RunContext, event: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in event:
        await handle_event(event)


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async with weather_agent.run_stream(user_prompt, event_stream_handler=event_stream_handler) as run:
        final_output = ''
        async for output in run.stream_text():
            final_output = output

        output_messages.append(f'[Final Output] {final_output}')
    print('\n'.join(output_messages))

asyncio.run(main())
