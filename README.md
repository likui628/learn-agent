# Pydantic AI Examples

A demonstration project showcasing the use of Pydantic AI for building AI-powered applications.

## Installation

Ensure you have Python 3.14+ installed. This project uses `uv` for dependency management.

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd pydantic-ai-examples
uv sync
```

## Environment Setup

Create a `.env` file in the root directory and add your API keys:

```
OPENROUTER_API_KEY=your_openrouter_api_key
```

## How to Run

Run the example:

```bash
uv run -m examples.<example_module_name>
```

## Models
| Model | Price (Input/Output) | Note |
| --- | --- | --- |
| anthropic/claude-3-haiku | 0.25/1.25 | |
| openai/gpt-4o-mini | 0.15/0.60 |  |
| xiaomi/mimo-v2-flash:free | - | Did not support tool_choice for structured outputs |
| mistralai/devstral-2512:free | - |  |
| anthropic/claude-haiku-4.5 | 1.00/5.00 |  |
| google/gemini-2.5-flash-lite | 0.10/0.40|  Agentic stability is slightly inferior to Claude/GPT. |
## Dependencies

- pydantic-ai: Core AI framework
- python-dotenv: Environment variable management