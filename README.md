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

## Dependencies

- pydantic-ai: Core AI framework
- python-dotenv: Environment variable management