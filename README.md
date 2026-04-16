# Learn Agent

A demonstration project showcasing the use of Pydantic AI for building AI-powered applications.

## Installation

Ensure you have Python 3.14+ installed. This project uses `uv` for dependency management.

Clone the repository and install dependencies:

```bash
git clone https://github.com/likui628/learn-agent
cd learn-agent
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
uv run -m agents.s01_agent_loop
```
