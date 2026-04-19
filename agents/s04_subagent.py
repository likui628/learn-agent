#!/usr/bin/env python3
# Harness: context isolation -- protecting the model's clarity of thought.
"""
s04_subagent.py - Subagents
Spawn a child agent with fresh messages=[]. The child works in its own
context, sharing the filesystem, then returns only a summary to the parent.
    Parent agent                     Subagent
    +------------------+             +------------------+
    | messages=[...]   |             | messages=[]      |  <-- fresh
    |                  |  dispatch   |                  |
    | tool: task       | ----------> | while tool_use:  |
    |   prompt="..."   |             |   call tools     |
    |   description="" |             |   append results |
    |                  |  summary    |                  |
    |   result = "..." | <---------  | return last text |
    +------------------+             +------------------+
              |
    Parent context stays clean.
    Subagent context is discarded.
Key insight: "Fresh messages=[] gives context isolation. The parent stays clean."
Note: Real Claude Code also uses in-process isolation (not OS-level process
forking). The child runs in the same process with a fresh message array and
isolated tool context -- same pattern as this teaching implementation.
    Comparison with real Claude Code:
    +-------------------+------------------+----------------------------------+
    | Aspect            | This demo        | Real Claude Code                 |
    +-------------------+------------------+----------------------------------+
    | Backend           | in-process only  | 5 backends: in-process, tmux,    |
    |                   |                  | iTerm2, fork, remote             |
    | Context isolation | fresh messages=[]| createSubagentContext() isolates  |
    |                   |                  | ~20 fields (tools, permissions,  |
    |                   |                  | cwd, env, hooks, etc.)           |
    | Tool filtering    | manually curated | resolveAgentTools() filters from |
    |                   |                  | parent pool; allowedTools         |
    |                   |                  | replaces all allow rules         |
    | Agent definition  | hardcoded system | .claude/agents/*.md with YAML    |
    |                   | prompt           | frontmatter (AgentTemplate)      |
    +-------------------+------------------+----------------------------------+
"""

import json
import os
import subprocess
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

print("Starting agent loop. Type a prompt, and the model will respond with bash commands to execute. Type 'q' or Ctrl+C to quit.")

load_dotenv(override=True)

WORKDIR = Path.cwd()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("MODEL_ID", "anthropic/claude-haiku-4.5")

PLAN_REMINDER_INTERVAL = 3


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."

CHILD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }, {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
                "name": "write_file",
                "description": "Write content to file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
        },
    },
    {
        "type": "function",
        "function": {
                "name": "edit_file",
                "description": "Replace exact text in file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                    "required": ["path", "old_text", "new_text"],
                },
        },
    },
]

PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        },
    }
]

TOOL_HANDLERS = {
    "bash": lambda args: run_bash(args["command"]),
    "read_file": lambda args: read_file(args["path"], args.get("limit", 10000)),
    "write_file": lambda args: write_file(args["path"], args["content"]),
    "edit_file": lambda args: edit_file(args["path"], args["old_text"], args["new_text"]),
    "task": lambda args: spawn_subagent(args["prompt"]),
}


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def read_file(path: str, limit: int = 10000) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    try:
        safe_path(path).write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"Error: {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        p = safe_path(path)
        text = p.read_text()
        if old_text not in text:
            return f"Error: '{old_text}' not found in {path}"
        new_content = text.replace(old_text, new_text, 1)
        p.write_text(new_content)
        return f"Replaced '{old_text}' with '{new_text}' in {path}"
    except Exception as e:
        return f"Error: {e}"


def spawn_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]
    for _ in range(30):  # limit subagent to 5 tool calls
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUBAGENT_SYSTEM}
                      ] + sub_messages,
            tools=CHILD_TOOLS,
            max_tokens=8000,
        )
        msg = response.choices[0].message
        sub_messages.append({"role": "assistant", "content": msg.content,
                             "tool_calls": msg.tool_calls})
        if response.choices[0].finish_reason != "tool_calls":
            break

        results = []
        for tc in msg.tool_calls:
            handler = TOOL_HANDLERS.get(tc.function.name)
            print(
                f"\033[33m[subagent] $ {tc.function.name}({tc.function.arguments})\033[0m")
            if handler:
                args = json.loads(tc.function.arguments)
                output = handler(args)
            else:
                output = f"Error: No handler for tool '{tc.function.name}'"
            print(
                f"\033[33m[subagent] > {tc.function.name}: {str(output)[:200]}\033[0m")
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output)[:50000],
            })
        sub_messages.extend(results)
     # Only the final text returns to the parent -- child context is discarded
    final_text = msg.content if msg.content else ""
    return final_text.strip() or "(no summary)"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        msg = response.choices[0].message
        # Append assistant turn (convert to dict for history)
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": msg.tool_calls})

        if response.choices[0].finish_reason != "tool_calls":
            # Print final text response
            if msg.content:
                print(msg.content)
            return

        # Execute each tool call, collect results
        results = []
        for tc in msg.tool_calls:
            if tc.function.name == "task":
                output = spawn_subagent(json.loads(
                    tc.function.arguments)["prompt"])
            else:
                handler = TOOL_HANDLERS.get(tc.function.name)
                print(
                    f"\033[33m$ {tc.function.name}({tc.function.arguments})\033[0m")
                if handler:
                    args = json.loads(tc.function.arguments)
                    output = handler(args)
                else:
                    output = f"Error: No handler for tool '{tc.function.name}'"

                print(f"> {tc.function.name}: {str(output)[:200]}")

            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })

        messages.extend(results)


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
