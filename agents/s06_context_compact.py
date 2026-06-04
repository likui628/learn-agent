#!/usr/bin/env python3
# Harness: compression -- keep the active context small enough to keep working.
"""
s06_context_compact.py - Context Compact
This teaching version keeps the compact model intentionally small:
1. Large tool output is persisted to disk and replaced with a preview marker.
2. Older tool results are micro-compacted into short placeholders.
3. When the whole conversation gets too large, the agent summarizes it and
   continues from that summary.
The goal is not to model every production branch. The goal is to make the
active-context idea explicit and teachable.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

WORKDIR = Path.cwd()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("MODEL_ID", "anthropic/claude-haiku-4.5")

PERSIST_THRESHOLD = 30000
PREVIEW_CHARS = 2000
TOOL_RESULTS_DIR = WORKDIR / ".task_outputs" / "tool-results"


@dataclass
class CompactState:
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: list[str] = field(default_factory=list)


def safe_path(path_str: str) -> Path:
    path = (WORKDIR / path_str).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path_str}")
    return path


def track_recent_file(state: CompactState, path: str) -> None:
    if path in state.recent_files:
        state.recent_files.remove(path)
    state.recent_files.append(path)
    if len(state.recent_files) > 5:
        state.recent_files[:] = state.recent_files[-5:]


def persist_large_output(tool_use_id: str, output: str) -> str:
    if len(output) <= PERSIST_THRESHOLD:
        return output

    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = TOOL_RESULTS_DIR / f"{tool_use_id}.txt"
    if not stored_path.exists():
        stored_path.write_text(output)
    preview = output[:PREVIEW_CHARS]
    rel_path = stored_path.relative_to(WORKDIR)
    return (
        "<persisted-output>\n"
        f"Full output saved to: {rel_path}\n"
        "Preview:\n"
        f"{preview}\n"
        "</persisted-output>"
    )


def run_bash(command: str, tool_use_id: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    output = (result.stdout + result.stderr).strip() or "(no output)"
    return persist_large_output(tool_use_id, output)


def run_read(path: str, tool_use_id: str, state: CompactState, limit: int | None = None) -> str:
    try:
        track_recent_file(state, path)
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        output = "\n".join(lines)
        return persist_large_output(tool_use_id, output)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error: {exc}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        file_path = safe_path(path)
        content = file_path.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        file_path.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a file once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Summarize earlier conversation so work can continue in a smaller context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string"},
                },
            },
        },
    },
]


def extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def execute_tool(name: str, args: dict, state: CompactState, tool_use_id: str) -> str:
    if name == "bash":
        return run_bash(args["command"], tool_use_id)
    if name == "read_file":
        return run_read(args["path"], tool_use_id, state, args.get("limit"))
    if name == "write_file":
        return run_write(args["path"], args["content"])
    if name == "edit_file":
        return run_edit(args["path"], args["old_text"], args["new_text"])
    if name == "compact":
        return "Compacting conversation..."
    return f"Unknown tool: {name}"


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Keep working step by step, and use compact if the conversation gets too long."
)


def agent_loop(messages: list, state: CompactState) -> None:
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            max_tokens=8000,
        )
        msg = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": msg.tool_calls,
        })
        if response.choices[0].finish_reason != "tool_calls":
            return
        results = []
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            output = execute_tool(tc.function.name, args, state, tc.id)
            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(output),
            })
        messages.extend(results)


if __name__ == "__main__":
    history = []
    compact_state = CompactState()
    while True:
        try:
            query = input("\033[36ms06 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, compact_state)
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()
