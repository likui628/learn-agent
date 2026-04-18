#!/usr/bin/env python3
# Harness: planning -- keep the current session plan outside the model's head.
"""
s03_todo_write.py - The Todo Write

+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> | Tools   |
| prompt |      |       |      | + todo  |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                          |
              +-----------+-----------+
              | TodoManager state     |
              | [ ] task A            |
              | [>] task B  <- doing  |
              | [x] task C            |
              +-----------------------+
                          |
              if rounds_since_todo >= 3:
                inject <reminder> into tool_result
"""

from dataclasses import dataclass, field
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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool for multi-step work.
Keep exactly one step in_progress when a task has multiple steps.
Refresh the plan as work advances. Prefer tools over prose."""


@dataclass
class PlanItem:
    content: str
    status: str = 'pending'
    active_form: str = ''


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        pass

    def render(self) -> str:
        pass


TODO = TodoManager()


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
    {
        "type": "function",
        "function": {
                "name": "todo",
                "description": "Rewrite the current session plan for multi-step work.",
                "parameters": {
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                            "active_form": {"type": "string", "description": "Optional present-continuous label."},
                        },
                        "required": ["content", "status"],
                    }}},
                    "required": ["items"],
                },
        },
    }
]

TOOL_HANDLERS = {
    "bash": lambda args: run_bash(args["command"]),
    "read_file": lambda args: read_file(args["path"], args.get("limit", 10000)),
    "write_file": lambda args: write_file(args["path"], args["content"]),
    "edit_file": lambda args: edit_file(args["path"], args["old_text"], args["new_text"]),
    "todo": lambda args: TODO.update(args["items"]),
}


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("activeForm", "")).strip()
            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form,
            ))
        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")
        self.state.items = normalized
        self.state.rounds_since_update = 0
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."
        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(
            1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()


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

# -- The core pattern: a while loop that calls tools until the model stops --


def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
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
        used_todo = False
        for tc in msg.tool_calls:
            handler = TOOL_HANDLERS.get(tc.function.name)
            print(
                f"\033[33m$ {tc.function.name}({tc.function.arguments})\033[0m")
            if handler:
                import json
                args = json.loads(tc.function.arguments)
                output = handler(args)
            else:
                output = f"Error: No handler for tool '{tc.function.name}'"

            print(f"> {tc.function.name}: {str(output)[:200]}")

            if tc.function.name == "todo":
                used_todo = True

            results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })
        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                results.append(reminder)
                results.insert(0, {"type": "text", "text": reminder})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
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
