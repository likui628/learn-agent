#!/usr/bin/env python3
"""
v2_todo_agent.py - Mini Claude Code: Structured Planning (~300 lines)

Core Philosophy: "Make Plans Visible"
=====================================
v1 works great for simple tasks. But ask it to "refactor auth, add tests,
update docs" and watch what happens. Without explicit planning, the model:
  - Jumps between tasks randomly
  - Forgets completed steps
  - Loses focus mid-way

The Problem - "Context Fade":
----------------------------
In v1, plans exist only in the model's "head":

    v1: "I'll do A, then B, then C"  (invisible)
        After 10 tool calls: "Wait, what was I doing?"

The Solution - TodoWrite Tool:
-----------------------------
v2 adds ONE new tool that fundamentally changes how the agent works:

    v2:
      [ ] Refactor auth module
      [>] Add unit tests         <- Currently working on this
      [ ] Update documentation

Now both YOU and the MODEL can see the plan. The model can:
  - Update status as it works
  - See what's done and what's next
  - Stay focused on one task at a time

Key Constraints (not arbitrary - these are guardrails):
------------------------------------------------------
    | Rule              | Why                              |
    |-------------------|----------------------------------|
    | Max 20 items      | Prevents infinite task lists     |
    | One in_progress   | Forces focus on one thing        |
    | Required fields   | Ensures structured output        |

The Deep Insight:
----------------
> "Structure constrains AND enables."

Todo constraints (max items, one in_progress) ENABLE (visible plan, tracked progress).

This pattern appears everywhere in agent design:
  - max_tokens constrains -> enables manageable responses
  - Tool schemas constrain -> enable structured calls
  - Todos constrain -> enable complex task completion

Good constraints aren't limitations. They're scaffolding.

Usage:
    python v2_todo_agent.py
"""

import asyncio
import os
import subprocess
from pathlib import Path

from pydantic_ai import Agent, AgentRunResultEvent, FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
WORKDIR = Path.cwd()

model = OpenRouterModel(
    'mistralai/devstral-2512:free',
    provider=OpenRouterProvider(
        api_key=os.getenv('OPENROUTER_API_KEY'),
        app_url='https://github.com/likui628/learn-agent',
        app_title='Learn Agent',
    )
)


# =============================================================================
# Agent Type Registry - The core of subagent mechanism
# =============================================================================

AGENT_TYPES = {
    # Explore: Read-only agent for searching and analyzing
    # Cannot modify files - safe for broad exploration
    "explore": {
        "description": "Read-only agent for exploring code, finding files, searching",
        "tools": ["bash", "read_file"],  # No write access
        "prompt": "You are an exploration agent. Search and analyze, but never modify files. Return a concise summary.",
    },

    # Code: Full-powered agent for implementation
    # Has all tools - use for actual coding work
    "code": {
        "description": "Full agent for implementing features and fixing bugs",
        "tools": "*",  # All tools
        "prompt": "You are a coding agent. Implement the requested changes efficiently.",
    },

    # Plan: Analysis agent for design work
    # Read-only, focused on producing plans and strategies
    "plan": {
        "description": "Planning agent for designing implementation strategies",
        "tools": ["bash", "read_file"],  # Read-only
        "prompt": "You are a planning agent. Analyze the codebase and output a numbered implementation plan. Do NOT make changes.",
    },
}


def get_agent_descriptions() -> str:
    """Generate agent type descriptions for the Task tool."""
    return "\n".join(
        f"- {name}: {cfg['description']}"
        for name, cfg in AGENT_TYPES.items()
    )

# =============================================================================
# TodoManager - The core addition in v2
# =============================================================================


class TodoManager:
    """
    Manages a structured task list with enforced constraints.

    Key Design Decisions:
    --------------------
    1. Max 20 items: Prevents the model from creating endless lists
    2. One in_progress: Forces focus - can only work on ONE thing at a time
    3. Required fields: Each item needs content, status, and activeForm

    The activeForm field deserves explanation:
    - It's the PRESENT TENSE form of what's happening
    - Shown when status is "in_progress"
    - Example: content="Add tests", activeForm="Adding unit tests..."

    This gives real-time visibility into what the agent is doing.
    """

    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        """
        Validate and update the todo list.

        The model sends a complete new list each time. We validate it,
        store it, and return a rendered view that the model will see.

        Validation Rules:
        - Each item must have: content, status, activeForm
        - Status must be: pending | in_progress | completed
        - Only ONE item can be in_progress at a time
        - Maximum 20 items allowed

        Returns:
            Rendered text view of the todo list
        """
        validated = []
        in_progress_count = 0

        for i, item in enumerate(items):
            # Extract and validate fields
            content = str(item.get("content", "")).strip()
            status = str(item.get("status", "pending")).lower()
            active_form = str(item.get("activeForm", "")).strip()

            # Validation checks
            if not content:
                raise ValueError(f"Item {i}: content required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {i}: invalid status '{status}'")
            if not active_form:
                raise ValueError(f"Item {i}: activeForm required")

            if status == "in_progress":
                in_progress_count += 1

            validated.append({
                "content": content,
                "status": status,
                "activeForm": active_form
            })

        # Enforce constraints
        if len(validated) > 20:
            raise ValueError("Max 20 todos allowed")
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        """
        Render the todo list as human-readable text.

        Format:
            [x] Completed task
            [>] In progress task <- Doing something...
            [ ] Pending task

            (2/3 completed)

        This rendered text is what the model sees as the tool result.
        It can then update the list based on its current state.
        """
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            if item["status"] == "completed":
                lines.append(f"[x] {item['content']}")
            elif item["status"] == "in_progress":
                lines.append(f"[>] {item['content']} <- {item['activeForm']}")
            else:
                lines.append(f"[ ] {item['content']}")

        completed = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({completed}/{len(self.items)} completed)")

        return "\n".join(lines)


# Global todo manager instance
TODO = TodoManager()

# =============================================================================
# System Prompt - The only "configuration" the model needs
# =============================================================================

SYSTEM_PROMPT = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> update todos -> report.

Workflow:
1. For any multi-step or non-trivial task, FIRST create a plan using todo_write
2. Execute tasks using tools (bash, read_file, write_file, edit_file)
3. After completing each step, update the task status using todo_write
4. When all tasks are completed, summarize the changes made

Rules for tool usage:
- Complex tasks MUST be planned with todo_write before starting
- You may use todo_read to check current progress
- Prefer tools over natural language explanations
- Only ONE task may be marked as in_progress at any time
- Do NOT work on tasks that are not in_progress
"""


def safe_path(p: str) -> Path:
    """
    Ensure path stays within workspace (security measure).

    Prevents the model from accessing files outside the project directory.
    Resolves relative paths and checks they don't escape via '../'.
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

# =============================================================================
# System Reminders - Soft prompts to encourage todo usage
# =============================================================================


# Shown at the start of conversation
INITIAL_REMINDER = "<reminder>Use TodoWrite for multi-step tasks.</reminder>"

# Shown if model hasn't updated todos in a while
NAG_REMINDER = "<reminder>10+ turns without todo update. Please update todos.</reminder>"

# Track how often we nudge for todo usage
rounds_without_todo = 0
first_message = True


# =============================================================================
# Tool Definitions & Implementations - 4 tools cover 90% of coding tasks
# =============================================================================

def bash(command: str) -> str:
    """
    Run a shell command. Use for: ls, find, grep, git, npm, python, etc.

    Args:
        command: The shell command to execute

    Notes:
        - Execute shell command with safety checks.
        - Security: Blocks obviously dangerous commands.
        - Timeout: 60 seconds to prevent hanging.
        - Output: Truncated to 50KB to prevent context overflow.
    """
    # Basic safety - block dangerous patterns
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
            timeout=60
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out (60s)"
    except Exception as e:
        return f"Error: {e}"


def read_file(path: str, limit: int = None) -> str:
    """
    Read file contents. Returns UTF-8 text.
    Args:
        path: Relative path to the file
        limit: Optional maximum number of lines to read

    Notes:
        Read file contents with optional line limit.

        For large files, use limit to read just the first N lines.
        Output truncated to 50KB to prevent context overflow.
    """
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()

        if limit and limit < len(lines):
            lines = lines[:limit]
            lines.append(f"... ({len(text.splitlines()) - limit} more lines)")

        return "\n".join(lines)[:50000]

    except Exception as e:
        return f"Error: {e}"


def write_file(path: str, content: str) -> str:
    """
    Write content to a file. Creates parent directories if needed.
    Args:
        path: Relative path to the file
        content: Text content to write

    Notes:
        Write content to file, creating parent directories if needed.

        This is for complete file creation/overwrite.
        For partial edits, use edit_file instead.
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"

    except Exception as e:
        return f"Error: {e}"


def edit_file(path: str, old_text: str, new_text: str) -> str:
    """
    Replace exact text in a file. Use for surgical edits.
    Args:
        path: Relative path to the file
        old_text: Exact text to find (must match precisely)
        new_text: Replacement text

    Notes:
        Replace exact text in a file (surgical edit).

        Uses exact string matching - the old_text must appear verbatim.
        Only replaces the first occurrence to prevent accidental mass changes.
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()

        if old_text not in content:
            return f"Error: Text not found in {path}"

        # Replace only first occurrence for safety
        new_content = content.replace(old_text, new_text, 1)
        fp.write_text(new_content)
        return f"Edited {path}"

    except Exception as e:
        return f"Error: {e}"


def todo_write(items: list) -> str:
    """
    Update the task list. Use to plan and track progress.

    Args:
        items: List of todo items with fields:
            - content: Task description
            - status: pending | in_progress | completed
            - activeForm: Present tense action, e.g. 'Reading files'

    Notes:
        Update the todo list.

        The model sends a complete new list (not a diff).
        We validate it and return the rendered view.
    """
    try:
        return TODO.update(items)
    except Exception as e:
        return f"Error: {e}"

async def task_tool(ctx, prompt: str, agent_type: str = "explore") -> str:
    """
    Spawn a subagent to handle a focused task.

    Args:
        prompt: Task description for the subagent
        agent_type: Type of subagent to create (explore, code, plan)
    """

    return await run_task(ctx, prompt, agent_type)

# =============================================================================
# Subagents - Specialized agents for focused tasks
# =============================================================================

TOOL_REGISTRY = {
    "bash": bash,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "todo_write": todo_write,
}


def create_agent(agent_type: str = "main") -> Agent:
    config = AGENT_TYPES.get(agent_type, AGENT_TYPES["code"])
    if agent_type == "main":
        system_prompt = f"""You are a coding agent at {WORKDIR}.

Loop: plan -> act with tools -> report.

You can spawn subagents for complex subtasks:
{get_agent_descriptions()}

Rules:
- Use Task tool for subtasks that need focused exploration or implementation
- Use TodoWrite to track multi-step work
- Prefer tools over prose. Act, don't just explain.
- After finishing, summarize what changed."""
    else:
        system_prompt = f"""You are a {agent_type} agent at {WORKDIR}.
{config['prompt']}
Complete the task and return a clear, concise summary."""

    if (config['tools'] == "*"):
        tools = [bash, read_file, write_file, edit_file, todo_write]
    else:
        tools = [TOOL_REGISTRY[name] for name in config['tools']]

    if agent_type == "main":
        async def task_tool(ctx, prompt: str, agent_type: str = "explore") -> str:
            """
            Spawn a subagent to handle a focused task.

            Args:
                prompt: Task description for the subagent
                agent_type: Type of subagent to create (explore, code, plan)
            """

            return await run_task(ctx, prompt, agent_type)

        tools.append(task_tool)

    agent = Agent(
        model=model,
        tools=tools,
        output_type=str,
        system_prompt=system_prompt,
    )
    return agent

# =============================================================================
# Subagent Tool - Spawns isolated agents for complex subtasks
# =============================================================================


async def run_task(ctx, prompt: str, agent_type: str = "explore") -> str:
    """
    Spawn a subagent to handle a focused task.

    Args:
        prompt: Task description for the subagent
        agent_type: Type of subagent to create (explore, code, plan)

    Notes:
        Spawns a subagent in isolation to handle complex subtasks.

        The subagent runs with its own context and tools.
        Only the final summary is returned to the main agent.
    """
    if agent_type not in AGENT_TYPES:
        return f"Error: Unknown agent type '{agent_type}'"

    subagent = create_agent(agent_type)
    try:
        print(f"Spawning subagent ({agent_type}) for task:\n{prompt}\n")
        response = await subagent.run(prompt)
        return response.output
    except Exception as e:
        print(f"Subagent error: {e}")
        return f"Error: Subagent failed - {e}"


# =============================================================================
# Main REPL
# =============================================================================
async def main():
    print(f"Mini Claude Code v3 (with Subagents) - {WORKDIR}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            break

        try:
            main_agent = create_agent("main")

            response = await main_agent.run(user_input)
            print(response.output)
            print()  # Blank line between turns
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

if __name__ == "__main__":
    asyncio.run(main())
