#!/usr/bin/env python3
"""
Python Code Assistant Agent using the Anthropic API.

Tools:
  - run_python(code): executes Python code via subprocess, returns stdout/stderr
  - write_file(filename, content): saves content to a local file

The agent loops until Claude gives a final answer (no more tool calls).
"""

import json
import subprocess
import sys
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code in a subprocess and return its output. "
            "Use this to test code, compute results, or verify logic. "
            "Returns a dict with 'stdout', 'stderr', and 'returncode'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write text content to a local file. "
            "Use this to save generated code, results, or any text output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to the file to create or overwrite.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write into the file.",
                },
            },
            "required": ["filename", "content"],
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert Python programming assistant. You can execute Python code \
and write files to help users accomplish their tasks.

When given a task:
1. Think through the approach
2. Use run_python to test and execute code
3. Use write_file to save any files the user needs
4. Provide a clear final answer summarising what was done

Always show the code you run. If execution fails, debug and retry.
"""


def run_python(code: str) -> dict:
    """Execute Python code in a subprocess and return the result."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def write_file(filename: str, content: str) -> dict:
    """Write content to a local file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"status": "ok", "filename": str(path), "bytes_written": len(content.encode())}


def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call and return its result as a JSON string."""
    if name == "run_python":
        result = run_python(tool_input["code"])
    elif name == "write_file":
        result = write_file(tool_input["filename"], tool_input["content"])
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, indent=2)


def run_agent(task: str) -> str:
    """
    Run the agentic loop for a given task.
    Returns the final text response from the model.
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]

    print(f"\nTask: {task}\n{'─' * 60}")

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any text content streamed alongside tool calls
        text_parts = [b.text for b in response.content if b.type == "text"]
        if text_parts:
            print("\n".join(text_parts))

        # If Claude is done, return the final answer
        if response.stop_reason == "end_turn":
            return "\n".join(text_parts)

        # Otherwise process tool calls
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            # No tool calls and stop_reason is not end_turn — treat as done
            return "\n".join(text_parts)

        # Append the full assistant turn (preserves tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and collect results
        tool_results = []
        for block in tool_use_blocks:
            print(f"\n[Tool] {block.name}({json.dumps(block.input, indent=2)})")
            result_str = execute_tool(block.name, block.input)
            print(f"[Result]\n{result_str}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        # Feed all results back as a user turn
        messages.append({"role": "user", "content": tool_results})


def main():
    if len(sys.argv) > 1:
        # Task provided as a command-line argument
        task = " ".join(sys.argv[1:])
        run_agent(task)
    else:
        # Interactive REPL
        print("Python Code Assistant (powered by Claude)")
        print("Type your task and press Enter. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                task = input("Task> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not task:
                continue
            if task.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            run_agent(task)
            print()


if __name__ == "__main__":
    main()
