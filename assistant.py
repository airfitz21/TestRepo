#!/usr/bin/env python3
"""
Python Code Assistant Agent using the Anthropic API.

Tools:
  - run_python(code)                          execute Python code via subprocess
  - write_file(filename, content)             save a file locally
  - read_file(filename)                       read an existing file
  - search_code(pattern, path, *, glob)       grep/regex search within files
  - list_directory(path, *, recursive)        browse project structure
  - install_package(package)                  pip-install a dependency

Loop behaviour:
  - After each tool result the model decides whether to call another tool
    or give a final answer.
  - If a tool returns an error (non-zero returncode / exception) the result
    is flagged with is_error=True so the model sees it and self-corrects.
  - Hard cap: MAX_RETRIES consecutive error-only turns before giving up.
"""

import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 5          # max consecutive turns that end in tool errors


# ---------------------------------------------------------------------------
# Tool definitions (JSON schema sent to the API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code in a subprocess and return its output. "
            "Returns a dict with 'stdout', 'stderr', and 'returncode'. "
            "returncode 0 means success; non-zero means failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
            "required": ["code"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a local file with the given text content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Path to write."},
                "content":  {"type": "string", "description": "Text to write."},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read an existing file and return its content as a string. "
            "Useful for inspecting source code, configs, or previous output."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Path of the file to read."}
            },
            "required": ["filename"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search files for a regex pattern (like grep -rn). "
            "Returns matching lines with file paths and line numbers. "
            "Use 'path' to restrict the search root and 'glob' to filter by filename pattern."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in. Defaults to '.' (cwd).",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Filename glob filter, e.g. '*.py'. Defaults to all files.",
                    "default": "*",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List files and directories at a given path. "
            "Set recursive=true to walk the entire tree. "
            "Returns an annotated tree showing files (F) and directories (D)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Defaults to '.' (cwd).",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to recurse into subdirectories.",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    {
        "name": "install_package",
        "description": (
            "Install one or more Python packages with pip. "
            "Pass a single package name or a requirements-style spec, e.g. 'requests>=2.28'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package name / spec to install.",
                }
            },
            "required": ["package"],
        },
    },
]

SYSTEM_PROMPT = """\
You are an expert Python programming assistant with access to six tools:
  run_python, write_file, read_file, search_code, list_directory, install_package.

Workflow:
1. Understand the task; explore files with list_directory / read_file if needed.
2. Search existing code with search_code before writing from scratch.
3. Install missing packages with install_package, then run code with run_python.
4. If a tool returns an error (non-zero returncode or an 'error' key), analyse the
   problem, fix it, and retry — do NOT give up after one failure.
5. Save any deliverables with write_file.
6. Finish with a concise summary of what was done.
"""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def run_python(code: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    return {
        "stdout":     result.stdout,
        "stderr":     result.stderr,
        "returncode": result.returncode,
    }


def write_file(filename: str, content: str) -> dict:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"status": "ok", "filename": str(path), "bytes_written": len(content.encode())}


def read_file(filename: str) -> dict:
    path = Path(filename)
    if not path.exists():
        return {"error": f"File not found: {filename}"}
    if not path.is_file():
        return {"error": f"Not a file: {filename}"}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        return {"filename": str(path), "content": content, "lines": content.count("\n") + 1}
    except OSError as exc:
        return {"error": str(exc)}


def search_code(pattern: str, path: str = ".", glob: str = "*") -> dict:
    import re
    root = Path(path)
    if not root.exists():
        return {"error": f"Path not found: {path}"}

    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}"}

    matches = []
    targets = [root] if root.is_file() else root.rglob("*")
    for p in targets:
        if not p.is_file():
            continue
        if not fnmatch.fnmatch(p.name, glob):
            continue
        try:
            for lineno, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    matches.append({"file": str(p), "line": lineno, "text": line.rstrip()})
        except OSError:
            continue

    return {"pattern": pattern, "match_count": len(matches), "matches": matches[:200]}


def list_directory(path: str = ".", recursive: bool = False) -> dict:
    root = Path(path)
    if not root.exists():
        return {"error": f"Path not found: {path}"}

    entries = []
    if recursive:
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            entries.append({"type": "F" if p.is_file() else "D", "path": str(rel)})
    else:
        for p in sorted(root.iterdir()):
            entries.append({"type": "F" if p.is_file() else "D", "path": p.name})

    return {"path": str(root.resolve()), "entry_count": len(entries), "entries": entries}


def install_package(package: str) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package],
        capture_output=True, text=True, timeout=120,
    )
    return {
        "stdout":     result.stdout[-2000:],   # trim verbose pip output
        "stderr":     result.stderr[-500:],
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_FNS = {
    "run_python":      lambda i: run_python(i["code"]),
    "write_file":      lambda i: write_file(i["filename"], i["content"]),
    "read_file":       lambda i: read_file(i["filename"]),
    "search_code":     lambda i: search_code(i["pattern"], i.get("path", "."), i.get("glob", "*")),
    "list_directory":  lambda i: list_directory(i.get("path", "."), i.get("recursive", False)),
    "install_package": lambda i: install_package(i["package"]),
}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """
    Run a tool and return (result_json, is_error).

    is_error is True when the tool signals failure so the model knows to retry.
    """
    fn = _TOOL_FNS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"}), True

    try:
        result = fn(tool_input)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Tool timed out."}), True
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"Tool raised exception: {exc}"}), True

    # Treat non-zero returncode or a top-level 'error' key as an error
    is_error = bool(result.get("error") or result.get("returncode", 0) != 0)
    return json.dumps(result, indent=2), is_error


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def _all_errors(tool_results: list[dict]) -> bool:
    """Return True if every result in this turn was flagged as an error."""
    return all(r.get("is_error") for r in tool_results)


def run_agent(task: str) -> str:
    """
    Run the agentic loop for a given task.
    Auto-retries when tools fail; gives up after MAX_RETRIES consecutive error turns.
    Returns the final text response from the model.
    """
    client = anthropic.Anthropic()
    messages: list[dict] = [{"role": "user", "content": task}]

    print(f"\nTask: {task}\n{'─' * 60}")

    consecutive_errors = 0

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Print any inline text
        text_parts = [b.text for b in response.content if b.type == "text"]
        if text_parts:
            print("\n".join(text_parts))

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            return "\n".join(text_parts)

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            return "\n".join(text_parts)

        # Preserve full assistant turn (tool_use blocks must stay in history)
        messages.append({"role": "assistant", "content": response.content})

        # Execute tools
        tool_results = []
        for block in tool_use_blocks:
            print(f"\n[Tool] {block.name}({json.dumps(block.input, indent=2)})")
            result_str, is_error = execute_tool(block.name, block.input)
            label = "[Error]" if is_error else "[Result]"
            print(f"{label}\n{result_str}")
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_str,
                "is_error":    is_error,
            })

        messages.append({"role": "user", "content": tool_results})

        # Track consecutive all-error turns for the retry cap
        if _all_errors(tool_results):
            consecutive_errors += 1
            if consecutive_errors >= MAX_RETRIES:
                msg = (
                    f"Giving up after {MAX_RETRIES} consecutive error turns. "
                    "Check the errors above for details."
                )
                print(f"\n[Agent] {msg}")
                return msg
        else:
            consecutive_errors = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        run_agent(task)
    else:
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
