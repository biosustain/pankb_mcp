"""Eval runner using Phoenix's native Datasets + Experiments workflow.

Loads questions from golden_set.yaml, uploads them as a Phoenix Dataset, then
runs each through MCPClient.chat() (the real production agent — no mock) inside
a Phoenix Experiment. Scores tool-selection with a deterministic evaluator.

Phoenix UI replaces the old markdown report:
  - Dataset:    http://<phoenix>/datasets       (versioned, upserts on re-upload)
  - Experiment: http://<phoenix>/projects/pankb-eval/experiments

Exit codes:
  0  all examples passed
  1  some examples failed scoring
  2  Phoenix unreachable (fail loud; opposite of tracing.py's lenient fallback)

Usage:
    cd evals
    MCP_SERVER_URL=http://localhost:8000/mcp \\
    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 \\
    uv run python run_evals.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fastmcp.client.auth import BearerAuth
from openai import AsyncOpenAI

# Make the ai_client app importable so we reuse the real production agent.
AI_CLIENT_APP = Path(__file__).parent.parent / "ai_client" / "app"
sys.path.insert(0, str(AI_CLIENT_APP))

# Force eval traces into a separate Phoenix project so they don't mix with
# user traffic. Must be set BEFORE init_tracing() / MCPClient import.
os.environ.setdefault("PHOENIX_PROJECT_NAME", "pankb-eval")

from mcp_client_stream import MCPClient  # noqa: E402
from phoenix.client import AsyncClient as PhoenixAsyncClient  # noqa: E402
from phoenix.client.experiments import async_run_experiment  # noqa: E402
from prompts import get_prompt  # noqa: E402
from tracing import init_tracing  # noqa: E402

load_dotenv(Path(__file__).parent.parent / "ai_client" / ".env")

DATASET_NAME = "pankb-golden"


def score_case(
    expected: list[str],
    actual: list[str],
    forbidden: list[str],
) -> tuple[bool, str | None]:
    """Pass iff every expected tool was called and no forbidden tool was.
    Empty expected = negative case: actual must also be empty."""
    forbidden_hit = [t for t in actual if t in forbidden]
    if forbidden_hit:
        return False, f"called forbidden tool(s): {forbidden_hit}"
    if not expected:
        if actual:
            return False, f"expected no tool calls but got {actual}"
        return True, None
    missing = [t for t in expected if t not in actual]
    if missing:
        return False, f"missing expected tool(s): {missing}; actual={actual}"
    return True, None


def extract_tool_sequence(messages: list[dict]) -> list[str]:
    """Pull the ordered list of tool names invoked from client.messages."""
    return [
        m["name"]
        for m in messages
        if m.get("type") == "function_call" and m.get("name")
    ]


def make_task(client: MCPClient):
    """Build an async task closure over a connected MCPClient. Phoenix binds
    the `example` arg by name and the function returns whatever dict it likes."""

    async def task(example) -> dict:
        client.clear_history()
        async for _event in client.chat(example.input["question"]):
            pass
        return {"actual_tools": extract_tool_sequence(client.messages)}

    return task


def _decode_tool_list(value) -> list[str]:
    """Phoenix Dataset's CSV-backed metadata stringifies lists. Be lenient:
    accept real lists, JSON strings, or fall back to empty."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


def tool_selection_correct(output, metadata) -> dict:
    """Deterministic evaluator. Phoenix auto-binds output (from task) and
    metadata (from dataset). Returns a Score-shaped dict."""
    expected = _decode_tool_list(metadata.get("expected_tools"))
    forbidden = _decode_tool_list(metadata.get("forbidden_tools"))
    passed, reason = score_case(expected, output["actual_tools"], forbidden)
    return {
        "score": 1.0 if passed else 0.0,
        "label": "pass" if passed else "fail",
        "explanation": reason or "all expected tools called, no forbidden tools",
    }


def build_dataframe(cases: list[dict]) -> pd.DataFrame:
    # JSON-encode the tool lists so Phoenix doesn't stringify them with
    # Python's repr (which makes them un-parseable on the evaluator side).
    return pd.DataFrame(
        [
            {
                "id": c["id"],
                "question": c["question"],
                "category": c["category"],
                "expected_tools": json.dumps(c.get("expected_tools") or []),
                "forbidden_tools": json.dumps(c.get("forbidden_tools") or []),
            }
            for c in cases
        ]
    )


async def main() -> int:
    init_tracing()

    mcp_url = os.getenv("MCP_SERVER_URL")
    mcp_key = os.getenv("MCP_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    if not (mcp_url and mcp_key and openai_key and phoenix_endpoint):
        print(
            "ERROR: MCP_SERVER_URL, MCP_API_KEY, OPENAI_API_KEY, "
            "PHOENIX_COLLECTOR_ENDPOINT must all be set",
            file=sys.stderr,
        )
        return 2

    prompt_version, system_prompt = get_prompt()
    print(f"Using prompt version: {prompt_version}")

    # Fail loud if Phoenix is unreachable — eval is part of the mlops loop,
    # we should not silently downgrade like the live app does.
    px = PhoenixAsyncClient(base_url=phoenix_endpoint)
    try:
        await px.datasets.list()
    except Exception as e:
        print(
            f"ERROR: Phoenix unreachable at {phoenix_endpoint}: {e}",
            file=sys.stderr,
        )
        return 2

    yaml_path = Path(__file__).parent / "golden_set.yaml"
    cases = yaml.safe_load(yaml_path.read_text())
    print(f"Loaded {len(cases)} cases from {yaml_path.name}")

    df = build_dataframe(cases)
    dataset = await px.datasets.create_dataset(
        name=DATASET_NAME,
        dataframe=df,
        input_keys=["question"],
        output_keys=[],
        metadata_keys=["id", "category", "expected_tools", "forbidden_tools"],
    )
    print(
        f"Dataset {DATASET_NAME!r} ready: {dataset.example_count} examples, "
        f"version={dataset.version_id}"
    )

    client = MCPClient(
        mcp_server_url=mcp_url,
        openai_client=AsyncOpenAI(api_key=openai_key),
        model=model,
        auth=BearerAuth(token=mcp_key),
        system_prompt=system_prompt,
        prompt_version=prompt_version,
    )
    await client.connect()
    print(f"Connected to MCP server, {len(client.tools_cache)} tools available")

    # concurrency=1 because the task closes over a single MCPClient instance
    # whose `.messages` list is mutated per-call — running cases in parallel
    # would interleave their function_call/output items and crash.
    experiment = await async_run_experiment(
        dataset=dataset,
        task=make_task(client),
        evaluators=[tool_selection_correct],
        experiment_name=f"pankb-eval-{prompt_version}",
        experiment_metadata={
            "prompt_version": prompt_version,
            "model": model,
        },
        client=px,
        concurrency=1,
    )

    # RanExperiment is a TypedDict — use index access, not attribute.
    # Each ExperimentEvaluationRun has .result (EvaluationResult) with .score.
    scores: list[float] = []
    for run in experiment.get("evaluation_runs") or []:
        result = run.result
        if result is not None and result.get("score") is not None:
            scores.append(float(result["score"]))

    if not scores:
        print(
            "WARNING: could not extract scores from experiment object; "
            "see Phoenix UI for results",
            file=sys.stderr,
        )
        return 1

    passed = sum(1 for s in scores if s == 1.0)
    total = len(scores)
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
