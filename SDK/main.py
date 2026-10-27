"""CLI entry point: run a scenario through the Google GenAI SDK agent and write results.

    python -m SDK.main --scenario scenario.json --out results/sdk.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # allow `python SDK/main.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "SDK"

from google import genai  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402

from shared import scenario as sc  # noqa: E402

from .agent import RecipeAgent  # noqa: E402

SDK_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Google GenAI SDK recipe agent on a scenario.")
    p.add_argument("--scenario", default="scenario.json", help="Scenario JSON file.")
    p.add_argument("--out", default="results/sdk.json", help="Where to write the results JSON.")
    p.add_argument("--state-dir", help="Directory for persisted memory/history (default results/state).")
    p.add_argument("--model", help="Override the scenario's model.")
    p.add_argument("--thinking-budget", type=int, help="Thinking token budget (0 disables on Flash).")
    p.add_argument("--max-tool-calls", type=int, help="Max automatic function calls per message.")
    return p.parse_args(argv)


def run_scenario(scenario: dict[str, Any], args: argparse.Namespace,
                 client: genai.Client | None = None) -> dict[str, Any]:
    """Run every session; each session is a new Chat that loads persisted state."""
    paths = sc.prepare_state("sdk", scenario, args.state_dir)
    start = time.perf_counter()
    sessions = []
    previous: Path | None = None
    for session in scenario["sessions"]:
        history_path = paths["sessions"][session["name"]]
        # Long-term memory always carries over via the memory file; the chat transcript
        # itself carries over only when the session asks to resume.
        agent = RecipeAgent(
            model=args.model or scenario.get("model"),
            max_tool_calls=args.max_tool_calls or scenario.get("max_tool_iterations"),
            thinking_budget=args.thinking_budget if args.thinking_budget is not None
            else scenario.get("thinking_budget"),
            resume_history=previous if session.get("resume_previous") else None,
            client=client,
        )
        sessions.append(agent.run_session(session["name"], session["turns"]))
        agent.save_history(history_path)
        previous = history_path

    results: dict[str, Any] = {
        "implementation": "SDK",
        "scenario": scenario.get("name", "unnamed"),
        "model": agent.model,
        "sessions": sessions,
    }
    results["assertions"] = sc.evaluate_assertions(scenario, results, paths)
    results["metrics"] = {
        "usage": sc.sum_usage([s["usage"] for s in sessions]),
        "tools": sc.tool_summary(sessions),
        "duration_s": round(time.perf_counter() - start, 2),
        "turns": sum(len(s["turns"]) for s in sessions),
        "compactions": sum(s["compactions"] for s in sessions),
        "harness_line_counts": sc.line_counts(sorted(SDK_DIR.glob("*.py"))),
    }
    results["state_files"] = {"memory": str(paths["memory"]),
                              **{k: str(v) for k, v in paths["sessions"].items()}}
    return results


def main(argv: list[str] | None = None) -> int:
    sc.load_dotenv()
    args = parse_args(argv)
    try:
        scenario = sc.load_scenario(args.scenario)
    except (OSError, ValueError) as exc:
        print(f"Cannot load scenario: {exc}", file=sys.stderr)
        return 2
    try:
        results = run_scenario(scenario, args)
    except PermissionError as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr)
        return 1
    except genai_errors.APIError as exc:
        print(f"Gemini API error {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    out = sc.write_results(args.out, results)
    a, m = results["assertions"], results["metrics"]
    print(f"Wrote {out}: assertions {a['passed']}/{a['total']} passed, "
          f"cost ${m['usage']['estimated_cost_usd']}, {m['duration_s']}s")
    return 0 if a["failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
