"""Offline tests: exercise shared/, the API harness, the scenario runner and the
Google GenAI SDK agent against a scripted mock model. No GEMINI_API_KEY or
network: a real genai.Client is used with its per-call network method replaced,
so the SDK's own Chat + automatic function calling loop runs for real.

    python -m unittest discover -s tests
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with contextlib.redirect_stdout(io.StringIO()):
    from google import genai
    from google.genai import types

    import shared
    from API import main as api_main
    from API.agent import RecipeAgent
    from API.harness import compaction, context, loop, tools
    from API.harness.caching import PromptCache
    from API.harness.history import History
    from API.harness.tokens import TokenTracker
    from SDK import main as sdk_main
    from shared import scenario as sc


# --------------------------------------------------------------------------- #
# Mock Gemini model
# --------------------------------------------------------------------------- #

def make_response(parts: list[dict[str, Any]], *, prompt: int = 900, cached: int = 0,
                  thoughts: int = 20, finish: str = "STOP") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(**p) for p in parts]),
                                    finish_reason=finish)] if parts or finish != "SAFETY" else
        [types.Candidate(finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt, cached_content_token_count=cached or None,
            candidates_token_count=40, thoughts_token_count=thoughts),
    )


def as_contents(contents: Any) -> list[types.Content]:
    items = contents if isinstance(contents, list) else [contents]
    out = []
    for c in items:
        if isinstance(c, types.Content):
            out.append(c)
        elif isinstance(c, dict):
            out.append(types.Content.model_validate(c))
        else:
            out.append(types.Content(role="user", parts=[types.Part(text=str(c))]))
    return out


def validate_request(contents: list[types.Content], config: types.GenerateContentConfig) -> None:
    """Fail loudly on request shapes the real API would reject."""
    assert contents and contents[0].role == "user", "first content must be user"
    for prev, cur in zip(contents, contents[1:]):
        assert prev.role != cur.role, "roles must alternate"
        calls = [p.function_call for p in prev.parts or [] if p.function_call]
        if prev.role == "model" and calls:
            responses = [p.function_response for p in cur.parts or [] if p.function_response]
            assert [c.name for c in calls] == [r.name for r in responses], "function calls not all answered"
    for c in contents:
        assert c.parts, "empty content"


def text_of(content: types.Content) -> str:
    return "\n".join(p.text for p in content.parts or [] if p.text)


class ScriptedModel:
    """Decides the mock model's next action from the conversation, mimicking a
    model that follows the system prompt for the scenario's turns."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def __call__(self, contents: list[types.Content], config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        n = len(self.requests)
        last = contents[-1]
        cached = 600 if len(contents) > 1 else 0
        mode = config.tool_config.function_calling_config.mode if config.tool_config else None
        if mode and mode.value == "NONE":
            return make_response([{"text": "- user summary"}])
        if any(p.function_response for p in last.parts or []):
            payloads = [json.dumps(p.function_response.response) for p in last.parts if p.function_response]
            return make_response([{"text": "Done: " + " | ".join(payloads)[:400]}], cached=cached)
        text = text_of(last).lower()

        def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
            return {"function_call": {"name": name, "args": args}, "thought_signature": f"sig{n}".encode()}

        if "allergic" in text:
            parts = [call("save_user_preference_or_recipe", {"type": "preference", "key": "diet", "value": "vegetarian"}),
                     call("save_user_preference_or_recipe", {"type": "preference", "key": "allergies", "value": "peanuts"})]
        elif "dinner idea" in text:
            parts = [call("search_recipes", {"query": "quick dinner", "tag": "vegetarian"})]
        elif "calories" in text:
            parts = [call("fetch_external_nutrition_info", {"ingredient_or_dish": "pasta"})]
        elif "save it" in text:
            parts = [call("save_user_preference_or_recipe", {"type": "recipe", "recipe": shared.get_recipe("r5")})]
        elif "what do you remember" in text:
            mem = text.split("<user_memory>")[1].split("</user_memory>")[0] if "<user_memory>" in text else "nothing"
            return make_response([{"text": f"I remember: {mem.strip()}"}], cached=cached)
        elif "chicken stir fry" in text:
            reply = ("Chicken Stir Fry contains chicken, which conflicts with your vegetarian diet. "
                     "Try the Crispy Tofu Bowl instead?" if "vegetarian" in text else "Here is the recipe.")
            return make_response([{"text": reply}], cached=cached)
        elif "chickpeas" in text:
            parts = [call("fetch_external_nutrition_info", {"ingredient_or_dish": "chickpeas"})]
        else:
            return make_response([{"text": "OK."}])
        return make_response(parts, cached=cached)


def mock_client(model: Callable[[list[types.Content], types.GenerateContentConfig], Any] | None = None
                ) -> tuple[genai.Client, Any]:
    """A real genai.Client whose network call is replaced by `model`."""
    model = model or ScriptedModel()
    client = genai.Client(api_key="offline-test-key")
    requests: list[dict[str, Any]] = []

    def fake_call(*, model: str, contents: Any, config: Any = None) -> types.GenerateContentResponse:
        contents = as_contents(contents)
        cfg = config if isinstance(config, types.GenerateContentConfig) else types.GenerateContentConfig(**(config or {}))
        validate_request(contents, cfg)
        requests.append({"contents": [c.model_dump(mode="json", exclude_none=True) for c in contents],
                         "config": cfg})
        if isinstance(fake_model, ScriptedModel):
            fake_model.requests.append(requests[-1])
        return fake_model(contents, cfg)

    fake_model = model
    client.models._generate_content_with_continuation = fake_call
    client._mock_requests = requests
    return client, requests


# --------------------------------------------------------------------------- #
# Base fixture: isolated memory file, quiet stdout, no API key
# --------------------------------------------------------------------------- #

class OfflineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
        env["RECIPE_MEMORY_PATH"] = str(self.tmpdir / "memory.json")
        self._env = mock.patch.dict(os.environ, env, clear=True)
        self._env.start()
        self._out = contextlib.redirect_stdout(io.StringIO())
        self._out.__enter__()

    def tearDown(self) -> None:
        self._out.__exit__(None, None, None)
        self._env.stop()
        self.tmp.cleanup()


# --------------------------------------------------------------------------- #
# shared/
# --------------------------------------------------------------------------- #

class SharedDomainTests(OfflineTestCase):
    def test_search_ranks_and_filters_by_tag(self) -> None:
        results = shared.search_recipes("tomato", tag="vegetarian")
        self.assertTrue(results)
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all("vegetarian" in r["tags"] for r in results))
        self.assertEqual(results[0]["title"], "Tomato Basil Pasta")

    def test_search_no_match_returns_empty(self) -> None:
        self.assertEqual(shared.search_recipes("xyzzy"), [])

    def test_save_preference_recipe_note_persist(self) -> None:
        self.assertTrue(shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegan"})["ok"])
        self.assertTrue(shared.save_user_preference_or_recipe({"recipe": {"title": "Toast"}})["ok"])
        self.assertTrue(shared.save_user_preference_or_recipe({"text": "likes spicy"})["ok"])
        mem = shared.load_memory()
        self.assertEqual(mem["preferences"]["diet"], "vegan")
        self.assertEqual(mem["saved_recipes"][0]["title"], "Toast")
        self.assertEqual(mem["notes"][0]["text"], "likes spicy")
        self.assertEqual(shared.search_recipes("toast")[0]["title"], "Toast")

    def test_save_invalid_returns_error(self) -> None:
        self.assertFalse(shared.save_user_preference_or_recipe({"foo": 1})["ok"])
        self.assertFalse(shared.save_user_preference_or_recipe({"type": "preference", "key": "x"})["ok"])

    def test_nutrition_exact_recipe_partial_and_missing(self) -> None:
        self.assertEqual(shared.fetch_external_nutrition_info("Tofu")["calories"], 76)
        dish = shared.fetch_external_nutrition_info("Baked Salmon with Spinach")
        self.assertIn("salmon", dish["ingredients"])
        self.assertEqual(shared.fetch_external_nutrition_info("grilled salmon")["name"], "salmon")
        self.assertFalse(shared.fetch_external_nutrition_info("pizza")["found"])

    def test_corrupt_memory_file_loads_empty(self) -> None:
        Path(os.environ["RECIPE_MEMORY_PATH"]).write_text("{not json")
        self.assertEqual(shared.load_memory()["preferences"], {})


# --------------------------------------------------------------------------- #
# API harness components
# --------------------------------------------------------------------------- #

class HarnessComponentTests(OfflineTestCase):
    def test_declarations_match_handlers(self) -> None:
        decls = tools.get_tools()[0].function_declarations
        self.assertEqual([d.name for d in decls], list(tools.HANDLERS))

    def test_dispatch_success_and_errors(self) -> None:
        ok, err = tools.execute_tool_call({"id": "1", "name": "search_recipes", "args": {"query": "hummus"}})
        self.assertFalse(err)
        self.assertEqual(ok["function_response"]["id"], "1")
        self.assertIn("Classic Hummus", json.dumps(ok))
        unknown, err = tools.execute_tool_call({"name": "nope", "args": {}})
        self.assertTrue(err)
        self.assertIn("error", unknown["function_response"]["response"])
        _, err = tools.execute_tool_call({"name": "save_user_preference_or_recipe", "args": {"type": "bogus"}})
        self.assertTrue(err)
        with mock.patch.dict(tools.HANDLERS, {"search_recipes": lambda _: 1 / 0}):
            crash, err = tools.execute_tool_call({"name": "search_recipes", "args": {"query": ""}})
        self.assertTrue(err)
        self.assertIn("ZeroDivisionError", crash["function_response"]["response"]["error"])

    def test_history_merge_roundtrip_and_signatures(self) -> None:
        h = History()
        h.append_user("hi")
        h.append_user([{"text": "again"}])
        self.assertEqual(len(h), 1)
        h.append_model([{"function_call": {"name": "f", "args": {}}, "thought_signature": "AP9zaWc="}])
        path = self.tmpdir / "h.json"
        h.save(path)
        loaded = History.load(path)
        self.assertEqual(loaded.messages, h.messages)
        self.assertEqual(loaded.as_contents()[1].parts[0].thought_signature, b"\x00\xffsig")
        self.assertEqual(len(History.load(self.tmpdir / "missing.json")), 0)

    def test_prefix_config_is_stable(self) -> None:
        client, _ = mock_client()
        cache = PromptCache(client, "gemini-2.5-flash", context.SYSTEM_PROMPT, tools.get_tools())
        c1 = context.build_config(cache, max_output_tokens=100, thinking_budget=None)
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegan"})
        c2 = context.build_config(cache, max_output_tokens=100, thinking_budget=None)
        self.assertEqual(c1.system_instruction, c2.system_instruction)
        self.assertEqual(c1.tools, c2.tools)
        self.assertTrue(c1.automatic_function_calling.disable)
        self.assertIsNone(c1.cached_content)

    def test_explicit_cache_falls_back_when_prefix_small(self) -> None:
        client, _ = mock_client()
        with mock.patch.object(client.models, "count_tokens", return_value=types.CountTokensResponse(total_tokens=300)):
            cache = PromptCache(client, "m", "sys", [], explicit=True)
        self.assertIsNone(cache.cache_name)
        self.assertIn("system_instruction", cache.prefix_config())

    def test_explicit_cache_used_when_created(self) -> None:
        client, _ = mock_client()
        with mock.patch.object(client.models, "count_tokens", return_value=types.CountTokensResponse(total_tokens=5000)), \
             mock.patch.object(client.caches, "create", return_value=types.CachedContent(name="cachedContents/abc")):
            cache = PromptCache(client, "m", "sys", [], explicit=True)
        self.assertEqual(cache.prefix_config(), {"cached_content": "cachedContents/abc"})

    def test_token_accounting(self) -> None:
        t = TokenTracker()
        t.record(types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1000, cached_content_token_count=600, candidates_token_count=50,
            thoughts_token_count=30))
        self.assertEqual((t.input_tokens, t.cache_read_tokens, t.output_tokens, t.thinking_tokens),
                         (400, 600, 80, 30))
        self.assertEqual(t.cost_usd(), 0.0)
        self.assertTrue(t.over_budget(999))

    def test_memory_block_in_user_turn(self) -> None:
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "vegetarian"})
        parts = context.build_user_turn("dinner?", summary="earlier stuff")
        self.assertIn("<conversation_summary>", parts[0]["text"])
        self.assertIn("diet: vegetarian", parts[1]["text"])
        self.assertEqual(parts[-1]["text"], "dinner?")


class LoopTests(OfflineTestCase):
    def test_tool_round_then_answer_single_response_message(self) -> None:
        client, requests = mock_client()
        agent = RecipeAgent(client=client)
        result = agent.chat("I'm vegetarian and allergic to peanuts")
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertEqual(len(requests), 2)
        responses = agent.history.messages[2]
        self.assertEqual(responses["role"], "user")
        self.assertEqual(sum("function_response" in p for p in responses["parts"]), 2)
        # Thought signatures are replayed unchanged on the next request.
        sig = requests[1]["contents"][1]["parts"][0]["thought_signature"]
        self.assertEqual(sig, agent.history.messages[1]["parts"][0]["thought_signature"])

    def test_history_is_append_only_across_requests(self) -> None:
        client, requests = mock_client()
        agent = RecipeAgent(client=client)
        agent.chat("I'm vegetarian and allergic to peanuts")
        agent.chat("Any quick dinner idea?")
        for prev, cur in zip(requests, requests[1:]):
            n = len(prev["contents"])
            self.assertEqual(cur["contents"][:n], prev["contents"])
            self.assertEqual(cur["config"].system_instruction, prev["config"].system_instruction)

    def test_max_iterations_closes_open_calls(self) -> None:
        def always_call(contents: Any, config: Any) -> types.GenerateContentResponse:
            return make_response([{"function_call": {"name": "search_recipes", "args": {"query": "x"}}}])

        client, requests = mock_client(always_call)
        agent = RecipeAgent(client=client, max_tool_iterations=3)
        result = agent.chat("loop forever")
        self.assertEqual(result["stop_reason"], "max_iterations")
        self.assertEqual(len(requests), 3)
        self.assertEqual(agent.history.messages[-1]["role"], "user")
        agent.chat("next")  # must still produce a valid request (validated by the mock)

    def test_safety_block_returns_message(self) -> None:
        client, _ = mock_client(lambda c, cfg: make_response([], finish="SAFETY"))
        result = RecipeAgent(client=client).chat("hi")
        self.assertEqual(result["stop_reason"], "refusal")
        self.assertTrue(result["text"])


class CompactionTests(OfflineTestCase):
    def test_compacts_over_threshold_and_injects_summary(self) -> None:
        client, requests = mock_client()
        agent = RecipeAgent(client=client, compaction_threshold_tokens=500)
        agent.chat("Any quick dinner idea?")
        result = agent.chat("chickpeas?")
        self.assertTrue(result["compacted"])
        summary_req = requests[2]
        self.assertEqual(summary_req["config"].tool_config.function_calling_config.mode.value, "NONE")
        self.assertEqual(summary_req["contents"][-1]["parts"][-1]["text"], compaction.SUMMARY_PROMPT)
        after = requests[3]["contents"]
        self.assertEqual(after[0]["role"], "user")
        self.assertIn("<conversation_summary>", after[0]["parts"][0]["text"])
        self.assertEqual(agent.history.compactions, 1)

    def test_fallback_summary_when_summarizer_fails(self) -> None:
        h = History()
        h.append_user("I love tofu")
        h.append_model([{"text": "Noted"}])

        def boom(_: Any) -> str:
            raise RuntimeError("offline")

        self.assertIn("I love tofu", compaction.compact(h, boom))
        self.assertEqual(len(h), 0)

    def test_no_compaction_under_threshold(self) -> None:
        h = History([{"role": "user", "parts": [{"text": "a"}]}, {"role": "model", "parts": [{"text": "b"}]}])
        t = TokenTracker()
        t.record({"prompt_token_count": 100})
        self.assertFalse(compaction.needs_compaction(h, t, 1000))


# --------------------------------------------------------------------------- #
# Full scenario, offline, both implementations
# --------------------------------------------------------------------------- #

def api_args(state_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(state_dir=str(state_dir), model=None, thinking_budget=None,
                              max_tool_iterations=None, compaction_threshold_tokens=None, explicit_cache=None)


def sdk_args(state_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(state_dir=str(state_dir), model=None, thinking_budget=None, max_tool_calls=None)


class ScenarioTests(OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.scenario = sc.load_scenario(ROOT / "scenario.json")

    def check_results(self, results: dict[str, Any], requests: list[Any]) -> None:
        failed = [c for c in results["assertions"]["checks"] if not c["passed"]]
        self.assertEqual(failed, [], failed)
        m = results["metrics"]
        for key in sc.TOKEN_KEYS:
            self.assertIn(key, m["usage"])
        self.assertEqual(m["usage"]["model_calls"], len(requests))
        self.assertTrue(m["usage"]["usage_complete"])
        self.assertGreater(m["usage"]["cache_read_tokens"], 0)
        self.assertEqual(m["usage"]["estimated_cost_usd"], 0.0)
        self.assertEqual(m["tools"]["by_tool"]["fetch_external_nutrition_info"], 2)
        self.assertEqual(m["turns"], 7)
        self.assertGreater(m["harness_line_counts"]["total_lines"], 0)
        # Session 2 is a fresh conversation: its first request carries only one content.
        s2_first = next(r for r in requests if "what do you remember" in json.dumps(r["contents"][0]).lower())
        self.assertEqual(len(s2_first["contents"]), 1)
        json.dumps(results, default=str)

    def test_scenario_shape(self) -> None:
        sessions = self.scenario["sessions"]
        self.assertEqual(len(sessions), 2)
        self.assertTrue(6 <= sum(len(s["turns"]) for s in sessions) <= 8)
        state_types = {"memory_contains", "memory_not_contains", "saved_recipe_exists",
                       "state_file_exists", "memory_file_exists"}
        self.assertGreaterEqual(sum(a["type"] in state_types for a in self.scenario["assertions"]), 5)

    def test_api_scenario_end_to_end(self) -> None:
        client, requests = mock_client()
        results = api_main.run_scenario(self.scenario, api_args(self.tmpdir), client=client)
        self.check_results(results, requests)
        self.assertIn("API/harness/loop.py", results["metrics"]["harness_line_counts"]["files"])

    def test_sdk_scenario_end_to_end(self) -> None:
        client, requests = mock_client()
        results = sdk_main.run_scenario(self.scenario, sdk_args(self.tmpdir), client=client)
        self.check_results(results, requests)
        self.assertIn("SDK/agent.py", results["metrics"]["harness_line_counts"]["files"])

    def test_assertions_detect_failures(self) -> None:
        paths = sc.prepare_state("api", self.scenario, self.tmpdir)
        empty = {"sessions": [{"name": s["name"], "turns": [{"text": "", "tool_calls": []} for _ in s["turns"]]}
                              for s in self.scenario["sessions"]]}
        report = sc.evaluate_assertions(self.scenario, empty, paths)
        failed = {c["id"] for c in report["checks"] if not c["passed"]}
        self.assertTrue({"A1", "A2", "A3", "A5", "A6", "A7", "A8", "A10"} <= failed)
        self.assertNotIn("A4", failed)  # nothing saved, so nothing was overwritten

    def test_overwritten_preference_fails_a4(self) -> None:
        paths = sc.prepare_state("api", self.scenario, self.tmpdir)
        shared.save_user_preference_or_recipe({"type": "preference", "key": "diet", "value": "chicken lover"})
        a4 = next(a for a in self.scenario["assertions"] if a["id"] == "A4")
        self.assertFalse(sc.evaluate_assertion(a4, {"sessions": []}, paths)["passed"])

    def test_missing_key_is_reported(self) -> None:
        with mock.patch.object(sc, "load_dotenv"), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(api_main.main(["--scenario", str(ROOT / "scenario.json"),
                                            "--state-dir", str(self.tmpdir)]), 1)
            self.assertEqual(sdk_main.main(["--scenario", str(ROOT / "scenario.json"),
                                            "--state-dir", str(self.tmpdir)]), 1)


# --------------------------------------------------------------------------- #
# SDK wiring
# --------------------------------------------------------------------------- #

class SDKWiringTests(OfflineTestCase):
    def test_tools_wrap_shared_functions(self) -> None:
        from SDK.tools import Recipe, fetch_external_nutrition_info, save_user_preference_or_recipe, search_recipes

        self.assertIn("Classic Hummus", json.dumps(search_recipes("hummus")))
        self.assertIn("error", save_user_preference_or_recipe(type="bogus"))
        save_user_preference_or_recipe(type="preference", key="diet", value="vegetarian")
        save_user_preference_or_recipe(type="recipe", recipe=Recipe(title="Toast", ingredients=["bread"]))
        mem = shared.load_memory()
        self.assertEqual(mem["preferences"]["diet"], "vegetarian")
        self.assertEqual(mem["saved_recipes"][0]["title"], "Toast")
        self.assertEqual(fetch_external_nutrition_info("tofu")["result"]["calories"], 76)

    def test_declarations_generated_from_callables(self) -> None:
        from SDK.tools import TOOLS

        client, _ = mock_client()
        names = [types.FunctionDeclaration.from_callable(client=client._api_client, callable=f).name for f in TOOLS]
        self.assertEqual(names, list(tools.HANDLERS))

    def test_chat_config_and_resume(self) -> None:
        from SDK.agent import RecipeAgent as SDKAgent

        client, requests = mock_client()
        agent = SDKAgent(client=client, max_tool_calls=5)
        self.assertEqual(agent.config.automatic_function_calling.maximum_remote_calls, 5)
        self.assertIsInstance(agent.config.system_instruction, str)
        result = agent.chat("I'm vegetarian and allergic to peanuts")
        self.assertEqual(len(result["tool_calls"]), 2)
        self.assertEqual(result["model_calls"], 2)
        path = self.tmpdir / "chat.json"
        agent.save_history(path)
        resumed = SDKAgent(client=client, resume_history=path)
        self.assertEqual(len(resumed.chat_session.get_history()), len(agent.chat_session.get_history()))
        resumed.chat("Any quick dinner idea?")
        self.assertGreater(len(requests[-1]["contents"]), 4)

    def test_sdk_replaces_markers_present(self) -> None:
        text = (ROOT / "SDK" / "agent.py").read_text() + (ROOT / "SDK" / "tools.py").read_text()
        for name in ("loop.py", "history.py", "compaction.py", "caching.py", "tokens.py",
                     "tools.py", "context.py", "memory.py"):
            self.assertIn(f"# replaces: {name}", text)

    def test_no_anthropic_imports(self) -> None:
        for path in [*ROOT.glob("API/**/*.py"), *ROOT.glob("SDK/*.py"), *ROOT.glob("shared/*.py")]:
            self.assertNotIn("anthropic", path.read_text().lower(), path)
            self.assertNotIn("claude_agent_sdk", path.read_text(), path)


if __name__ == "__main__":
    unittest.main()
