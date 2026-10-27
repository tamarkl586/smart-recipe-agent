# Recipe Agent: Manual API Harness vs. Google GenAI SDK

Two implementations of the same recipe agent, both on Google Gemini (Google AI Studio, free tier):

- **API** (`API/`): a hand-written harness over `client.models.generate_content`. Eight modules own the loop, tools, history, memory, context, compaction, caching and token accounting.
- **SDK** (`SDK/`): the Google GenAI SDK's high-level constructs. A `client.chats` session with automatic function calling (AFC), with tools as plain typed Python functions.

Both use the same domain code (`shared/recipes_db.py`), the same three tools, the same system prompt and memory block, and the same scenario and assertions. Results: `results/api.json`, `results/sdk.json`. Logs: `results/api_run.log`, `results/sdk_run.log`.

> **Read this first: the two runs used different models.** The API run used `gemini-3.5-flash`. The free tier allows **20 `generate_content` requests per day per model**, and that run (15 calls) plus a probe used most of it. The SDK run on the same model was cut off by a daily-quota 429 after about 4 calls, so the SDK was re-run on **`gemini-3.5-flash-lite`**, which has its own quota. `gemini-2.0-flash` and `gemini-2.5-flash` were tried first and return 404 ("no longer available"). `gemini-3.8-flash` disconnected on every request at the time. **Model calls, tool calls, assertion results and code size compare like for like. Output tokens, thinking tokens and latency mostly reflect the model difference, not the harness.** Section 3 separates the two.

## 1. Scenario

7 turns across 2 sessions (`scenario.json`). Session 2 is a brand-new conversation. Its only link to session 1 is state persisted on disk: the memory file plus a saved transcript.

| Session | Turn | Exercises |
|---|---|---|
| 1 | "I'm vegetarian and allergic to peanuts, remember both" | memory write (2 parallel calls) |
| 1 | "Find me a quick vegetarian dinner" | internal tool `search_recipes` |
| 1 | "Calories and protein in pasta?" | external tool `fetch_external_nutrition_info` |
| 1 | "Save that Tomato Basil Pasta" | memory write (recipe) |
| 2 | "What do you remember about me?" | memory retrieval in a new session |
| 2 | "Give me the Chicken Stir Fry recipe" | request contradicting the saved diet |
| 2 | "Nutrition for chickpeas?" | external tool in session 2 |

12 assertions. **6 check persisted state** (A1–A6): diet saved, allergy saved, recipe saved, preferences not overwritten by the chicken request, session-1 transcript on disk, memory file present. 3 check tool use (A7–A9) and 3 check replies (A10–A12). Each implementation gets its own memory file under `results/state/`.

## 2. Summary comparison

| Metric | API (manual harness) | SDK (Chat + AFC) |
|---|---|---|
| Model | `gemini-3.5-flash` | `gemini-3.5-flash-lite` |
| **Assertions passed** | **12 / 12 (100%)** | **12 / 12 (100%)** |
| Input tokens (uncached) | 23,019 | 18,172 |
| Output tokens (incl. thinking) | 2,514 | 711 |
| — of which thinking | 1,799 | 0 |
| Cache read tokens | 0 | 0 |
| Cache write tokens | 0 | 0 |
| Cache hit % | 0.0% | 0.0% |
| Model calls | 15 | 14 |
| Tool calls (errors) | 9 (0) | 8 (0) |
| — save / search / nutrition | 3 / 4 / 2 | 3 / 3 / 2 |
| Cost (free tier) | **$0.00** | **$0.00** |
| Wall time, whole scenario | 118.5 s | 10.8 s |
| Median turn latency | 9.7 s | 1.4 s |
| Compactions | 0 | 0 (no SDK equivalent) |
| Lines of code (agent + CLI) | 943 | 365 |

Per session:

| | API s1 | API s2 | SDK s1 | SDK s2 |
|---|---|---|---|---|
| Model calls | 8 | 7 | 9 | 5 |
| Input tokens | 12,778 | 10,241 | 12,604 | 5,568 |
| Output tokens | 1,507 | 1,007 | 436 | 275 |
| Duration | 60.7 s | 57.7 s | 6.9 s | 3.7 s |

All token counts come from Gemini's `usage_metadata`. The uncached input count is `prompt_token_count − cached_content_token_count`, and output is `candidates_token_count + thoughts_token_count`. Implicit caching has no write charge, so cache write is 0 by definition. Both result files have `usage_complete: true`; see 4.1 for what that took on the SDK side.

## 3. Token consumption: where the SDK cost more or less, and why

**Model calls are a harness property, and they came out equal.** Both implementations make exactly *(tool rounds + 1)* calls per turn: 2 calls for a turn with one tool round, 1 for a turn with none. Neither adds hidden requests. The totals differ (15 vs 14) only because the models chose differently:
- The API's model made **3 searches** to find Chicken Stir Fry in session 2: one extra call costing about 1.2–1.6k input tokens.
- The SDK's model did an **extra search before saving** the pasta recipe in session 1.

Per tool round, the two harnesses cost the same number of calls.

**Input tokens: the SDK used 21% fewer (18.2k vs 23.0k). This is mostly not the harness.** Every call re-sends the whole conversation, and both harnesses send the same pieces: system prompt, tool declarations, transcript, and a memory block per turn. The gap comes from three things:
1. **Longer answers from the larger model.** `flash` wrote about 3.5× more output (2,514 vs 711 tokens), and every earlier answer is re-sent as input on every later call. That growth explains most of session 2's gap (10.2k vs 5.6k).
2. **Replayed thinking.** `flash` used thinking (1,799 tokens) and its model turns carry thought signatures that must be sent back. `flash-lite` didn't think (0).
3. **The extra search call** described above.

Session 1, where both models behaved alike, shows the harness overhead. The token totals there are nearly identical (12.8k vs 12.6k), and the SDK made one *more* call. The SDK generates its tool declarations from Python signatures and docstrings, the API sends hand-written JSON schemas, and the two come out about the same size.

**Output tokens: the SDK used 72% fewer (711 vs 2,514).** This is entirely the model. 1,799 of the API run's 2,514 output tokens (72%) were thinking tokens. On the paid tier, thinking is billed as output.

**Caching: 0% for both.** Neither run got an implicit cache hit. Gemini's implicit caching is best-effort and needs a repeated prefix above a minimum size, and these requests are small: the first request totals about 0.7k tokens including the user message, and the largest context was about 2.5k. The API harness has an explicit-cache path (`--explicit-cache`), which would put the system prompt and tools in a server-side `CachedContent`. It falls back to implicit caching because the stable prefix is below the 1,024-token minimum, so it was off for this run. At this size, caching can't matter for either implementation.

**Compaction never triggered.** The threshold is 20k tokens, and the largest single context was about 2.5k.

**Cost.** Both runs cost **$0.00** on the free tier. The real constraint here is the **request quota (20 per day per model)**, not tokens. On the free tier, the SDK's lack of a token-saving mechanism costs nothing, and an extra model call costs real quota.

**Latency (118.5 s vs 10.8 s).** This is mostly the model and the rate limit, not the harness:
- `flash` turns took 1.8–14 s, except two turns at about 42 s each. Those two match the client's rate-limit retry back-off on 429 responses.
- `flash-lite` turns took 0.9–2.2 s.

Neither harness adds measurable latency of its own.

## 4. Control: what the raw API gives you that the SDK takes away

### 4.1 Per-call visibility
- **Token usage.** With AFC, a single `send_message()` can make several model calls, but the response carries `usage_metadata` from **the last call only**. Without a workaround, the SDK run would have under-reported tokens on every turn that used a tool. To get complete numbers, the SDK agent wraps the client's private per-call method (`models._generate_content_with_continuation`) in a `UsageMeter`. It only measures, but it depends on a private API that could change in any release, which is why the results carry a `usage_complete` flag. The API harness sees every response directly.
- **Which tools were called.** In this SDK version (`google-genai` 2.33), `Chat.send_message` never fills in `response.automatic_function_calling_history`. The SDK agent rebuilds the list of tool calls by comparing the chat history before and after each turn. The API harness logs each call as it dispatches it.

### 4.2 Hooks between calls
The API loop runs *your* code between every model call. AFC runs the whole call → execute → respond cycle inside the SDK, which takes away:
- **Approval gates and policy checks** before a tool runs. For example, you can't confirm a memory write or block a save that contradicts the user's diet.
- **Per-round limits other than a call count.** AFC offers only `maximum_remote_calls`. When that limit is reached, the SDK returns the model's pending function call *without running it*. The API harness instead answers any unanswered calls with an error, so history stays valid (`_close_open_tool_calls`).
- **Per-call transport handling.** The SDK's retry doesn't cover a server dropping the connection (`RemoteProtocolError`), which this project hit. The API harness wraps its one call. In the SDK, the only place to add the same retry was the same private per-call method, so that a retry can't duplicate chat history.

### 4.3 Tool contract
- **Schemas.** The SDK generates them from type hints and docstrings, which is less code but less control. `from __future__ import annotations` turns the hints into strings, and **every SDK tool call then failed** with `isinstance() arg 2 must be a type`. The offline tests caught this. The API harness sends explicit JSON Schema.
- **Errors.** AFC catches tool exceptions and returns `{"error": str(e)}` with whatever message the exception had. The API harness decides exactly what the model sees, for example listing the available tools when the model names an unknown one.

### 4.4 Context management
- **Compaction: none in the SDK.** A `Chat` keeps growing until the 1M-token context window is full. The API harness summarizes the whole history into one block when context passes a threshold. It does this only between turns, never in the middle of a tool round, and has an offline fallback if the summary call fails.
- **History.** Both persist the transcript and can resume from it: the API saves its own message list, the SDK saves `chat.get_history()` and passes it back as `history=`. Only the API harness *guarantees* the transcript is append-only, so the request prefix is byte-stable across calls. That property is what implicit caching relies on, and it keeps thought signatures valid.
- **Caching.** The API harness can set up explicit caching itself. The SDK relies on implicit caching. Explicit caching is possible through `config.cached_content`, but the SDK doesn't build or manage it.

### 4.5 What the SDK does better
Less code (365 vs 943 lines), schemas that can't drift from the Python signatures, automatic conversion of arguments into pydantic objects (the `Recipe` model), and no tool-loop bugs to write. Both implementations passed the same 12 assertions.

## 5. Line counts per harness responsibility

API: one file per responsibility. SDK: lines attributable to each `# replaces:` marker in `SDK/agent.py` and `SDK/tools.py` (approximate). Counts include docstrings and comments.

| Responsibility | API file | API lines | SDK lines | How the SDK covers it |
|---|---|---|---|---|
| Agent loop | `loop.py` | 115 | ~6 | `AutomaticFunctionCallingConfig` + one `send_message()` |
| Tools (schemas, dispatch, errors) | `tools.py` | 146 | 78 | typed functions in `SDK/tools.py`; the SDK generates declarations and dispatches |
| History (order, save/load) | `history.py` | 82 | ~13 | `client.chats` keeps the transcript; `get_history()` / `history=` |
| Context (system prompt, config) | `context.py` | 75 | ~26 | `GenerateContentConfig(system_instruction=…, tools=…)` |
| Memory injection | `memory.py` | 58 | ~11 | not provided by the SDK; same `<user_memory>` block written by hand |
| Compaction | `compaction.py` | 72 | 0 | **not provided**; relies on the 1M-token window |
| Caching | `caching.py` | 68 | 0 | implicit caching only (no code) |
| Token accounting | `tokens.py` | 108 | ~42 | `usage_metadata` plus the `UsageMeter` workaround (4.1) |
| Package init | `harness/__init__.py` | 7 | 1 | |
| **Harness subtotal** | `API/harness/` | **731** | **~177** | |
| Agent wiring | `agent.py` | 100 | (included above) | |
| CLI / scenario driver | `main.py` | 112 | 106 | |
| **Total** (as recorded in results) | | **943** | **365** | **2.6× less code** |

Shared by both and not counted: `shared/recipes_db.py` (domain + memory file), `shared/scenario.py` (scenario runner, assertions, metrics) and `shared/transport.py` (timeout and retry settings, identical for both).

Where the SDK saves code: the loop (about 19× less), history (about 6×) and tools (about 2×). Where it saves nothing: memory injection must be written by hand either way, and token accounting needed extra code just to get correct numbers. Compaction and caching cost 0 lines in the SDK only because it doesn't do them.

## 6. Recommendation

**For this recipe agent, I would ship the SDK version (Chat + AFC), with the usage meter kept as a known dependency on a private API.**

- **It met every requirement with 2.6× less code.** 12/12 assertions passed, including the cross-session memory checks and the conflicting request. Memory, which is the hard part of this agent, is our own code in both versions anyway.
- **The API harness's extra machinery has nothing to do here.** Sessions are a few turns, the largest context was about 2.5k tokens against a 1M window, so compaction never triggers. Prompts are too small to cache. Model calls per tool round are identical between the two. On the free tier, the binding constraint is requests per day, and the harness doesn't reduce requests.
- **The tools are low-risk.** Searching recipes and looking up nutrition are read-only. The only write is to the user's own preferences, so per-call approval gates aren't needed.

**When I would switch to the manual API harness:** if the agent gains long-running sessions (compaction), side-effecting or expensive tools that need approval gates or per-call policies, a large stable prompt worth caching explicitly, or a requirement for exact per-call token and cost accounting without relying on SDK internals. The last two points of section 4.1 are the biggest risk in the SDK version: correct token numbers depend on a private method, and tool calls had to be reconstructed from history.

**Caveat on the evidence:** the token and latency columns compare `gemini-3.5-flash` (API) with `gemini-3.5-flash-lite` (SDK), because of the free-tier daily quota. Re-running the SDK on `gemini-3.5-flash` after the quota resets would make those columns comparable. The structural findings (calls per round, the control differences, code size) don't depend on the model.

## 7. Verification and reproduction

```bash
.venv/bin/pip install -r requirements.txt              # google-genai
.venv/bin/python -m unittest discover -s tests         # 32 offline tests, no API key needed
# GEMINI_API_KEY=... in .env
.venv/bin/python -m API.main --scenario scenario.json --out results/api.json
.venv/bin/python -m SDK.main --scenario scenario.json --out results/sdk.json --model gemini-3.5-flash-lite
```

The offline suite uses a real `genai.Client` whose network call is replaced by a scripted model that rejects requests the real API would refuse: roles out of order, or a function call without a matching response. This runs the API harness *and* the SDK's real Chat/AFC code through the full two-session scenario offline, with all 12 assertions checked.
