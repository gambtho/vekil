# Proxy-mediated Web Search

`web_search` is an optional top-level block in the JSON/YAML [providers config](provider-routing.md), alongside `providers`. It is disabled by default, and leaving it unset preserves the normal passthrough behavior byte-for-byte.

Anthropic's hosted `web_search` server tool is rejected by GitHub Copilot on `/v1/messages`, so `WebSearch` in Claude Code and other Anthropic-protocol clients fails for every `claude-*` model routed to Copilot. Copilot does serve hosted web search on `/responses` for `gpt-5.x` models. When this block is enabled, Vekil intercepts the hosted tool, runs the search as an internal `/responses` call on the same provider, and returns Anthropic-shaped `server_tool_use` and `web_search_tool_result` blocks. No external search vendor and no additional credential are involved.

## When mediation engages

All of the following must hold. Any miss falls through to today's passthrough, untouched:

1. `web_search.enabled` is `true`.
2. The inbound `/v1/messages` body carries a `type: web_search_*` tool.
3. The model resolves to the direct-passthrough route **and** its provider is `copilot`.
4. A delegate model advertising `/responses` is discoverable **on that same provider**.
5. The client does not already define its own client tool named `web_search`.
6. The effective search budget — the client's own `max_uses` on the hosted tool, min'd with `web_search.max_searches` — is greater than zero.

Condition 3 is a hard exclusion, not an optimization. `anthropic-compatible` providers also take the direct path, and hosted `web_search` already works natively there; mediating would replace a working path with a synthesized approximation.

Condition 4 is a privacy boundary. Delegation never leaves the provider the client is already talking to, so search queries cannot be routed to an unrelated Azure or Codex provider.

Condition 6 has a sharp edge: a client that sends `max_uses: 0` (or a negative value) does not get an in-band `max_uses_exceeded` result — mediation declines entirely, the same as any other miss, and the hosted tool is forwarded upstream as-is. Against Copilot that forwarded request then gets the exact 400 rejection this feature exists to avoid. Budget exhaustion mid-turn (below) is a different, later case: once mediation has already engaged, running out of remaining searches during the loop *does* produce the in-band error.

## Behavior

- **Request rewrite.** The hosted tool entry is replaced with a plain `web_search` function tool taking a single `query` string. `max_uses`, `allowed_domains`, `blocked_domains`, and `user_location` are held proxy-side and never forwarded. `stream` is forced to `false` upstream; the client's original value is honored on the way back.
- **Loop.** Each turn is dispatched to the same pinned route target. `web_search` tool calls are delegated, the assistant turn is replayed with a user turn of tool results appended, and the next turn is dispatched. The loop stops when no unresolved search remains, when the budget is exhausted, or on a mixed turn.
- **Send budget.** A mediated turn's dispatches and the delegated `/responses` searches it triggers both show up in `upstream_sends`, but neither draws on the route's `max_upstream_sends`: mediated dispatches use their own attempt kind and counter. A failed mediated turn therefore always has the client's full budget available to re-issue the original request through normal passthrough. Failover across targets mid-loop is not supported: a target change would invalidate the accumulated conversation, so a failed continuation falls back to passthrough instead.
- **Budget exhaustion is in-band, but only once mediation has already engaged.** Unserved calls are answered with `web_search_tool_result_error` and error code `max_uses_exceeded`, which is Anthropic's own documented code for exactly this condition. Note that the error form's `content` is a single object, not a list. A request that arrives with a zero or negative budget never reaches this path at all — see condition 6 above.
- **Mixed turns.** When a turn contains both a search and real client `tool_use` blocks, the search is resolved inline, both are emitted, `stop_reason` is `tool_use`, and the loop stops. The client answers only its own tools.
- **Streaming clients.** Streaming clients receive a replay of the finished message: `message_start`, per-block `content_block_start` / deltas / `content_block_stop`, `message_delta`, `message_stop`. `web_search_tool_result` is emitted whole inside `content_block_start` with no deltas, per the Anthropic contract.
- **Delegation constraints.** Each delegated call pins `reasoning.effort: low` and the configured `search_context_size`, and constrains the model to a single search with no refinement. Both constraints are load-bearing: without them latency roughly quadruples.
- **Domain filters are enforced locally.** `allowed_domains` and `blocked_domains` are passed to the delegate *and* applied again to the returned results, so filtering does not depend on unverified upstream enforcement.
- **`/v1/messages/count_tokens` keeps working, but it is not untouched.** Count-tokens takes the translated path, so it does two things: it substitutes the stand-in `web_search` function tool (a hosted tool in the request does not fail the probe whether or not mediation is enabled), and it decodes replayed `server_tool_use` / `web_search_tool_result` blocks back into plain `tool_use` / `tool_result` before translating. That decode is deliberately not gated on `web_search.enabled` — a history can still carry blocks synthesized while the feature was on in an earlier session, and the translator rejects unknown block types, so without it the first count-tokens call after a mediated search would 400.
- **A mediated turn loses `redacted_thinking.data` and text-block `citations`.** The mediated path re-marshals decoded content blocks, and `models.ContentBlock` has no field for either, so both are dropped. This affects the response the CLIENT receives, not just the upstream continuation. Non-mediated turns are unaffected: plain passthrough never re-marshals the body.
- **Fail-open.** Delegation errors, timeouts, invalid JSON, an unusable delegate, or a failed mediated turn all re-issue the original client body through the normal passthrough. The cost is one wasted upstream call. Plain-string message content that never touched a synthesized block passes through byte-identical, except when an adjacency merge (see below) pulls it into a converted neighbor.

## Replaying a prior turn back to the model

A client that persists conversation history and resubmits it (for example on the next turn) sends Vekil's own synthesized `server_tool_use` / `web_search_tool_result` blocks back as part of the request body. Vekil decodes these before forwarding upstream, splitting by block **position**, not just type: everything in the original assistant turn that came before the last synthesized result stays in a "pre" assistant message, the synthesized results become a `tool_result` user turn inserted immediately after, and anything that came after (such as the model's own answer text) becomes a "post" assistant message. Either assistant side is omitted when empty, and a final pass guarantees no two adjacent messages end up sharing a role. A message can therefore expand into up to three messages on replay. Bodies with no synthesized blocks are returned unchanged.

## `encrypted_content` is a stateless token, not authenticated encryption

Synthesized results carry a Vekil-minted, versioned, size-capped `encrypted_content` token that stateless-encodes the result snippet, mirroring the compaction token format. It is deliberately not a store key, so it survives a proxy restart.

It is **not** authenticated encryption and grants no capability the client lacks: the client already controls the whole message history and can fabricate a `tool_result` containing arbitrary text. What is required instead is strict handling on the way back in — version check, size cap, and schema validation — failing **closed** to a `web_search_tool_result_error` rather than passing malformed or forged content to the model.

## Minimal example

```yaml
web_search:
  enabled: true
```

With auto-discovery, Vekil picks the first catalog model on the conversation's provider that advertises `/responses`.

## Pinned-delegate example

```yaml
web_search:
  enabled: true
  delegate_model: gpt-5.1
  max_searches: 3
  timeout_ms: 45000
  search_context_size: medium
  max_results: 5
```

Defaults:

| Setting | Default |
|---------|---------|
| `web_search.enabled` | `false` |
| `web_search.delegate_model` | `""` (auto-discover the first `/responses` model on the same provider) |
| `web_search.max_searches` | `5` (per turn, min'd with the client's `max_uses`) |
| `web_search.timeout_ms` | `30000` (per delegated call) |
| `web_search.search_context_size` | `low` (`low` \| `medium` \| `high`) |
| `web_search.max_results` | `10` |

There is no CLI flag or environment variable: the config block is the whole surface and `enabled: false` is the kill switch. There is no separate wall-clock knob either — each delegated call is bounded by `timeout_ms` and the loop as a whole by the non-streaming upstream deadline.

## Observability

- A mediated turn's dispatches and the delegated `/responses` searches it triggers both appear in `upstream_sends`, so the traffic someone debugging this would look for is visible there.
- Delegated tokens **and** non-final loop-turn tokens are booked additively as internal usage on a context that survives client cancellation, so a three-search turn does not under-report roughly three model turns of spend.
- Debug logs use the endpoint label `messages/web_search/internal` and carry a `reason` field on every skip and fallback.

## Limits and tradeoffs

- **Latency.** Roughly 15s of silence for a one-search turn and ~35s for three, composed from live probes measured against Copilot in August 2026 (~8.3-10.7s per delegated search once constrained to a single search with `reasoning.effort: low` and `search_context_size: low`, plus ~2.1s per base model call). Expect variation by query and model; these are approximate, not a guarantee. Unconstrained, one delegated call fanned out to 12 internal searches and took 39.8s, which is why the single-search/no-refinement constraint above is load-bearing. This is the deliberate cost of avoiding a live-stream multiplexer; searches run non-streaming upstream and the finished message is replayed as SSE.
- **Quota.** Each delegated search is an extra provider request.
- **Results are a model's synthesis**, not raw rankings; `page_age` and ordering are approximations.
- **Not supported.** `web_fetch`, Anthropic's dynamic-filtering tool versions (`web_search_20260209`, `web_search_20260318`), the `pause_turn` state machine, and providers with no hosted search anywhere in their catalog (the feature no-ops).
