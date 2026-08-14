# Proxy-mediated Anthropic `web_search`

Date: 2026-08-13
Status: approved for planning
Upstream discussion: https://github.com/sozercan/vekil/issues/341

## Problem

Anthropic's hosted `web_search` server tool fails for every `claude-*` model
behind Vekil. `claude-*` models on Copilot advertise `/v1/messages`, so
`shouldForwardAnthropicMessagesDirect` (`proxy/chat_handlers.go:1626-1641`)
forwards the client body verbatim and Copilot rejects the hosted tool:

```
HTTP 400
{"error":{"message":"The use of the web search tool is not supported.","code":"unsupported_value"}}
```

Vekil is not at fault — it forwards correctly and relays the rejection. The
practical result is that `WebSearch` in Claude Code is unusable through the
proxy.

Copilot does serve hosted web search, but only for `gpt-5.x` models on
`/responses`. That makes a proxy-mediated implementation possible without any
external search vendor or additional subscription.

## Goals

- `WebSearch` works transparently in Claude Code and any other Anthropic-protocol
  client, for `claude-*` models routed to Copilot.
- No external search dependency, no new credential.
- Opt-in and fail-open: when disabled or when anything goes wrong, behavior is
  byte-for-byte what it is today.

## Non-goals

- `web_fetch_20250910`. No `web_fetch` handling exists anywhere in the codebase
  today, and Claude Code's `WebFetch` is a client-side tool.
- Anthropic's dynamic-filtering tool versions (`web_search_20260209`,
  `web_search_20260318`). These require a code-execution container and emit
  nested `caller`-tagged pairs that cannot be reproduced.
- The `pause_turn` state machine. Mixed turns are handled by resolving searches
  inline (see Mixed turns).
- Providers with no hosted search anywhere in their catalog. The feature no-ops.

## Evidence and constraints

Established from the repository and from probes against a live Copilot upstream.

### Repository

- The direct path forwards raw client bytes with no body rewriting
  (`proxy/chat_handlers.go:1665-1666`). This introduces the first rewrite there.
- The direct path has no policy involvement; `planOpenAIChatPolicy` is
  translated-path only (`proxy/chat_handlers.go:2019`).
- No Anthropic SSE emitter from a complete message exists. `anthropicStreamState`
  (`proxy/streaming.go:1724-2024`) has correct block bookkeeping but its input
  surface is entirely OpenAI-shaped and it knows only `text` and `tool_use`.
  A sibling emitter is net-new, built on `writeSSEEvent`
  (`proxy/streaming.go:102`).
- Precedent for synthesizing a full SSE stream from proxy-internal state exists
  in the Responses protocol: `syntheticCompactionTriggerResponse`
  (`proxy/responses_handler.go:3841-3880`).
- Internal-call template (policy classifier): separate route target,
  `newProviderJSONRequest` for auth, `singleInferenceSend` with no failover so
  internal calls cannot burn the client's route-attempt budget, usage booked to
  a separate bucket (`proxy/policy_routing.go:862-931`).
- Internal calls are tagged `withRouteAttemptKind` so route observability does
  not confuse them with client attempts (`proxy/responses_handler.go:3106`).
- Usage accounting on the direct path is single-shot body sniffing
  (`proxy/upstream_http.go:734`). Multiple upstream calls aggregate nothing;
  `addInternalUsage` / `observeInternalResponsesUsage`
  (`proxy/request_summary.go:570-599`) is the additive mechanism, flushed via
  `defer` on a context that survives client cancellation.
- Fail-open convention is "return the zero-value result and let the caller keep
  the original", never propagating to the client, and discarding invalid
  replacements rather than trusting them (`proxy/tool_optimizer.go:245-315`).
- Providers config is strict-decoded (`proxy/providers.go:435`, `:462`), so a new
  block must be an explicit struct field.
- Non-200 responses on the direct path are relayed verbatim by
  `writeUpstreamResponse` (`proxy/chat_handlers.go:1701`), so Copilot-shaped
  error bodies reach clients without an Anthropic envelope.
- Streaming commits `downstreamCommitmentProtocolFrame` before any byte is
  written (`proxy/chat_handlers.go:1683`).
- `translator.go:288` hard-rejects unknown content block types, so synthesized
  blocks would 400 if a mediated conversation crossed to the translated path.
- Explicit routes default to `MaxUpstreamSends: 1`
  (`proxy/model_routes_config.go:1041-1045`); `reserveSendAtDispatch` decrements
  it per dispatch and refuses further sends with `routeRetrySuppressedBudget`
  once exhausted (`proxy/route_executor.go:499-519`). A multi-turn loop on the
  client's own route operation is therefore refused on turn 2.
- `shouldForwardAnthropicMessagesDirect` returns true for
  `providerTypeAnthropicCompatible` as well as Copilot
  (`proxy/chat_handlers.go:1626-1638`), where hosted `web_search` already works.

### Measured against live Copilot

- Hosted `web_search` works on `/responses` for `gpt-5.x`: real searches, real
  `url_citation` annotations carrying `url` and `title`.
- Non-streaming tool calls on the Anthropic passthrough are reliable — 3/3 runs
  returned `stop_reason: tool_use` with all expected parallel `tool_use` blocks,
  ~2.1s each. The forced-streaming rationale in CLAUDE.md concerns the
  translated chat path, not this one.
- Timeouts are not a constraint: 5 min non-streaming (`proxy/handler.go:37`),
  60+5 min server write timeout (`proxy/handler.go:39`, `:986`).
- Delegation is prompt-sensitive. Unconstrained, one delegated call fanned out to
  12 internal searches and took 39.8s. Constrained to a single search with
  `reasoning.effort: low` and `search_context_size: low`: 8–11s, 10 results
  carrying `url`, `title`, `snippet` and `page_age`.
- `filters.allowed_domains` is accepted and echoed on the delegated side.
- `filters.blocked_domains` is accepted and echoed, but **enforcement is
  unverified** — a single probe showed no blocked-domain citations, which is not
  proof. Vekil post-filters regardless.
- **`/v1/messages/count_tokens` must keep working.** It takes the translated
  path, and Copilot's `/chat/completions` accepts the degenerate parameterless
  tool, so it returns 200 today. Because rejection lands in the shared
  `TranslateAnthropicToOpenAI`, the count_tokens probe must substitute the
  stand-in function tool rather than reject — Claude Code calls count_tokens
  routinely, and a 400 there would break clients whose message turns work fine.

## Design

### Activation

Mediation engages only when all hold:

1. `web_search.enabled` is true in the providers config.
2. The inbound `/v1/messages` body carries a `type: web_search_*` tool.
3. The model resolves to the direct-passthrough route **and its provider is
   `providerTypeCopilot`**.
4. A delegate model is discoverable **on that same provider**.
5. The client does not already define a client tool named `web_search`.

Any miss falls through to today's passthrough, untouched.

Condition 3 is a hard exclusion, not an optimization.
`shouldForwardAnthropicMessagesDirect` also returns true for
`providerTypeAnthropicCompatible` (`proxy/chat_handlers.go:1630-1632`), where
hosted `web_search` **already works natively**. Mediating there would replace a
working path with a synthesized approximation. Only Copilot-owned models, which
have no other way to search, are eligible.

Condition 4 constrains delegation to the conversation's own provider. Vekil is a
multi-provider proxy and an unconstrained catalog scan could route a user's
search queries to an Azure or Codex provider that has nothing to do with the
conversation. That is a privacy boundary, not just a correctness one: search
queries must not leave the provider the client is already talking to.

Condition 5 avoids a renaming scheme for a rare case: a client that defines its
own `web_search` client tool alongside the hosted one would make the intercepted
call ambiguous.

### Request rewrite

The hosted entry is removed from `tools[]` and replaced with:

```json
{"name":"web_search",
 "input_schema":{"type":"object",
                 "properties":{"query":{"type":"string"}},
                 "required":["query"]}}
```

`max_uses`, `allowed_domains`, `blocked_domains` and `user_location` are held in
proxy-side state and never forwarded to Copilot. `stream` is forced to `false`;
the client's original value is remembered for emission.

### Loop

Bounded by `min(client max_uses, config max_searches)`. There is no separate
wall-clock knob: each delegated call is bounded by `timeout_ms`, and the loop as
a whole is bounded by the inherited non-streaming upstream deadline
(`upstreamTimeout`, `proxy/handler.go:37`).

Each iteration POSTs the rewritten body. If the assistant turn contains
`tool_use` blocks named `web_search`, each is delegated, the assistant turn is
appended verbatim, and a user turn carrying the `tool_result` blocks is appended
before re-POSTing.

The loop terminates when the assistant turn contains no unresolved `web_search`
call, when the budget is exhausted, or on a mixed turn.

If the client omits `max_uses`, the ceiling is `max_searches` alone.

#### Route send budget

This is the constraint the loop must be built around. Explicit routes default to
`MaxUpstreamSends: 1` (`proxy/model_routes_config.go:1041-1045`), every dispatch
decrements it in `reserveSendAtDispatch`, and once exhausted further sends are
refused with `routeRetrySuppressedBudget` (`proxy/route_executor.go:499-519`).
A naive loop is therefore rejected on its second turn.

**Every mediated dispatch, turn 1 included**, must run under its own attempt kind
and counter — not just continuations. `HandleAnthropicMessages` admits a route
operation onto the request context (`proxy/chat_handlers.go:1941-1950`) and the
direct forwarder reuses it (`:1654`). If turn 1 drew on the client's counter, it
would leave `remainingUpstreamSends == 0` and the passthrough fallback would
itself be refused with `routeRetrySuppressedBudget` — the fail-open guarantee
would be dead on arrival. The precedent for the attempt kind is
`routeAttemptCompaction` (`proxy/responses_handler.go:3106`).

Every continuation must also pin the same target that served turn 1. Failover
across targets mid-loop is not supported: a target change would invalidate the
accumulated conversation state, so a failed continuation falls back to
passthrough instead of switching targets. Target pinning does not need a forced
pin — the executor soft-pins on success and adding the mediation attempt kind to
`allowsAutomaticTargetSwitch`'s deny list (`proxy/route_executor.go:468-470`)
blocks the switch. A hard pin would persist into the fallback and strip its
failover, so it is deliberately avoided.

#### Budget exhaustion

Exhaustion is client-visible and in-band, not a silent stop. When the assistant
requests more searches than remain, the unserved calls are answered with:

```json
{"type":"web_search_tool_result","tool_use_id":"srvtoolu_…",
 "content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}}
```

`max_uses_exceeded` is Anthropic's own documented code for exactly this
condition, so a compliant client already handles it. Note the error form's
`content` is a single object, not a list.

### Delegation

One internal `/responses` call per intercepted `tool_use`, following the
classifier template: resolve the delegate model's own route target, authenticate
via `newProviderJSONRequest`, dispatch via `singleInferenceSend` so failures
cannot consume the client's failover budget, tagged with a `routeAttemptKind` so
route observability does not treat it as a client attempt.

The delegated request pins `reasoning.effort: low` and the configured
`search_context_size`, and its developer prompt constrains the model to a single
search with no refinement, returning only a JSON array of
`{url, title, snippet, page_age}`. Both constraints are load-bearing: without
them latency roughly quadruples.

Client `allowed_domains` / `blocked_domains` map to `filters.allowed_domains` /
`filters.blocked_domains`; `user_location` maps across directly.

Responses are read under a package-level byte cap (mirroring the classifier's
`MaxResponseBytes`, `proxy/chat_policy_classifier.go:747-749`), JSON-validated,
and post-filtered against `blocked_domains` and `allowed_domains` before block
synthesis, regardless of what was sent upstream. Invalid or unparseable output
is treated as a delegation failure.

### Block synthesis

```json
{"type":"server_tool_use","id":"srvtoolu_<22 base64url>","name":"web_search",
 "input":{"query":"…"}}
{"type":"web_search_tool_result","tool_use_id":"srvtoolu_<same>",
 "content":[{"type":"web_search_result","url":"…","title":"…",
             "encrypted_content":"…","page_age":"…"}]}
```

`usage.server_tool_use.web_search_requests` counts **delegated calls**, not
Copilot's internal fan-out, which is invisible to the client and can be an order
of magnitude larger.

`encrypted_content` is a Vekil-minted, versioned, size-capped token that
stateless-encodes the result snippet. It is deliberately not a store key: a
stateless token survives proxy restart, unlike the Responses replay store.
Format is a version prefix plus base64, mirroring `encodeSyntheticCompaction`
(`proxy/compaction.go:19-24`).

It is **not** authenticated encryption, and the spec does not claim it is. The
client holds these tokens and returns them, so it can trivially forge one. That
is acceptable because it grants no capability the client lacks: the client
already controls the entire message history and can fabricate a `tool_result`
containing arbitrary text. Forging a token is a longer path to something already
permitted, not a privilege escalation. Nor is there a confidentiality
requirement — the payload is search results the user asked for.

What is required instead is strict handling: version check, size cap, and
schema validation on decode, failing **closed** to a `web_search_tool_result_error`
rather than passing malformed content to the model. Adding a MAC would mean key
provisioning, rotation and replica-sharing for no threat that isn't already
open, so it is deliberately excluded.

Success `content` is a list; the error form is a single object
(`web_search_tool_result_error`). These are different JSON types and must not be
conflated.

### Mixed turns

When a turn contains both our search and real client `tool_use` blocks, the
search is resolved and emitted as `server_tool_use` + `web_search_tool_result`
inline, followed by the client's `tool_use` blocks, with
`stop_reason: tool_use`. The loop then stops. The client answers only its own
tools. This matches Anthropic's real shape for a completed server tool preceding
client calls and avoids the `pause_turn` machinery entirely.

### Replay decode

Inbound messages on subsequent turns carry the synthesized blocks. Before
forwarding, `server_tool_use` is converted back to an assistant `tool_use` and
`web_search_tool_result` to a user `tool_result` with the decoded snippet.

This must be route-aware, and route-awareness must cover failover as well as
model choice: `translator.go:288` rejects unknown content block types, so a
mediated conversation that crosses to the translated path would 400.

### Emission

Non-streaming clients receive the aggregated JSON message.

Streaming clients receive a replay of the finished message: `message_start`,
per-block `content_block_start` / deltas / `content_block_stop`,
`message_delta`, `message_stop`. `web_search_tool_result` is emitted whole
inside `content_block_start` with no deltas, per the Anthropic contract. This is
a net-new emitter built on `writeSSEEvent`; `anthropicStreamState` is not
reusable because its input surface is OpenAI-shaped.

### Failure model

Every failure path re-issues the **original** client body through the normal
passthrough. Because upstream is forced non-streaming and downstream commitment
is deferred until the final emit, fallback is always available; the cost is one
wasted upstream call.

Hard rule: never fall back after `markExplicitRouteDownstreamCommitment`.

When mediation is disabled, or when it declines to engage, the request is byte-for-byte
what it is today. When mediation engages and then fails, the client sees the
same outcome it would have seen with the feature off — modulo the error-envelope
fix below, which deliberately changes that outcome for all direct-path errors.
This matches the tool-optimizer fail-open contract in CLAUDE.md.

## Configuration

A top-level block in the providers config, mirroring `tool_optimizers`:

```yaml
web_search:
  enabled: false
  delegate_model: ""        # empty = auto-discover first /responses model
  max_searches: 5           # per-turn ceiling, min'd with the client's max_uses
  timeout_ms: 30000         # per delegated call
  search_context_size: low  # low | medium | high
  max_results: 10
```

Package-level `defaultWebSearch*` consts, `defaultWebSearchConfig()`, an
idempotent `withDefaults()` applied at load and at construction, and
`initializeWebSearch()` from `NewProxyHandler`.

No CLI flag or env var: the config block is the whole surface and
`enabled: false` is already the kill switch.

Auto-discovery picks the first catalog model advertising `/responses`, cached,
overridable via `delegate_model`. If none is found the feature no-ops.

## Observability

- `summary.RecordUpstreamSend()` per physical dispatch so `upstream_sends` stays
  honest across the loop.
- `observeInternalResponsesUsage` for delegated tokens, flushed once via `defer`
  on a context that survives client cancellation, so internal spend is additive
  rather than clobbering the turn's usage.
- **Continuation turns must be accounted too.** Existing direct-path accounting
  observes only the emitted response body (`proxy/upstream_http.go:733-735`), so
  loop turns 2..N would otherwise spend Copilot tokens invisibly. Their usage is
  accumulated alongside the delegated Responses spend and flushed through the
  same additive path. Without this, a three-search turn under-reports roughly
  three model turns of tokens.
- Debug logs labelled `messages/web_search/internal`, with a `reason` field on
  every skip and fallback.

## Structure

One new file `proxy/anthropic_web_search.go`, plus a branch in
`HandleAnthropicMessages` ahead of `forwardAnthropicMessagesDirect`. Four
internal units:

| Unit | Responsibility |
|---|---|
| detector/rewriter | Find the hosted tool, swap it, capture filter state, decode replayed blocks |
| loop | Drive bounded turns against Copilot; own the budget |
| delegator | One internal `/responses` call; parse, validate, post-filter |
| emitter | Synthesize blocks; write JSON or replay as SSE |

## Included fixes

Both fixes below change behavior **even when `web_search.enabled` is false**.
That is intentional and was an explicit scope decision, but it means the
feature-off guarantee covers the mediation path only, not these two. Each is
justified independently of the feature.

**`AnthropicTool` decode.** `models.AnthropicTool`
(`models/anthropic.go:58-63`) has only `Name`, `Description`, `InputSchema`, so
`type`, `max_uses` and the domain filters are dropped by `encoding/json`. Add
`Type`, `MaxUses`, `AllowedDomains`, `BlockedDomains`, `UserLocation`. Required
for detection; `models/` stays data-only.

**Translated-path rejection.** With hosted tools now detectable, the translated
path rejects them with a clear Anthropic 400 rather than emitting
`{"type":"function","function":{"name":"web_search"}}` with no parameters. Today
that degenerate tool becomes either a provider 400, a misleading Vekil 400
naming `tools[N].parameters` (`proxy/policy_responses_translate.go:880`), or —
via the chat-over-Responses bridge which backfills the schema
(`proxy/chat_over_responses_request.go:882-885`) — a callable tool nothing will
ever answer. No current outcome works, so rejecting loses nothing. Anthropic's
own org-level kill switch returns a 400 `invalid_request_error`, which is the
precedent for the shape.

The count_tokens probe is the exception: it substitutes the stand-in function
tool for hosted `web_search_*` instead of rejecting, and rejects other hosted
types as normal. Counting the tool the model will actually be given is the
meaningful number, and it keeps count_tokens working for clients that send
hosted web search.

**Error envelope.** On the direct path, parse non-200 upstream bodies: relay
verbatim when already Anthropic-shaped, otherwise wrap into
`{"type":"error","error":{...}}` preserving status and message. Today
`writeUpstreamResponse` copies them verbatim (`proxy/chat_handlers.go:1701`,
`proxy/upstream_http.go:563-572`), so a Copilot-shaped body reaches an
Anthropic-protocol client. Justified on its own terms: `/v1/messages` clients
are entitled to the Anthropic error shape regardless of which provider served
the request.

## Testing

New `proxy/anthropic_web_search_test.go` modeled on
`proxy/anthropic_policy_ingress_test.go`, with a fake upstream serving both
`/v1/messages` and `/responses`. TDD; each case lands red first.

- disabled → bytes forwarded unchanged (assert exact body)
- enabled, no hosted tool → unchanged
- **anthropic-compatible provider → no mediation**, hosted tool forwarded intact
- **delegate discovery never selects a model on another provider**
- happy path: one search, correct block shapes and ids
- loop bounded by the lower of `max_uses` and `max_searches`
- **omitted `max_uses` → bounded by `max_searches` alone**
- **exhaustion → `web_search_tool_result_error` with `max_uses_exceeded`,
  `content` as a single object not a list**
- **continuation dispatches do not consume the client's `MaxUpstreamSends`**
- **malformed or forged `encrypted_content` → fails closed to an error block**,
  never reaching the model
- mixed turn: search resolved inline, client `tool_use` preserved,
  `stop_reason: tool_use`
- delegation failure → original body reaches upstream, feature invisible
- `blocked_domains` result dropped even when upstream returns it
- client tool named `web_search` → passthrough
- streaming replay: event order, and `web_search_tool_result` whole inside
  `content_block_start`
- second turn: synthesized blocks decode back correctly
- usage: `web_search_requests` counts delegated calls; delegated tokens **and
  continuation-turn tokens** land in internal usage, not the turn's

Plus `proxy/providers_config_test.go` for strict-decode and defaults, mirroring
the `tool_optimizer_config` default tests.

## Documentation

New `docs/web-search.md` following `docs/tool-optimizers.md` — optional
top-level block, disabled by default, YAML examples, defaults table. Rows added
to `docs/README.md`, `docs/configuration.md` (topic map, provider-configs
bullet), `docs/api.md`, `docs/provider-routing.md`, `docs/development.md` (test
matrix), and the root `README.md`.

## Risks

- **Delegation quality is unevaluated at breadth.** Probes covered a handful of
  factual lookups. Ambiguous and recency-sensitive queries, where a single
  unrefined search is weakest, are untested.
- **`blocked_domains` upstream enforcement is unverified.** Mitigated by
  post-filtering, which is authoritative regardless.
- **Quota.** Each delegated search is an extra Copilot request; search-heavy
  sessions consume more.
- **Latency.** Roughly 15s of silence for a one-search turn, ~35s for three.
  This is the deliberate cost of avoiding a live-stream multiplexer.
- **Results are a model's synthesis**, not raw rankings; `page_age` and ordering
  are approximations.
- **Copilot could start rejecting hosted `web_search` on `/responses`** as it
  already does on `/v1/messages`. The feature would fail open to passthrough,
  but the capability would be gone.

## Excluded

- Live-stream multiplexing (search turns are non-streaming upstream).
- Any change to the chat executor seam, the Responses replay store, or policy
  routing.
- External search backends (Brave, Tavily, SearXNG) and routing to a real
  Anthropic provider. Both were considered and are recorded in issue #341.
