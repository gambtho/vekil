package proxy

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/sozercan/vekil/models"
)

const (
	// anthropicWebSearchToolName is the tool name Anthropic uses for both the
	// hosted server tool and the client-side stand-in the proxy substitutes for
	// it when mediating searches.
	anthropicWebSearchToolName = "web_search"

	// anthropicHostedWebSearchTypePrefix matches Anthropic's versioned hosted
	// search tool types (web_search_20250305 and successors). The bare name
	// "web_search" is deliberately not a hosted type: an unversioned type is not
	// a shape Anthropic emits, and treating it as hosted would misclassify a
	// client tool.
	anthropicHostedWebSearchTypePrefix = "web_search_"

	// anthropicClientToolType is the explicit type Anthropic allows on a
	// client-defined tool. An omitted type means the same thing.
	anthropicClientToolType = "custom"
)

// hostedWebSearchTool returns the first hosted web-search tool in tools along
// with its index, or false when the request carries none.
func hostedWebSearchTool(tools []models.AnthropicTool) (models.AnthropicTool, int, bool) {
	for index, tool := range tools {
		if strings.HasPrefix(strings.TrimSpace(tool.Type), anthropicHostedWebSearchTypePrefix) {
			return tool, index, true
		}
	}
	return models.AnthropicTool{}, -1, false
}

// clientDefinesWebSearchTool reports whether the client declared its own tool
// named web_search. Mediation declines in that case rather than inventing a
// renaming scheme to disambiguate the intercepted call. Client tools are those
// isAnthropicClientTool accepts — no type, or an explicit "custom" — so the two
// functions must agree on what a client tool is; a request carrying both a
// hosted web_search and a "custom"-typed web_search would otherwise slip past
// this guard and enter mediation with two identically named tools.
func clientDefinesWebSearchTool(tools []models.AnthropicTool) bool {
	for _, tool := range tools {
		if !isAnthropicClientTool(tool) {
			continue
		}
		if tool.Name == anthropicWebSearchToolName {
			return true
		}
	}
	return false
}

// isAnthropicClientTool reports whether a tool is one the client itself will
// answer. Anthropic marks those with no type or with "custom"; every other type
// names a hosted server tool that the upstream, not the client, executes.
func isAnthropicClientTool(tool models.AnthropicTool) bool {
	toolType := strings.TrimSpace(tool.Type)
	return toolType == "" || strings.EqualFold(toolType, anthropicClientToolType)
}

// webSearchStandInInputSchema is the client-tool schema that replaces
// Anthropic's hosted web_search tool. Mediation sends exactly this shape
// upstream, so anything that reasons about the mediated request — token
// counting included — must use exactly this shape too.
const webSearchStandInInputSchema = `{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`

// webSearchStandInTool is the client tool the proxy substitutes for Anthropic's
// hosted web_search server tool.
func webSearchStandInTool() models.AnthropicTool {
	return models.AnthropicTool{
		Name:        anthropicWebSearchToolName,
		InputSchema: json.RawMessage(webSearchStandInInputSchema),
	}
}

// translateAnthropicToolsForTokenCount replaces hosted web_search tools with the
// stand-in the proxy actually sends upstream, so /v1/messages/count_tokens keeps
// working for clients that declare the hosted tool and reports the count for the
// tool the model will really be given. Other hosted types have no stand-in and
// are left in place for TranslateAnthropicToOpenAI to reject. The input slice is
// never mutated; the second result reports whether a copy was made.
func translateAnthropicToolsForTokenCount(tools []models.AnthropicTool) ([]models.AnthropicTool, bool) {
	out := tools
	substituted := false
	for index, tool := range tools {
		if !strings.HasPrefix(strings.TrimSpace(tool.Type), anthropicHostedWebSearchTypePrefix) {
			continue
		}
		if !substituted {
			out = make([]models.AnthropicTool, len(tools))
			copy(out, tools)
			substituted = true
		}
		out[index] = webSearchStandInTool()
	}
	return out, substituted
}

// webSearchMediation is the resolved, immutable decision to mediate one
// Anthropic /v1/messages turn. Everything the client asked for that must not
// reach the upstream (max_uses and the domain/location filters) lives on tool
// and is applied proxy-side instead.
type webSearchMediation struct {
	cfg         WebSearchConfig
	tool        models.AnthropicTool
	delegate    providerModel
	provider    *providerRuntime
	maxSearches int
	// summary counts physical dispatches so upstream_sends stays honest;
	// singleInferenceSend bypasses the route executor's RecordUpstreamAttempt.
	summary *RequestSummary
	// delegatedUsage accumulates internal Responses spend across the turn so
	// Task 12 can flush it additively instead of clobbering the turn's usage.
	delegatedUsage responsesUsage
}

// webSearchMediationFor decides whether this request is eligible for
// proxy-mediated web search. Every miss is a silent decline: the caller keeps
// today's byte-for-byte passthrough.
func (h *ProxyHandler) webSearchMediationFor(req *models.AnthropicRequest) (*webSearchMediation, bool) {
	if h == nil || req == nil || !h.webSearch.Enabled {
		return nil, false
	}
	tool, _, ok := hostedWebSearchTool(req.Tools)
	if !ok {
		return nil, false
	}
	// A client tool of the same name would make the intercepted call ambiguous.
	if clientDefinesWebSearchTool(req.Tools) {
		return nil, false
	}
	provider, _, known := h.resolveProviderModelForRequest(req.Model, providerEndpointMessages)
	// Hard exclusion, not an optimization: providerTypeAnthropicCompatible also
	// takes the direct path and already serves hosted web_search natively.
	// Mediating there would replace a working path with an approximation.
	if provider == nil || !known || provider.kind != providerTypeCopilot {
		return nil, false
	}
	delegate, ok := h.webSearchDelegateModel(provider)
	if !ok {
		return nil, false
	}

	maxSearches := h.webSearch.MaxSearches
	if tool.MaxUses != nil && *tool.MaxUses < maxSearches {
		maxSearches = *tool.MaxUses
	}
	if maxSearches <= 0 {
		return nil, false
	}
	return &webSearchMediation{
		cfg:         h.webSearch,
		tool:        tool,
		delegate:    delegate,
		provider:    provider,
		maxSearches: maxSearches,
	}, true
}

// webSearchDelegateModel picks the model that runs the hosted search. It is
// constrained to provider — a privacy boundary, not just a correctness one:
// search queries must not leave the provider the client is already talking to.
func (h *ProxyHandler) webSearchDelegateModel(provider *providerRuntime) (providerModel, bool) {
	if h == nil || provider == nil {
		return providerModel{}, false
	}
	candidates := h.providerSetup().modelsForProvider(provider.id)
	// modelsForProvider walks a map, so order is not stable across calls.
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].publicID < candidates[j].publicID })

	if configured := strings.TrimSpace(h.webSearch.DelegateModel); configured != "" {
		for _, candidate := range candidates {
			if candidate.publicID != configured || candidate.disabled {
				continue
			}
			// A pinned delegate must advertise /responses just like a discovered
			// one. Without this check a Messages-only pin activates mediation and
			// then fails every search, costing a wasted upstream call per turn
			// with no error explaining why.
			if !supportsEndpoint(candidate.supportedEndpoints, providerEndpointResponses) {
				return providerModel{}, false
			}
			return candidate, true
		}
		// A configured delegate owned by another provider is a decline, never a
		// cross-provider fallback.
		return providerModel{}, false
	}
	for _, candidate := range candidates {
		if candidate.disabled {
			continue
		}
		// Explicit advertisement only: supportsEndpoint is false for an empty
		// endpoint list, so unknown models are never guessed into delegation.
		if supportsEndpoint(candidate.supportedEndpoints, providerEndpointResponses) {
			return candidate, true
		}
	}
	return providerModel{}, false
}

// anthropicWebSearchClientToolJSON is the degenerate client tool the upstream
// sees in place of the hosted entry. It carries no filters: max_uses,
// allowed_domains, blocked_domains and user_location are held proxy-side and
// applied to the results instead.
const anthropicWebSearchClientToolJSON = `{"name":"web_search","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}`

// rewriteAnthropicWebSearchRequest swaps the hosted tool for a plain client
// tool and forces non-streaming, operating on the raw body so unknown
// top-level and per-tool fields survive untouched. Key order within the object
// is not preserved (encoding/json sorts map keys); field content is.
func rewriteAnthropicWebSearchRequest(body []byte, m *webSearchMediation) ([]byte, error) {
	if m == nil {
		return nil, fmt.Errorf("web search mediation is required")
	}
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("decode messages request: %w", err)
	}
	rawTools, ok := request["tools"]
	if !ok {
		return nil, fmt.Errorf("messages request carries no tools")
	}
	var rawToolEntries []json.RawMessage
	if err := json.Unmarshal(rawTools, &rawToolEntries); err != nil {
		return nil, fmt.Errorf("decode tools: %w", err)
	}
	// Re-derive the index from the raw array with the same detector used at
	// activation, so detection has exactly one implementation.
	decodedTools := make([]models.AnthropicTool, len(rawToolEntries))
	if err := json.Unmarshal(rawTools, &decodedTools); err != nil {
		return nil, fmt.Errorf("decode tools: %w", err)
	}
	_, index, found := hostedWebSearchTool(decodedTools)
	if !found || index < 0 || index >= len(rawToolEntries) {
		return nil, fmt.Errorf("messages request carries no hosted %s tool", anthropicWebSearchToolName)
	}

	rewrittenTools := make([]json.RawMessage, len(rawToolEntries))
	copy(rewrittenTools, rawToolEntries)
	rewrittenTools[index] = json.RawMessage(anthropicWebSearchClientToolJSON)

	encodedTools, err := json.Marshal(rewrittenTools)
	if err != nil {
		return nil, fmt.Errorf("encode tools: %w", err)
	}
	request["tools"] = encodedTools
	// The client's own stream value is remembered by the caller for emission;
	// upstream is always non-streaming so fallback stays available.
	request["stream"] = json.RawMessage("false")

	rewritten, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode messages request: %w", err)
	}
	return rewritten, nil
}

// webSearchMaxResponseBytes bounds a delegated /responses body, mirroring the
// policy classifier's policyClassifierResponseLimit (proxy/policy_routing.go:18)
// while staying independently tunable.
const webSearchMaxResponseBytes = 64 << 10

// webSearchDelegateInstruction is load-bearing, not decoration: without the
// single-search and no-refinement constraints one delegated call fans out to a
// dozen internal searches and latency roughly quadruples.
const webSearchDelegateInstruction = "You are a search executor, not an assistant. " +
	"Run exactly ONE web search for the user's query. Do not refine, retry, or run additional searches. " +
	"Reply with ONLY a JSON array of objects with the keys \"url\", \"title\", \"snippet\" and \"page_age\". " +
	"Use an empty string for any field you do not know. " +
	"No prose, no explanation, no markdown fences, no other keys."

// webSearchResult is one post-filtered search hit.
type webSearchResult struct {
	URL     string `json:"url"`
	Title   string `json:"title"`
	Snippet string `json:"snippet"`
	PageAge string `json:"page_age"`
}

// delegateWebSearch performs one internal /responses call on the same provider
// that owns the conversation. It dispatches through singleInferenceSend so a
// delegation failure cannot consume the client's route failover budget. Any
// error means the caller falls back to plain passthrough.
func (h *ProxyHandler) delegateWebSearch(ctx context.Context, m *webSearchMediation, query string) ([]webSearchResult, error) {
	if h == nil || m == nil || m.provider == nil {
		return nil, fmt.Errorf("web search delegation requires a resolved mediation")
	}
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Errorf("web search query is empty")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if m.cfg.TimeoutMS > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(m.cfg.TimeoutMS)*time.Millisecond)
		defer cancel()
	}

	body, err := buildWebSearchDelegateRequest(m, query)
	if err != nil {
		return nil, err
	}
	req, err := h.newProviderJSONRequest(ctx, m.provider, http.MethodPost, providerEndpointResponses, body, http.Header{"Content-Type": []string{"application/json"}}, "", m.delegate)
	if err != nil {
		return nil, fmt.Errorf("build web search delegate request: %w", err)
	}
	resp, err := h.singleInferenceSend(req, newRouteSendObservation(time.Now(), nil))
	if err != nil {
		return nil, fmt.Errorf("web search delegate send: %w", err)
	}
	// This dispatch never passes through the route executor, so nothing else
	// counts it. Record it here, immediately: singleInferenceSend has already
	// performed the HTTP request by the time it returns a response, and the
	// counter represents physical sends. Recording it further down would drop
	// non-200s, oversized bodies and read failures from upstream_sends —
	// precisely the traffic someone debugging this would be looking for.
	m.summary.RecordUpstreamSend()
	defer drainAndClose(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("web search delegate returned HTTP %d", resp.StatusCode)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, webSearchMaxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read web search delegate response: %w", err)
	}
	if len(raw) > webSearchMaxResponseBytes {
		return nil, fmt.Errorf("web search delegate response exceeds %d bytes", webSearchMaxResponseBytes)
	}

	m.delegatedUsage.add(parseWebSearchDelegateUsage(raw))

	results, err := parseWebSearchResults(webSearchDelegateOutputText(raw))
	if err != nil {
		return nil, err
	}
	// Post-filtering is authoritative regardless of the filters sent upstream:
	// blocked_domains enforcement on the delegated side is unverified.
	return filterWebSearchResults(results, m.tool, m.cfg.MaxResults), nil
}

func parseWebSearchDelegateUsage(body []byte) responsesUsage {
	var envelope struct {
		Usage responsesUsage `json:"usage"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return responsesUsage{}
	}
	return envelope.Usage
}

func buildWebSearchDelegateRequest(m *webSearchMediation, query string) ([]byte, error) {
	upstreamModel := strings.TrimSpace(m.delegate.upstreamModel)
	if upstreamModel == "" {
		upstreamModel = strings.TrimSpace(m.delegate.publicID)
	}
	if upstreamModel == "" {
		return nil, fmt.Errorf("web search delegate has no upstream model")
	}

	tool := map[string]any{"type": anthropicWebSearchToolName}
	if size := strings.TrimSpace(m.cfg.SearchContextSize); size != "" {
		tool["search_context_size"] = size
	}
	filters := map[string]any{}
	if len(m.tool.AllowedDomains) > 0 {
		filters["allowed_domains"] = m.tool.AllowedDomains
	}
	if len(m.tool.BlockedDomains) > 0 {
		filters["blocked_domains"] = m.tool.BlockedDomains
	}
	if len(filters) > 0 {
		tool["filters"] = filters
	}
	if len(m.tool.UserLocation) > 0 {
		tool["user_location"] = m.tool.UserLocation
	}

	request := map[string]any{
		"model":     upstreamModel,
		"stream":    false,
		"store":     false,
		"reasoning": map[string]any{"effort": "low"},
		"tools":     []any{tool},
		"input": []any{
			map[string]any{"role": "developer", "content": webSearchDelegateInstruction},
			map[string]any{"role": "user", "content": query},
		},
	}
	encoded, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode web search delegate request: %w", err)
	}
	return encoded, nil
}

func webSearchDelegateOutputText(body []byte) string {
	var envelope struct {
		OutputText string `json:"output_text"`
		Output     []struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ""
	}
	if strings.TrimSpace(envelope.OutputText) != "" {
		return envelope.OutputText
	}
	var text strings.Builder
	for _, item := range envelope.Output {
		for _, part := range item.Content {
			if part.Type == "output_text" {
				text.WriteString(part.Text)
			}
		}
	}
	return text.String()
}

func parseWebSearchResults(text string) ([]webSearchResult, error) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, fmt.Errorf("web search delegate returned no output text")
	}
	// Tolerate a fenced block even though the prompt forbids it.
	if strings.HasPrefix(trimmed, "```") {
		trimmed = strings.TrimPrefix(trimmed, "```")
		trimmed = strings.TrimPrefix(trimmed, "json")
		if end := strings.LastIndex(trimmed, "```"); end >= 0 {
			trimmed = trimmed[:end]
		}
		trimmed = strings.TrimSpace(trimmed)
	}
	var results []webSearchResult
	if err := json.Unmarshal([]byte(trimmed), &results); err != nil {
		return nil, fmt.Errorf("web search delegate output is not a JSON result array: %w", err)
	}
	return results, nil
}

// filterWebSearchResults enforces the client's domain filters proxy-side and
// truncates to maxResults. It is the authoritative filter regardless of what
// was sent upstream.
func filterWebSearchResults(results []webSearchResult, tool models.AnthropicTool, maxResults int) []webSearchResult {
	filtered := make([]webSearchResult, 0, len(results))
	for _, result := range results {
		host := webSearchResultHost(result.URL)
		if host == "" {
			continue
		}
		if webSearchHostMatchesAny(host, tool.BlockedDomains) {
			continue
		}
		if len(tool.AllowedDomains) > 0 && !webSearchHostMatchesAny(host, tool.AllowedDomains) {
			continue
		}
		filtered = append(filtered, result)
		if maxResults > 0 && len(filtered) >= maxResults {
			break
		}
	}
	return filtered
}

const (
	// webSearchContentVersion prefixes every minted encrypted_content token,
	// mirroring encodeSyntheticCompaction (proxy/compaction.go:19-25).
	webSearchContentVersion = "vkws1:"
	// webSearchMaxContentBytes caps a minted token and rejects an oversize one
	// on the way back in.
	webSearchMaxContentBytes = 8192
	webSearchCallIDPrefix    = "srvtoolu_"
)

type webSearchResultItem struct {
	Type             string `json:"type"`
	URL              string `json:"url"`
	Title            string `json:"title"`
	EncryptedContent string `json:"encrypted_content"`
	PageAge          string `json:"page_age"`
}

type webSearchResultErrorContent struct {
	Type      string `json:"type"`
	ErrorCode string `json:"error_code"`
}

func newWebSearchCallID() string {
	var raw [16]byte
	if _, err := rand.Read(raw[:]); err != nil {
		// crypto/rand failure is fatal for uniqueness; fall back to a
		// time-derived id rather than emitting a colliding constant.
		binary.BigEndian.PutUint64(raw[0:8], uint64(time.Now().UnixNano()))
		binary.BigEndian.PutUint64(raw[8:16], uint64(time.Now().UnixNano())^0x9e3779b97f4a7c15)
	}
	// 16 random bytes encode to exactly 22 base64url characters.
	return webSearchCallIDPrefix + base64.RawURLEncoding.EncodeToString(raw[:])
}

// encodeWebSearchContent mints the stateless token handed to the client as
// encrypted_content. It is deliberately NOT authenticated encryption: the
// client already controls the whole message history, so forging one grants no
// capability it lacks. What matters is that decode is strict.
func encodeWebSearchContent(r webSearchResult) string {
	encoded := encodeWebSearchContentOnce(r)
	for len(encoded) > webSearchMaxContentBytes && len(r.Snippet) > 0 {
		r.Snippet = truncateUTF8Prefix(r.Snippet, len(r.Snippet)/2)
		encoded = encodeWebSearchContentOnce(r)
	}
	if len(encoded) > webSearchMaxContentBytes {
		// Identity fields alone still exceed the cap; drop the payload rather
		// than mint a token that will fail closed on the way back.
		return ""
	}
	return encoded
}

func encodeWebSearchContentOnce(r webSearchResult) string {
	payload, err := json.Marshal(r)
	if err != nil {
		return ""
	}
	return webSearchContentVersion + base64.RawURLEncoding.EncodeToString(payload)
}

// decodeWebSearchContent fails closed. A wrong or missing version prefix, an
// oversize token, bad base64 or bad JSON all return ok=false so malformed or
// forged content is never passed through to the model.
func decodeWebSearchContent(encoded string) (webSearchResult, bool) {
	if len(encoded) > webSearchMaxContentBytes {
		return webSearchResult{}, false
	}
	if !strings.HasPrefix(encoded, webSearchContentVersion) {
		return webSearchResult{}, false
	}
	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(encoded, webSearchContentVersion))
	if err != nil {
		return webSearchResult{}, false
	}
	var result webSearchResult
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&result); err != nil {
		return webSearchResult{}, false
	}
	if decoder.More() {
		return webSearchResult{}, false
	}
	return result, true
}

func truncateUTF8Prefix(s string, limit int) string {
	if limit <= 0 {
		return ""
	}
	if limit >= len(s) {
		return s
	}
	for limit > 0 && !utf8.RuneStart(s[limit]) {
		limit--
	}
	return s[:limit]
}

// synthesizeWebSearchBlocks builds the pair Anthropic clients expect for a
// completed server tool call. The success content is a JSON list; the error
// form (webSearchErrorResultBlock) is a single object. That distinction is
// load-bearing and must not be conflated.
func synthesizeWebSearchBlocks(callID, query string, results []webSearchResult) (models.ContentBlock, models.ContentBlock) {
	input, err := json.Marshal(map[string]string{"query": query})
	if err != nil {
		input = []byte(`{"query":""}`)
	}
	items := make([]webSearchResultItem, 0, len(results))
	for _, result := range results {
		items = append(items, webSearchResultItem{
			Type:             "web_search_result",
			URL:              result.URL,
			Title:            result.Title,
			EncryptedContent: encodeWebSearchContent(result),
			PageAge:          result.PageAge,
		})
	}
	content, err := json.Marshal(items)
	if err != nil {
		content = []byte("[]")
	}
	use := models.ContentBlock{
		Type:  "server_tool_use",
		ID:    callID,
		Name:  anthropicWebSearchToolName,
		Input: input,
	}
	result := models.ContentBlock{
		Type:      "web_search_tool_result",
		ToolUseID: callID,
		Content:   content,
	}
	return use, result
}

func webSearchErrorResultBlock(callID, errorCode string) models.ContentBlock {
	content, err := json.Marshal(webSearchResultErrorContent{
		Type:      "web_search_tool_result_error",
		ErrorCode: errorCode,
	})
	if err != nil {
		content = []byte(`{"type":"web_search_tool_result_error","error_code":"unavailable"}`)
	}
	return models.ContentBlock{
		Type:      "web_search_tool_result",
		ToolUseID: callID,
		Content:   content,
	}
}

func webSearchResultHost(rawURL string) string {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Scheme == "" {
		return ""
	}
	// A trailing root-label dot (e.g. "spam.test.") is DNS-equivalent to
	// "spam.test" and fully resolvable as such, so it must not evade
	// blocked_domains or fail allowed_domains just by comparing unequal
	// strings.
	return strings.TrimSuffix(strings.ToLower(parsed.Hostname()), ".")
}

func webSearchHostMatchesAny(host string, domains []string) bool {
	for _, domain := range domains {
		candidate := strings.ToLower(strings.TrimSpace(domain))
		candidate = strings.TrimPrefix(candidate, "*.")
		candidate = strings.TrimPrefix(candidate, ".")
		// Normalize a trailing dot on the configured domain too, so a
		// blocked_domains/allowed_domains entry authored with a root-label
		// dot still compares equal to a host without one.
		candidate = strings.TrimSuffix(candidate, ".")
		if candidate == "" {
			continue
		}
		if host == candidate || strings.HasSuffix(host, "."+candidate) {
			return true
		}
	}
	return false
}
