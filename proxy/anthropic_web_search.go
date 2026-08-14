package proxy

import (
	"encoding/json"
	"sort"
	"strings"

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
