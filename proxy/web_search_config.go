package proxy

import "strings"

const (
	defaultWebSearchMaxSearches = 5
	defaultWebSearchTimeoutMS   = 30000
	defaultWebSearchContextSize = "low"
	defaultWebSearchMaxResults  = 10
)

// WebSearchConfig configures proxy-mediated Anthropic web_search. The feature is
// intentionally disabled by default; with Enabled false the /v1/messages direct
// path behaves exactly as it does without this block.
type WebSearchConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
	// DelegateModel pins the model used for the internal /responses search call.
	// Empty auto-discovers the first model on the conversation's own provider
	// that advertises /responses.
	DelegateModel string `json:"delegate_model,omitempty" yaml:"delegate_model,omitempty"`
	// MaxSearches is the per-turn ceiling, min'd with the client's max_uses.
	MaxSearches int `json:"max_searches,omitempty" yaml:"max_searches,omitempty"`
	// TimeoutMS bounds one delegated search call. The loop as a whole is bounded
	// by the inherited non-streaming upstream deadline.
	TimeoutMS int `json:"timeout_ms,omitempty" yaml:"timeout_ms,omitempty"`
	// SearchContextSize is the upstream search_context_size tier: low, medium or
	// high. Low is the default because probes showed larger tiers roughly
	// quadruple latency.
	SearchContextSize string `json:"search_context_size,omitempty" yaml:"search_context_size,omitempty"`
	// MaxResults caps how many results a delegated search returns.
	MaxResults int `json:"max_results,omitempty" yaml:"max_results,omitempty"`
}

func defaultWebSearchConfig() WebSearchConfig {
	return WebSearchConfig{
		Enabled:           false,
		DelegateModel:     "",
		MaxSearches:       defaultWebSearchMaxSearches,
		TimeoutMS:         defaultWebSearchTimeoutMS,
		SearchContextSize: defaultWebSearchContextSize,
		MaxResults:        defaultWebSearchMaxResults,
	}
}

// withDefaults fills unset fields from defaultWebSearchConfig. It is idempotent:
// applying it to an already-defaulted config returns an identical value, so it
// is safe at both config load and handler construction.
func (c WebSearchConfig) withDefaults() WebSearchConfig {
	defaults := defaultWebSearchConfig()
	defaults.Enabled = c.Enabled
	if delegate := strings.TrimSpace(c.DelegateModel); delegate != "" {
		defaults.DelegateModel = delegate
	}
	if c.MaxSearches > 0 {
		defaults.MaxSearches = c.MaxSearches
	}
	if c.TimeoutMS > 0 {
		defaults.TimeoutMS = c.TimeoutMS
	}
	if size := strings.ToLower(strings.TrimSpace(c.SearchContextSize)); size != "" {
		defaults.SearchContextSize = size
	}
	if c.MaxResults > 0 {
		defaults.MaxResults = c.MaxResults
	}
	return defaults
}

// initializeWebSearch resolves the effective web_search configuration once at
// handler construction, mirroring initializeToolOptimizers.
func (h *ProxyHandler) initializeWebSearch() {
	if h == nil {
		return
	}
	h.webSearch = h.providersConfig.WebSearch.withDefaults()
}
