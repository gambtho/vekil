package proxy

import (
	"encoding/json"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestWebSearchConfigDefaultsWhenUnset(t *testing.T) {
	t.Parallel()

	got := (WebSearchConfig{}).withDefaults()
	if got.Enabled {
		t.Fatalf("enabled: got=true, want=false")
	}
	if got.DelegateModel != "" {
		t.Fatalf("delegate_model: got=%q, want empty", got.DelegateModel)
	}
	if got.MaxSearches != defaultWebSearchMaxSearches {
		t.Fatalf("max_searches: got=%d, want=%d", got.MaxSearches, defaultWebSearchMaxSearches)
	}
	if got.TimeoutMS != defaultWebSearchTimeoutMS {
		t.Fatalf("timeout_ms: got=%d, want=%d", got.TimeoutMS, defaultWebSearchTimeoutMS)
	}
	if got.SearchContextSize != defaultWebSearchContextSize {
		t.Fatalf("search_context_size: got=%q, want=%q", got.SearchContextSize, defaultWebSearchContextSize)
	}
	if got.MaxResults != defaultWebSearchMaxResults {
		t.Fatalf("max_results: got=%d, want=%d", got.MaxResults, defaultWebSearchMaxResults)
	}
}

func TestWebSearchConfigWithDefaultsIsIdempotent(t *testing.T) {
	t.Parallel()

	first := WebSearchConfig{
		Enabled:           true,
		DelegateModel:     "  gpt-5.4  ",
		MaxSearches:       2,
		TimeoutMS:         1500,
		SearchContextSize: "  HIGH  ",
		MaxResults:        3,
	}.withDefaults()
	second := first.withDefaults()

	if first != second {
		t.Fatalf("withDefaults not idempotent: got=%+v, want=%+v", second, first)
	}
	if !first.Enabled {
		t.Fatalf("enabled: got=false, want=true")
	}
	if first.DelegateModel != "gpt-5.4" {
		t.Fatalf("delegate_model: got=%q, want=%q", first.DelegateModel, "gpt-5.4")
	}
	if first.MaxSearches != 2 {
		t.Fatalf("max_searches: got=%d, want=2", first.MaxSearches)
	}
	if first.TimeoutMS != 1500 {
		t.Fatalf("timeout_ms: got=%d, want=1500", first.TimeoutMS)
	}
	if first.SearchContextSize != "high" {
		t.Fatalf("search_context_size: got=%q, want=%q", first.SearchContextSize, "high")
	}
	if first.MaxResults != 3 {
		t.Fatalf("max_results: got=%d, want=3", first.MaxResults)
	}
}

func TestWebSearchConfigNonPositiveValuesFallBackToDefaults(t *testing.T) {
	t.Parallel()

	got := WebSearchConfig{
		Enabled:     true,
		MaxSearches: 0,
		TimeoutMS:   -1,
		MaxResults:  0,
	}.withDefaults()

	if got.MaxSearches != defaultWebSearchMaxSearches {
		t.Fatalf("max_searches: got=%d, want=%d", got.MaxSearches, defaultWebSearchMaxSearches)
	}
	if got.TimeoutMS != defaultWebSearchTimeoutMS {
		t.Fatalf("timeout_ms: got=%d, want=%d", got.TimeoutMS, defaultWebSearchTimeoutMS)
	}
	if got.MaxResults != defaultWebSearchMaxResults {
		t.Fatalf("max_results: got=%d, want=%d", got.MaxResults, defaultWebSearchMaxResults)
	}
}

func TestWebSearchConfigDecodesFromProvidersConfig(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name      string
		body      string
		unmarshal func([]byte, interface{}) error
	}{
		{
			name: "yaml",
			body: `
web_search:
  enabled: true
  delegate_model: gpt-5.4
  max_searches: 2
  timeout_ms: 15000
  search_context_size: medium
  max_results: 4
`,
			unmarshal: yaml.Unmarshal,
		},
		{
			name: "json",
			body: `{
  "web_search": {
    "enabled": true,
    "delegate_model": "gpt-5.4",
    "max_searches": 2,
    "timeout_ms": 15000,
    "search_context_size": "medium",
    "max_results": 4
  }
}`,
			unmarshal: json.Unmarshal,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var cfg ProvidersConfig
			if err := tc.unmarshal([]byte(tc.body), &cfg); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.name, err)
			}

			got := cfg.WebSearch.withDefaults()
			if !got.Enabled {
				t.Fatalf("web_search.enabled: got=false, want=true")
			}
			if got.DelegateModel != "gpt-5.4" {
				t.Fatalf("delegate_model: got=%q, want=%q", got.DelegateModel, "gpt-5.4")
			}
			if got.MaxSearches != 2 {
				t.Fatalf("max_searches: got=%d, want=2", got.MaxSearches)
			}
			if got.TimeoutMS != 15000 {
				t.Fatalf("timeout_ms: got=%d, want=15000", got.TimeoutMS)
			}
			if got.SearchContextSize != "medium" {
				t.Fatalf("search_context_size: got=%q, want=%q", got.SearchContextSize, "medium")
			}
			if got.MaxResults != 4 {
				t.Fatalf("max_results: got=%d, want=4", got.MaxResults)
			}
		})
	}
}

func TestInitializeWebSearchAppliesDefaults(t *testing.T) {
	t.Parallel()

	h := &ProxyHandler{providersConfig: ProvidersConfig{
		WebSearch: WebSearchConfig{Enabled: true, MaxSearches: 2},
	}}
	h.initializeWebSearch()

	if !h.webSearch.Enabled {
		t.Fatalf("webSearch.Enabled: got=false, want=true")
	}
	if h.webSearch.MaxSearches != 2 {
		t.Fatalf("webSearch.MaxSearches: got=%d, want=2", h.webSearch.MaxSearches)
	}
	if h.webSearch.TimeoutMS != defaultWebSearchTimeoutMS {
		t.Fatalf("webSearch.TimeoutMS: got=%d, want=%d", h.webSearch.TimeoutMS, defaultWebSearchTimeoutMS)
	}
	if h.webSearch.SearchContextSize != defaultWebSearchContextSize {
		t.Fatalf("webSearch.SearchContextSize: got=%q, want=%q",
			h.webSearch.SearchContextSize, defaultWebSearchContextSize)
	}

	// The zero-value handler must stay disabled with usable bounds.
	disabled := &ProxyHandler{}
	disabled.initializeWebSearch()
	if disabled.webSearch.Enabled {
		t.Fatalf("zero-value webSearch.Enabled: got=true, want=false")
	}
	if disabled.webSearch.MaxSearches != defaultWebSearchMaxSearches {
		t.Fatalf("zero-value webSearch.MaxSearches: got=%d, want=%d",
			disabled.webSearch.MaxSearches, defaultWebSearchMaxSearches)
	}
}
