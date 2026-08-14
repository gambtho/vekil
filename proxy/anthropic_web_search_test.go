package proxy

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sozercan/vekil/auth"
	"github.com/sozercan/vekil/logger"
	"github.com/sozercan/vekil/models"
)

func TestAnthropicRequestDecodesHostedWebSearchToolFields(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"model": "claude-sonnet-4.5",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [
			{"name": "Bash", "input_schema": {"type": "object"}},
			{
				"type": "web_search_20250305",
				"name": "web_search",
				"max_uses": 3,
				"allowed_domains": ["example.com"],
				"blocked_domains": ["spam.example"],
				"user_location": {"type": "approximate", "city": "Seattle"}
			}
		]
	}`)

	var req models.AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal anthropic request: %v", err)
	}
	if len(req.Tools) != 2 {
		t.Fatalf("tool count: got=%d, want=2", len(req.Tools))
	}

	hosted := req.Tools[1]
	if hosted.Type != "web_search_20250305" {
		t.Fatalf("tools[1].type: got=%q, want=%q", hosted.Type, "web_search_20250305")
	}
	if hosted.MaxUses == nil || *hosted.MaxUses != 3 {
		t.Fatalf("tools[1].max_uses: got=%v, want=3", hosted.MaxUses)
	}
	if len(hosted.AllowedDomains) != 1 || hosted.AllowedDomains[0] != "example.com" {
		t.Fatalf("tools[1].allowed_domains: got=%v, want=[example.com]", hosted.AllowedDomains)
	}
	if len(hosted.BlockedDomains) != 1 || hosted.BlockedDomains[0] != "spam.example" {
		t.Fatalf("tools[1].blocked_domains: got=%v, want=[spam.example]", hosted.BlockedDomains)
	}
	if len(hosted.UserLocation) == 0 {
		t.Fatalf("tools[1].user_location: got empty, want raw JSON object")
	}

	// A client tool must keep decoding exactly as before: no type, schema intact.
	client := req.Tools[0]
	if client.Type != "" {
		t.Fatalf("tools[0].type: got=%q, want empty", client.Type)
	}
	if client.MaxUses != nil {
		t.Fatalf("tools[0].max_uses: got=%v, want=nil", client.MaxUses)
	}
}

func TestAnthropicToolOmitsHostedFieldsWhenUnset(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(models.AnthropicTool{
		Name:        "Bash",
		InputSchema: json.RawMessage(`{"type":"object"}`),
	})
	if err != nil {
		t.Fatalf("marshal anthropic tool: %v", err)
	}
	want := `{"name":"Bash","input_schema":{"type":"object"}}`
	if string(encoded) != want {
		t.Fatalf("encoded tool: got=%s, want=%s", encoded, want)
	}
}

func TestAnthropicUsageServerToolUseRoundTrip(t *testing.T) {
	t.Parallel()

	encoded, err := json.Marshal(models.AnthropicUsage{InputTokens: 5, OutputTokens: 7})
	if err != nil {
		t.Fatalf("marshal usage: %v", err)
	}
	want := `{"input_tokens":5,"output_tokens":7}`
	if string(encoded) != want {
		t.Fatalf("usage without server_tool_use: got=%s, want=%s", encoded, want)
	}

	encoded, err = json.Marshal(models.AnthropicUsage{
		InputTokens:   5,
		OutputTokens:  7,
		ServerToolUse: &models.AnthropicServerToolUse{WebSearchRequests: 2},
	})
	if err != nil {
		t.Fatalf("marshal usage with server_tool_use: %v", err)
	}
	want = `{"input_tokens":5,"output_tokens":7,"server_tool_use":{"web_search_requests":2}}`
	if string(encoded) != want {
		t.Fatalf("usage with server_tool_use: got=%s, want=%s", encoded, want)
	}
}

func TestHostedWebSearchTool(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name      string
		tools     []models.AnthropicTool
		wantFound bool
		wantIndex int
		wantType  string
	}{
		{
			name:      "no tools",
			tools:     nil,
			wantFound: false,
			wantIndex: -1,
		},
		{
			name: "client tools only",
			tools: []models.AnthropicTool{
				{Name: "Bash"},
				{Name: "web_search"},
			},
			wantFound: false,
			wantIndex: -1,
		},
		{
			name: "hosted tool after client tools",
			tools: []models.AnthropicTool{
				{Name: "Bash"},
				{Type: "web_search_20250305", Name: "web_search"},
			},
			wantFound: true,
			wantIndex: 1,
			wantType:  "web_search_20250305",
		},
		{
			name: "first hosted tool wins",
			tools: []models.AnthropicTool{
				{Type: "web_search_20250305", Name: "web_search"},
				{Type: "web_search_20260209", Name: "web_search"},
			},
			wantFound: true,
			wantIndex: 0,
			wantType:  "web_search_20250305",
		},
		{
			name: "unrelated hosted tool ignored",
			tools: []models.AnthropicTool{
				{Type: "web_fetch_20250910", Name: "web_fetch"},
			},
			wantFound: false,
			wantIndex: -1,
		},
		{
			name: "bare web_search type is not a versioned hosted tool",
			tools: []models.AnthropicTool{
				{Type: "web_search", Name: "web_search"},
			},
			wantFound: false,
			wantIndex: -1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			tool, index, found := hostedWebSearchTool(tc.tools)
			if found != tc.wantFound {
				t.Fatalf("found: got=%t, want=%t", found, tc.wantFound)
			}
			if index != tc.wantIndex {
				t.Fatalf("index: got=%d, want=%d", index, tc.wantIndex)
			}
			if found && tool.Type != tc.wantType {
				t.Fatalf("tool.type: got=%q, want=%q", tool.Type, tc.wantType)
			}
		})
	}
}

func TestClientDefinesWebSearchTool(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name  string
		tools []models.AnthropicTool
		want  bool
	}{
		{name: "no tools", tools: nil, want: false},
		{
			name:  "client web_search",
			tools: []models.AnthropicTool{{Name: "web_search"}},
			want:  true,
		},
		{
			name:  "hosted web_search only",
			tools: []models.AnthropicTool{{Type: "web_search_20250305", Name: "web_search"}},
			want:  false,
		},
		{
			name: "custom-typed client web_search",
			tools: []models.AnthropicTool{
				{Type: "custom", Name: "web_search"},
			},
			// "custom" is a client tool per isAnthropicClientTool, so it must
			// trip the ambiguity guard exactly like an untyped one.
			want: true,
		},
		{
			name:  "unrelated client tool",
			tools: []models.AnthropicTool{{Name: "WebSearch"}},
			want:  false,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := clientDefinesWebSearchTool(tc.tools); got != tc.want {
				t.Fatalf("clientDefinesWebSearchTool: got=%t, want=%t", got, tc.want)
			}
		})
	}
}

func TestIsAnthropicClientTool(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name string
		tool models.AnthropicTool
		want bool
	}{
		{name: "untyped", tool: models.AnthropicTool{Name: "Bash"}, want: true},
		{name: "custom", tool: models.AnthropicTool{Type: "custom", Name: "Bash"}, want: true},
		{name: "custom mixed case", tool: models.AnthropicTool{Type: "Custom", Name: "Bash"}, want: true},
		{name: "hosted search", tool: models.AnthropicTool{Type: "web_search_20250305"}, want: false},
		{name: "hosted fetch", tool: models.AnthropicTool{Type: "web_fetch_20250910"}, want: false},
		{name: "bash tool", tool: models.AnthropicTool{Type: "bash_20250124", Name: "bash"}, want: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := isAnthropicClientTool(tc.tool); got != tc.want {
				t.Fatalf("isAnthropicClientTool(%q): got=%t, want=%t", tc.tool.Type, got, tc.want)
			}
		})
	}
}

func TestTranslateAnthropicToOpenAIRejectsHostedServerTools(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name     string
		tools    []models.AnthropicTool
		wantFrag string
	}{
		{
			name: "hosted web search",
			tools: []models.AnthropicTool{
				{Name: "Bash", InputSchema: json.RawMessage(`{"type":"object"}`)},
				{Type: "web_search_20250305", Name: "web_search"},
			},
			wantFrag: `tools[1]: hosted server tool type "web_search_20250305" is not supported`,
		},
		{
			name: "hosted web fetch",
			tools: []models.AnthropicTool{
				{Type: "web_fetch_20250910", Name: "web_fetch"},
			},
			wantFrag: `tools[0]: hosted server tool type "web_fetch_20250910" is not supported`,
		},
		{
			name: "hosted bash tool",
			tools: []models.AnthropicTool{
				{Type: "bash_20250124", Name: "bash"},
			},
			wantFrag: `tools[0]: hosted server tool type "bash_20250124" is not supported`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := &models.AnthropicRequest{
				Model:    "claude-sonnet-4.5",
				Messages: []models.AnthropicMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
				Tools:    tc.tools,
			}

			oaiReq, err := TranslateAnthropicToOpenAI(req)
			if err == nil {
				t.Fatalf("TranslateAnthropicToOpenAI: got nil error and %d tools, want rejection", len(oaiReq.Tools))
			}
			if !strings.Contains(err.Error(), tc.wantFrag) {
				t.Fatalf("error: got=%q, want substring=%q", err.Error(), tc.wantFrag)
			}
		})
	}
}

func TestTranslateAnthropicToOpenAIAcceptsClientTools(t *testing.T) {
	t.Parallel()

	req := &models.AnthropicRequest{
		Model:    "claude-sonnet-4.5",
		Messages: []models.AnthropicMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		Tools: []models.AnthropicTool{
			{Name: "Bash", Description: "run a command", InputSchema: json.RawMessage(`{"type":"object"}`)},
			{Type: "custom", Name: "Grep", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
	}

	oaiReq, err := TranslateAnthropicToOpenAI(req)
	if err != nil {
		t.Fatalf("TranslateAnthropicToOpenAI: got err=%v, want nil", err)
	}
	if len(oaiReq.Tools) != 2 {
		t.Fatalf("tool count: got=%d, want=2", len(oaiReq.Tools))
	}
	for index, want := range []string{"Bash", "Grep"} {
		if got := oaiReq.Tools[index].Function.Name; got != want {
			t.Fatalf("tools[%d].function.name: got=%q, want=%q", index, got, want)
		}
		if oaiReq.Tools[index].Type != "function" {
			t.Fatalf("tools[%d].type: got=%q, want=%q", index, oaiReq.Tools[index].Type, "function")
		}
	}
	if string(oaiReq.Tools[0].Function.Parameters) != `{"type":"object"}` {
		t.Fatalf("tools[0].function.parameters: got=%s, want=%s",
			oaiReq.Tools[0].Function.Parameters, `{"type":"object"}`)
	}
}

func TestTranslateAnthropicToolsForTokenCount(t *testing.T) {
	t.Parallel()

	original := []models.AnthropicTool{
		{Name: "Bash", InputSchema: json.RawMessage(`{"type":"object"}`)},
		{Type: "web_search_20250305", Name: "web_search", MaxUses: intPtr(3)},
		{Type: "web_fetch_20250910", Name: "web_fetch"},
	}

	got, substituted := translateAnthropicToolsForTokenCount(original)
	if !substituted {
		t.Fatalf("substituted: got=false, want=true")
	}
	if len(got) != 3 {
		t.Fatalf("tool count: got=%d, want=3", len(got))
	}
	if got[1].Type != "" {
		t.Fatalf("tools[1].type: got=%q, want empty", got[1].Type)
	}
	if got[1].Name != "web_search" {
		t.Fatalf("tools[1].name: got=%q, want=%q", got[1].Name, "web_search")
	}
	if string(got[1].InputSchema) != webSearchStandInInputSchema {
		t.Fatalf("tools[1].input_schema: got=%s, want=%s", got[1].InputSchema, webSearchStandInInputSchema)
	}
	if got[1].MaxUses != nil {
		t.Fatalf("tools[1].max_uses: got=%v, want=nil", got[1].MaxUses)
	}
	// Non-search hosted tools survive untouched so the translator still rejects them.
	if got[2].Type != "web_fetch_20250910" {
		t.Fatalf("tools[2].type: got=%q, want=%q", got[2].Type, "web_fetch_20250910")
	}
	// The caller's slice must not be mutated.
	if original[1].Type != "web_search_20250305" {
		t.Fatalf("original tools[1].type: got=%q, want unchanged", original[1].Type)
	}

	unchanged, substituted := translateAnthropicToolsForTokenCount([]models.AnthropicTool{{Name: "Bash"}})
	if substituted {
		t.Fatalf("substituted for client-only tools: got=true, want=false")
	}
	if len(unchanged) != 1 || unchanged[0].Name != "Bash" {
		t.Fatalf("client-only tools: got=%+v, want unchanged", unchanged)
	}
}

func TestHandleAnthropicMessagesRejectsHostedWebSearchTool(t *testing.T) {
	t.Parallel()

	var upstreamCalls int
	handler := newTestProxyHandler(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusOK)
	})

	body := `{
		"model": "claude-sonnet-4",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "search the web"}],
		"tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
	}`
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	handler.HandleAnthropicMessages(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d, want=%d (body=%s)", recorder.Code, http.StatusBadRequest, recorder.Body.String())
	}
	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode error body %s: %v", recorder.Body.Bytes(), err)
	}
	if envelope.Type != "error" || envelope.Error.Type != "invalid_request_error" {
		t.Fatalf("envelope: got type=%q error.type=%q, want error/invalid_request_error",
			envelope.Type, envelope.Error.Type)
	}
	if !strings.Contains(envelope.Error.Message, "tools[0]") {
		t.Fatalf("error.message: got=%q, want substring=%q", envelope.Error.Message, "tools[0]")
	}
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls: got=%d, want=0", upstreamCalls)
	}
}

type webSearchTestCatalog struct {
	provider *providerRuntime
	models   []providerModel
}

func webSearchTestProvider(id string, kind providerType, baseURL string) *providerRuntime {
	return &providerRuntime{
		id:            id,
		kind:          kind,
		baseURL:       baseURL,
		paths:         providerEndpointPolicyFor(kind).defaultEndpointPaths(),
		includeModels: map[string]struct{}{},
		excludeModels: map[string]struct{}{},
		staticModels:  map[string]providerModel{},
	}
}

func webSearchTestHandler(t *testing.T, cfg WebSearchConfig, entries ...webSearchTestCatalog) *ProxyHandler {
	t.Helper()
	setup := &providerSetup{
		providers:          map[string]*providerRuntime{},
		models:             map[string]providerModel{},
		hasConfiguredState: true,
	}
	for _, entry := range entries {
		setup.providers[entry.provider.id] = entry.provider
		setup.providerOrder = append(setup.providerOrder, entry.provider.id)
		if setup.defaultProviderID == "" {
			setup.defaultProviderID = entry.provider.id
		}
	}
	for _, entry := range entries {
		if err := setup.addProviderModels(entry.provider.id, entry.models); err != nil {
			t.Fatalf("addProviderModels(%q) error = %v", entry.provider.id, err)
		}
	}
	h := &ProxyHandler{
		auth:           auth.NewTestAuthenticator("test-token"),
		client:         http.DefaultClient,
		log:            logger.NewWithWriter(logger.LevelError, io.Discard),
		providersState: setup,
		webSearch:      cfg,
	}
	h.initializeLifecycle()
	return h
}

func webSearchTestEnabledConfig() WebSearchConfig {
	return WebSearchConfig{
		Enabled:           true,
		MaxSearches:       5,
		TimeoutMS:         30000,
		SearchContextSize: "low",
		MaxResults:        10,
	}
}

func webSearchTestHostedTool() models.AnthropicTool {
	return models.AnthropicTool{Type: "web_search_20250305", Name: "web_search"}
}

func webSearchTestRequest(model string, tools ...models.AnthropicTool) *models.AnthropicRequest {
	return &models.AnthropicRequest{
		Model:    model,
		Messages: []models.AnthropicMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		Tools:    tools,
	}
}

func TestWebSearchMediationForEngagesOnCopilotWithSameProviderDelegate(t *testing.T) {
	copilot := webSearchTestProvider("copilot", providerTypeCopilot, "https://copilot.invalid")
	h := webSearchTestHandler(t, webSearchTestEnabledConfig(), webSearchTestCatalog{
		provider: copilot,
		models: []providerModel{
			{publicID: "claude-sonnet-4.5", upstreamModel: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
			{publicID: "gpt-5.1", upstreamModel: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
		},
	})

	maxUses := 2
	tool := webSearchTestHostedTool()
	tool.MaxUses = &maxUses

	mediation, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", tool))
	if !ok {
		t.Fatal("webSearchMediationFor() did not engage for a Copilot-owned model")
	}
	if mediation.provider == nil || mediation.provider.id != "copilot" {
		t.Fatalf("mediation.provider = %#v, want copilot", mediation.provider)
	}
	if mediation.delegate.publicID != "gpt-5.1" {
		t.Fatalf("mediation.delegate = %q, want gpt-5.1", mediation.delegate.publicID)
	}
	if mediation.maxSearches != 2 {
		t.Fatalf("mediation.maxSearches = %d, want 2 (min of max_uses and max_searches)", mediation.maxSearches)
	}
}

func TestWebSearchMediationForUsesConfigCeilingWhenMaxUsesOmitted(t *testing.T) {
	copilot := webSearchTestProvider("copilot", providerTypeCopilot, "https://copilot.invalid")
	h := webSearchTestHandler(t, webSearchTestEnabledConfig(), webSearchTestCatalog{
		provider: copilot,
		models: []providerModel{
			{publicID: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
			{publicID: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
		},
	})

	mediation, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool()))
	if !ok {
		t.Fatal("webSearchMediationFor() did not engage without max_uses")
	}
	if mediation.maxSearches != 5 {
		t.Fatalf("mediation.maxSearches = %d, want 5 (config max_searches alone)", mediation.maxSearches)
	}
}

func TestWebSearchMediationForSkipsAnthropicCompatibleProvider(t *testing.T) {
	native := webSearchTestProvider("anthropic", providerTypeAnthropicCompatible, "https://anthropic.invalid")
	h := webSearchTestHandler(t, webSearchTestEnabledConfig(), webSearchTestCatalog{
		provider: native,
		models: []providerModel{
			{publicID: "claude-sonnet-4.5", providerID: "anthropic", supportedEndpoints: []string{providerEndpointMessages}},
			{publicID: "claude-responses", providerID: "anthropic", supportedEndpoints: []string{providerEndpointResponses}},
		},
	})

	if _, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool())); ok {
		t.Fatal("webSearchMediationFor() mediated an anthropic-compatible provider that already serves hosted web_search natively")
	}
}

func TestWebSearchMediationForNeverSelectsDelegateOnAnotherProvider(t *testing.T) {
	copilot := webSearchTestProvider("copilot", providerTypeCopilot, "https://copilot.invalid")
	azure := webSearchTestProvider("azure", providerTypeAzureOpenAI, "https://azure.invalid")
	h := webSearchTestHandler(t, webSearchTestEnabledConfig(),
		webSearchTestCatalog{
			provider: copilot,
			models: []providerModel{
				{publicID: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
			},
		},
		webSearchTestCatalog{
			provider: azure,
			models: []providerModel{
				{publicID: "gpt-5.4-pro", providerID: "azure", supportedEndpoints: []string{providerEndpointResponses}},
			},
		},
	)

	if mediation, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool())); ok {
		t.Fatalf("webSearchMediationFor() engaged with cross-provider delegate %q on provider %q", mediation.delegate.publicID, mediation.delegate.providerID)
	}
}

func TestWebSearchMediationForRejectsConfiguredDelegateOnAnotherProvider(t *testing.T) {
	cfg := webSearchTestEnabledConfig()
	cfg.DelegateModel = "gpt-5.4-pro"
	copilot := webSearchTestProvider("copilot", providerTypeCopilot, "https://copilot.invalid")
	azure := webSearchTestProvider("azure", providerTypeAzureOpenAI, "https://azure.invalid")
	h := webSearchTestHandler(t, cfg,
		webSearchTestCatalog{
			provider: copilot,
			models: []providerModel{
				{publicID: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
				{publicID: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
			},
		},
		webSearchTestCatalog{
			provider: azure,
			models: []providerModel{
				{publicID: "gpt-5.4-pro", providerID: "azure", supportedEndpoints: []string{providerEndpointResponses}},
			},
		},
	)

	if _, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool())); ok {
		t.Fatal("webSearchMediationFor() honored a configured delegate owned by a different provider")
	}
}

func TestWebSearchMediationForRejectsConfiguredDelegateWithoutResponses(t *testing.T) {
	cfg := webSearchTestEnabledConfig()
	cfg.DelegateModel = "claude-messages-only"
	copilot := webSearchTestProvider("copilot", providerTypeCopilot, "https://copilot.invalid")
	h := webSearchTestHandler(t, cfg, webSearchTestCatalog{
		provider: copilot,
		models: []providerModel{
			{publicID: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
			{publicID: "claude-messages-only", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
			{publicID: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
		},
	})

	// A pinned delegate that cannot serve /responses must decline outright rather
	// than silently fall through to an auto-discovered one: honoring the pin
	// would activate mediation and then fail every search, and falling through
	// would ignore an explicit operator choice.
	if _, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool())); ok {
		t.Fatal("webSearchMediationFor() honored a configured delegate that does not advertise /responses")
	}
}

func TestWebSearchMediationForDeclinesWhenDisabledOrAmbiguous(t *testing.T) {
	catalog := func() webSearchTestCatalog {
		return webSearchTestCatalog{
			provider: webSearchTestProvider("copilot", providerTypeCopilot, "https://copilot.invalid"),
			models: []providerModel{
				{publicID: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
				{publicID: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
			},
		}
	}

	disabledCfg := webSearchTestEnabledConfig()
	disabledCfg.Enabled = false
	if _, ok := webSearchTestHandler(t, disabledCfg, catalog()).webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool())); ok {
		t.Fatal("webSearchMediationFor() engaged while disabled")
	}

	h := webSearchTestHandler(t, webSearchTestEnabledConfig(), catalog())
	if _, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5")); ok {
		t.Fatal("webSearchMediationFor() engaged without a hosted tool")
	}
	clientTool := models.AnthropicTool{Name: "web_search", InputSchema: json.RawMessage(`{"type":"object"}`)}
	if _, ok := h.webSearchMediationFor(webSearchTestRequest("claude-sonnet-4.5", webSearchTestHostedTool(), clientTool)); ok {
		t.Fatal("webSearchMediationFor() engaged while the client defines its own web_search tool")
	}
}

func TestHandleAnthropicCountTokensSubstitutesHostedWebSearchTool(t *testing.T) {
	t.Parallel()

	handler := newTestProxyHandler(t, func(w http.ResponseWriter, r *http.Request) {
		var oaiReq models.OpenAIRequest
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &oaiReq); err != nil {
			t.Errorf("parse upstream request: %v", err)
			return
		}
		if len(oaiReq.Tools) != 2 {
			t.Errorf("upstream tool count: got=%d, want=2", len(oaiReq.Tools))
			return
		}
		if oaiReq.Tools[1].Function.Name != "web_search" {
			t.Errorf("tools[1].function.name: got=%q, want=%q", oaiReq.Tools[1].Function.Name, "web_search")
		}
		if string(oaiReq.Tools[1].Function.Parameters) != webSearchStandInInputSchema {
			t.Errorf("tools[1].function.parameters: got=%s, want=%s",
				oaiReq.Tools[1].Function.Parameters, webSearchStandInInputSchema)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-count","object":"chat.completion","created":1,` +
			`"model":"claude-sonnet-4","choices":[{"index":0,"message":{"role":"assistant","content":"x"},` +
			`"finish_reason":"length"}],"usage":{"prompt_tokens":321,"completion_tokens":1,"total_tokens":322}}`))
	})

	body := `{
		"model": "claude-sonnet-4",
		"messages": [{"role": "user", "content": "search the web"}],
		"tools": [
			{"name": "Bash", "input_schema": {"type": "object"}},
			{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
		]
	}`
	request := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	handler.HandleAnthropicMessagesCountTokens(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status: got=%d, want=%d (body=%s)", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var countResp models.AnthropicCountTokensResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &countResp); err != nil {
		t.Fatalf("decode count_tokens response %s: %v", recorder.Body.Bytes(), err)
	}
	if countResp.InputTokens != 321 {
		t.Fatalf("input_tokens: got=%d, want=321", countResp.InputTokens)
	}
}

func TestHandleAnthropicCountTokensRejectsHostedWebFetchTool(t *testing.T) {
	t.Parallel()

	var upstreamCalls int
	handler := newTestProxyHandler(t, func(w http.ResponseWriter, r *http.Request) {
		upstreamCalls++
		w.WriteHeader(http.StatusOK)
	})

	body := `{
		"model": "claude-sonnet-4",
		"messages": [{"role": "user", "content": "fetch this"}],
		"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]
	}`
	request := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	handler.HandleAnthropicMessagesCountTokens(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d, want=%d (body=%s)", recorder.Code, http.StatusBadRequest, recorder.Body.String())
	}
	if !strings.Contains(recorder.Body.String(), "tools[0]") {
		t.Fatalf("body: got=%s, want substring=%q", recorder.Body.String(), "tools[0]")
	}
	if upstreamCalls != 0 {
		t.Fatalf("upstream calls: got=%d, want=0", upstreamCalls)
	}
}

func TestRewriteAnthropicWebSearchRequestSwapsToolInPlace(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-4.5","stream":true,"metadata":{"user_id":"u1"},"vekil_unknown_field":{"keep":"me"},` +
		`"messages":[{"role":"user","content":"weather?"}],` +
		`"tools":[{"name":"Bash","description":"run","input_schema":{"type":"object"}},` +
		`{"type":"web_search_20250305","name":"web_search","max_uses":3,"allowed_domains":["example.com"],` +
		`"blocked_domains":["spam.test"],"user_location":{"type":"approximate","city":"Seattle"}},` +
		`{"name":"Read","input_schema":{"type":"object"}}]}`)

	maxUses := 3
	mediation := &webSearchMediation{
		cfg:  webSearchTestEnabledConfig(),
		tool: models.AnthropicTool{Type: "web_search_20250305", Name: "web_search", MaxUses: &maxUses},
	}

	rewritten, err := rewriteAnthropicWebSearchRequest(body, mediation)
	if err != nil {
		t.Fatalf("rewriteAnthropicWebSearchRequest() error = %v", err)
	}

	var decoded struct {
		Stream  bool              `json:"stream"`
		Unknown map[string]string `json:"vekil_unknown_field"`
		Tools   []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(rewritten, &decoded); err != nil {
		t.Fatalf("rewritten body is not valid JSON: %v", err)
	}
	if decoded.Stream {
		t.Fatal("rewritten body did not force stream=false")
	}
	if decoded.Unknown["keep"] != "me" {
		t.Fatalf("unknown top-level field was dropped: %s", rewritten)
	}
	if len(decoded.Tools) != 3 {
		t.Fatalf("len(tools) = %d, want 3", len(decoded.Tools))
	}

	var swapped map[string]any
	if err := json.Unmarshal(decoded.Tools[1], &swapped); err != nil {
		t.Fatalf("swapped tool is not valid JSON: %v", err)
	}
	if swapped["name"] != "web_search" {
		t.Fatalf("swapped tool name = %#v, want web_search", swapped["name"])
	}
	for _, forbidden := range []string{"type", "max_uses", "allowed_domains", "blocked_domains", "user_location"} {
		if _, present := swapped[forbidden]; present {
			t.Fatalf("hosted field %q leaked upstream: %s", forbidden, decoded.Tools[1])
		}
	}
	schema, ok := swapped["input_schema"].(map[string]any)
	if !ok {
		t.Fatalf("swapped tool has no input_schema object: %s", decoded.Tools[1])
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok || properties["query"] == nil {
		t.Fatalf("swapped tool schema lacks a query property: %s", decoded.Tools[1])
	}

	var first map[string]any
	if err := json.Unmarshal(decoded.Tools[0], &first); err != nil {
		t.Fatalf("neighbor tool is not valid JSON: %v", err)
	}
	if first["name"] != "Bash" {
		t.Fatalf("tool order changed: tools[0] = %s", decoded.Tools[0])
	}
}

func TestRewriteAnthropicWebSearchRequestErrorsWithoutHostedTool(t *testing.T) {
	body := []byte(`{"model":"claude-sonnet-4.5","tools":[{"name":"Bash","input_schema":{"type":"object"}}]}`)
	if _, err := rewriteAnthropicWebSearchRequest(body, &webSearchMediation{}); err == nil {
		t.Fatal("rewriteAnthropicWebSearchRequest() succeeded on a body with no hosted tool")
	}
	if _, err := rewriteAnthropicWebSearchRequest([]byte(`{`), &webSearchMediation{}); err == nil {
		t.Fatal("rewriteAnthropicWebSearchRequest() succeeded on malformed JSON")
	}
	if _, err := rewriteAnthropicWebSearchRequest([]byte(`{"model":"m"}`), nil); err == nil {
		t.Fatal("rewriteAnthropicWebSearchRequest() succeeded with a nil mediation")
	}
}

func TestDelegateWebSearchPinsRequestAndPostFiltersResults(t *testing.T) {
	var captured []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != providerEndpointResponses {
			t.Errorf("delegate path = %q, want %q", r.URL.Path, providerEndpointResponses)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read delegate body: %v", err)
		}
		captured = body
		payload := `[{"url":"https://example.com/a","title":"A","snippet":"first","page_age":"2 days ago"},` +
			`{"url":"https://spam.test/b","title":"B","snippet":"blocked","page_age":""},` +
			`{"url":"https://example.com/c","title":"C","snippet":"third","page_age":""}]`
		fenced, err := json.Marshal("```json\n" + payload + "\n```")
		if err != nil {
			t.Errorf("marshal delegate text: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp-1","status":"completed","usage":{"input_tokens":140,"output_tokens":60,"total_tokens":200},`+
			`"output":[{"type":"message","content":[{"type":"output_text","text":`+string(fenced)+`}]}]}`)
	}))
	defer upstream.Close()

	copilot := webSearchTestProvider("copilot", providerTypeCopilot, upstream.URL)
	h := webSearchTestHandler(t, webSearchTestEnabledConfig(), webSearchTestCatalog{
		provider: copilot,
		models: []providerModel{
			{publicID: "claude-sonnet-4.5", providerID: "copilot", supportedEndpoints: []string{providerEndpointMessages}},
			{publicID: "gpt-5.1", upstreamModel: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
		},
	})

	cfg := webSearchTestEnabledConfig()
	cfg.MaxResults = 1
	_, summary := WithRequestSummary(context.Background())
	mediation := &webSearchMediation{
		cfg:         cfg,
		provider:    copilot,
		delegate:    providerModel{publicID: "gpt-5.1", upstreamModel: "gpt-5.1", providerID: "copilot"},
		maxSearches: 1,
		summary:     summary,
		tool: models.AnthropicTool{
			Type:           "web_search_20250305",
			Name:           "web_search",
			AllowedDomains: []string{"example.com"},
			BlockedDomains: []string{"spam.test"},
			UserLocation:   json.RawMessage(`{"type":"approximate","city":"Seattle"}`),
		},
	}

	results, err := h.delegateWebSearch(context.Background(), mediation, "seattle weather")
	if err != nil {
		t.Fatalf("delegateWebSearch() error = %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("len(results) = %d, want 1 after blocked-domain filtering and max_results truncation: %#v", len(results), results)
	}
	if results[0].URL != "https://example.com/a" || results[0].PageAge != "2 days ago" {
		t.Fatalf("results[0] = %#v", results[0])
	}

	// Delegated spend is accumulated on the mediation, not booked to the turn.
	if mediation.delegatedUsage.InputTokens != 140 || mediation.delegatedUsage.OutputTokens != 60 {
		t.Fatalf("delegatedUsage after one call = %#v, want 140/60", mediation.delegatedUsage)
	}
	if _, err := h.delegateWebSearch(context.Background(), mediation, "seattle weather again"); err != nil {
		t.Fatalf("second delegateWebSearch() error = %v", err)
	}
	if mediation.delegatedUsage.InputTokens != 280 || mediation.delegatedUsage.OutputTokens != 120 {
		t.Fatalf("delegatedUsage is not additive across calls: %#v", mediation.delegatedUsage)
	}
	if got := summary.UpstreamSendCount(); got != 2 {
		t.Fatalf("summary.UpstreamSendCount() = %d, want 2 (one per delegated dispatch)", got)
	}

	var request struct {
		Model     string `json:"model"`
		Stream    bool   `json:"stream"`
		Reasoning struct {
			Effort string `json:"effort"`
		} `json:"reasoning"`
		Tools []struct {
			Type              string `json:"type"`
			SearchContextSize string `json:"search_context_size"`
			Filters           struct {
				AllowedDomains []string `json:"allowed_domains"`
				BlockedDomains []string `json:"blocked_domains"`
			} `json:"filters"`
			UserLocation json.RawMessage `json:"user_location"`
		} `json:"tools"`
		Input []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(captured, &request); err != nil {
		t.Fatalf("delegate request is not valid JSON: %v", err)
	}
	if request.Model != "gpt-5.1" || request.Stream {
		t.Fatalf("delegate request model/stream = %q/%v", request.Model, request.Stream)
	}
	if request.Reasoning.Effort != "low" {
		t.Fatalf("reasoning.effort = %q, want low", request.Reasoning.Effort)
	}
	if len(request.Tools) != 1 || request.Tools[0].Type != "web_search" {
		t.Fatalf("delegate tools = %#v", request.Tools)
	}
	if request.Tools[0].SearchContextSize != "low" {
		t.Fatalf("search_context_size = %q, want low", request.Tools[0].SearchContextSize)
	}
	if len(request.Tools[0].Filters.AllowedDomains) != 1 || request.Tools[0].Filters.AllowedDomains[0] != "example.com" {
		t.Fatalf("filters.allowed_domains = %#v", request.Tools[0].Filters.AllowedDomains)
	}
	if len(request.Tools[0].Filters.BlockedDomains) != 1 || request.Tools[0].Filters.BlockedDomains[0] != "spam.test" {
		t.Fatalf("filters.blocked_domains = %#v", request.Tools[0].Filters.BlockedDomains)
	}
	if !strings.Contains(string(request.Tools[0].UserLocation), "Seattle") {
		t.Fatalf("user_location = %s", request.Tools[0].UserLocation)
	}
	if len(request.Input) != 2 || request.Input[0].Role != "developer" {
		t.Fatalf("delegate input = %#v", request.Input)
	}
}

func TestDelegateWebSearchFailsOnUpstreamErrorAndInvalidJSON(t *testing.T) {
	mode := "error"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if mode == "error" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, `{"error":{"message":"nope"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp-2","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"I could not find anything."}]}]}`)
	}))
	defer upstream.Close()

	copilot := webSearchTestProvider("copilot", providerTypeCopilot, upstream.URL)
	h := webSearchTestHandler(t, webSearchTestEnabledConfig(), webSearchTestCatalog{
		provider: copilot,
		models: []providerModel{
			{publicID: "gpt-5.1", upstreamModel: "gpt-5.1", providerID: "copilot", supportedEndpoints: []string{providerEndpointResponses}},
		},
	})
	mediation := &webSearchMediation{
		cfg:      webSearchTestEnabledConfig(),
		provider: copilot,
		delegate: providerModel{publicID: "gpt-5.1", upstreamModel: "gpt-5.1", providerID: "copilot"},
		tool:     webSearchTestHostedTool(),
	}

	// A nil summary must not panic: delegation is also reachable from paths
	// that have no request summary attached.
	if _, err := h.delegateWebSearch(context.Background(), mediation, "q"); err == nil {
		t.Fatal("delegateWebSearch() succeeded on a non-200 upstream response")
	}
	mode = "prose"
	if _, err := h.delegateWebSearch(context.Background(), mediation, "q"); err == nil {
		t.Fatal("delegateWebSearch() succeeded on non-JSON delegate output")
	}
}

func TestFilterWebSearchResultsIsAuthoritative(t *testing.T) {
	results := []webSearchResult{
		{URL: "https://docs.example.com/x", Title: "sub"},
		{URL: "https://spam.test/y", Title: "blocked"},
		{URL: "https://evil.example.org/z", Title: "not allowed"},
		{URL: "not a url", Title: "unparseable"},
		{URL: "https://example.com/w", Title: "allowed"},
	}
	tool := models.AnthropicTool{
		AllowedDomains: []string{"example.com"},
		BlockedDomains: []string{"spam.test"},
	}

	filtered := filterWebSearchResults(results, tool, 10)
	if len(filtered) != 2 {
		t.Fatalf("len(filtered) = %d, want 2: %#v", len(filtered), filtered)
	}
	if filtered[0].URL != "https://docs.example.com/x" || filtered[1].URL != "https://example.com/w" {
		t.Fatalf("filtered = %#v", filtered)
	}
	if truncated := filterWebSearchResults(results, tool, 1); len(truncated) != 1 {
		t.Fatalf("len(truncated) = %d, want 1", len(truncated))
	}
	if unbounded := filterWebSearchResults(results, models.AnthropicTool{}, 3); len(unbounded) != 3 {
		t.Fatalf("len(unbounded) = %d, want 3 with no filters and max_results=3", len(unbounded))
	}
}

// TestFilterWebSearchResultsNormalizesTrailingDotHosts guards against a
// blocklist bypass: "spam.test." is DNS-equivalent to "spam.test" and fully
// resolvable as the blocked domain, so a naive string comparison that treats
// the trailing root-label dot as significant lets a blocked result through
// (and, symmetrically, drops an otherwise-allowed one).
func TestFilterWebSearchResultsNormalizesTrailingDotHosts(t *testing.T) {
	results := []webSearchResult{
		{URL: "https://spam.test./evades-blocklist", Title: "trailing-dot blocked host"},
		{URL: "https://example.com./trailing-dot-allowed", Title: "trailing-dot allowed host"},
	}
	tool := models.AnthropicTool{
		AllowedDomains: []string{"example.com"},
		BlockedDomains: []string{"spam.test"},
	}

	filtered := filterWebSearchResults(results, tool, 10)
	if len(filtered) != 1 {
		t.Fatalf("len(filtered) = %d, want 1: %#v", len(filtered), filtered)
	}
	if filtered[0].URL != "https://example.com./trailing-dot-allowed" {
		t.Fatalf("filtered = %#v, want only the trailing-dot allowed host to survive", filtered)
	}
}

func TestNewWebSearchCallIDShape(t *testing.T) {
	seen := map[string]struct{}{}
	for i := 0; i < 64; i++ {
		id := newWebSearchCallID()
		if !strings.HasPrefix(id, webSearchCallIDPrefix) {
			t.Fatalf("call id %q lacks prefix %q", id, webSearchCallIDPrefix)
		}
		suffix := strings.TrimPrefix(id, webSearchCallIDPrefix)
		if len(suffix) != 22 {
			t.Fatalf("call id suffix %q has length %d, want 22", suffix, len(suffix))
		}
		if _, err := base64.RawURLEncoding.DecodeString(suffix); err != nil {
			t.Fatalf("call id suffix %q is not base64url: %v", suffix, err)
		}
		if _, duplicate := seen[id]; duplicate {
			t.Fatalf("call id %q was generated twice", id)
		}
		seen[id] = struct{}{}
	}
}

func TestWebSearchContentRoundTripsAndFailsClosed(t *testing.T) {
	original := webSearchResult{URL: "https://example.com/a", Title: "A", Snippet: "some text", PageAge: "2 days ago"}
	encoded := encodeWebSearchContent(original)
	if !strings.HasPrefix(encoded, webSearchContentVersion) {
		t.Fatalf("encoded token %q lacks version prefix %q", encoded, webSearchContentVersion)
	}
	decoded, ok := decodeWebSearchContent(encoded)
	if !ok {
		t.Fatalf("decodeWebSearchContent(%q) failed on a token we minted", encoded)
	}
	if decoded != original {
		t.Fatalf("decoded = %#v, want %#v", decoded, original)
	}

	payload := strings.TrimPrefix(encoded, webSearchContentVersion)
	for name, token := range map[string]string{
		"empty":             "",
		"missing version":   payload,
		"wrong version":     "vkws0:" + payload,
		"bad base64":        webSearchContentVersion + "!!!not-base64!!!",
		"bad json":          webSearchContentVersion + base64.RawURLEncoding.EncodeToString([]byte("not json")),
		"oversize":          webSearchContentVersion + strings.Repeat("A", webSearchMaxContentBytes),
		"plain client text": "just some text the client made up",
	} {
		if _, ok := decodeWebSearchContent(token); ok {
			t.Fatalf("decodeWebSearchContent accepted %s token %q", name, token)
		}
	}
}

func TestEncodeWebSearchContentStaysUnderCap(t *testing.T) {
	huge := webSearchResult{URL: "https://example.com/a", Title: "A", Snippet: strings.Repeat("x", 4*webSearchMaxContentBytes)}
	encoded := encodeWebSearchContent(huge)
	if len(encoded) > webSearchMaxContentBytes {
		t.Fatalf("len(encoded) = %d, want <= %d", len(encoded), webSearchMaxContentBytes)
	}
	decoded, ok := decodeWebSearchContent(encoded)
	if !ok {
		t.Fatalf("size-capped token did not decode: %q", encoded)
	}
	if decoded.URL != huge.URL || decoded.Title != huge.Title {
		t.Fatalf("size-capped token lost identity fields: %#v", decoded)
	}
}

func TestSynthesizeWebSearchBlocksShapes(t *testing.T) {
	callID := newWebSearchCallID()
	results := []webSearchResult{
		{URL: "https://example.com/a", Title: "A", Snippet: "first", PageAge: "2 days ago"},
		{URL: "https://example.com/b", Title: "B", Snippet: "second"},
	}

	use, result := synthesizeWebSearchBlocks(callID, "seattle weather", results)
	if use.Type != "server_tool_use" || use.ID != callID || use.Name != anthropicWebSearchToolName {
		t.Fatalf("server_tool_use block = %#v", use)
	}
	var input struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(use.Input, &input); err != nil || input.Query != "seattle weather" {
		t.Fatalf("server_tool_use input = %s (err %v)", use.Input, err)
	}

	if result.Type != "web_search_tool_result" || result.ToolUseID != callID {
		t.Fatalf("web_search_tool_result block = %#v", result)
	}
	var items []struct {
		Type             string `json:"type"`
		URL              string `json:"url"`
		Title            string `json:"title"`
		EncryptedContent string `json:"encrypted_content"`
		PageAge          string `json:"page_age"`
	}
	if err := json.Unmarshal(result.Content, &items); err != nil {
		t.Fatalf("success content must be a JSON list: %v (%s)", err, result.Content)
	}
	if len(items) != 2 {
		t.Fatalf("len(items) = %d, want 2", len(items))
	}
	if items[0].Type != "web_search_result" || items[0].URL != "https://example.com/a" || items[0].PageAge != "2 days ago" {
		t.Fatalf("items[0] = %#v", items[0])
	}
	roundTripped, ok := decodeWebSearchContent(items[1].EncryptedContent)
	if !ok || roundTripped.Snippet != "second" {
		t.Fatalf("items[1].encrypted_content did not round trip: %#v (ok=%v)", roundTripped, ok)
	}
}

func TestWebSearchErrorResultBlockIsSingleObject(t *testing.T) {
	callID := newWebSearchCallID()
	block := webSearchErrorResultBlock(callID, "max_uses_exceeded")
	if block.Type != "web_search_tool_result" || block.ToolUseID != callID {
		t.Fatalf("error block = %#v", block)
	}
	var list []json.RawMessage
	if err := json.Unmarshal(block.Content, &list); err == nil {
		t.Fatalf("error content decoded as a list, must be a single object: %s", block.Content)
	}
	var object struct {
		Type      string `json:"type"`
		ErrorCode string `json:"error_code"`
	}
	if err := json.Unmarshal(block.Content, &object); err != nil {
		t.Fatalf("error content is not a JSON object: %v (%s)", err, block.Content)
	}
	if object.Type != "web_search_tool_result_error" || object.ErrorCode != "max_uses_exceeded" {
		t.Fatalf("error content = %s", block.Content)
	}
}
