package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

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
