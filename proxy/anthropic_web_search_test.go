package proxy

import (
	"encoding/json"
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

// TestTranslateAnthropicToOpenAIUnaffectedByHostedToolDecode pins the existing
// translated-path behavior while Type merely becomes decodable. Task 3 replaces
// this expectation with an explicit rejection.
func TestTranslateAnthropicToOpenAIUnaffectedByHostedToolDecode(t *testing.T) {
	t.Parallel()

	req := &models.AnthropicRequest{
		Model:    "claude-sonnet-4.5",
		Messages: []models.AnthropicMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		Tools: []models.AnthropicTool{
			{Type: "web_search_20250305", Name: "web_search"},
		},
	}

	oaiReq, err := TranslateAnthropicToOpenAI(req)
	if err != nil {
		t.Fatalf("TranslateAnthropicToOpenAI: got err=%v, want nil", err)
	}
	if len(oaiReq.Tools) != 1 {
		t.Fatalf("tool count: got=%d, want=1", len(oaiReq.Tools))
	}
	if oaiReq.Tools[0].Function.Name != "web_search" {
		t.Fatalf("tools[0].function.name: got=%q, want=%q", oaiReq.Tools[0].Function.Name, "web_search")
	}
}
