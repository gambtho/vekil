package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/sozercan/vekil/auth"
	"github.com/sozercan/vekil/logger"
	"github.com/sozercan/vekil/models"
)

// webSearchLoopUpstream is a fake Copilot upstream serving the three endpoints a
// mediated turn needs: /models for catalog discovery, /v1/messages for the loop
// turns, and /responses for delegated searches.
type webSearchLoopUpstream struct {
	mu             sync.Mutex
	messageBodies  [][]byte
	responsesCalls int
	turns          []string
}

func (u *webSearchLoopUpstream) handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case providerEndpointModels:
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{
				map[string]any{"id": "claude-mediated", "supported_endpoints": []string{providerEndpointMessages}},
				map[string]any{"id": "gpt-delegate", "supported_endpoints": []string{providerEndpointResponses}},
			}})
		case providerEndpointResponses:
			u.mu.Lock()
			u.responsesCalls++
			u.mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{
				"output": []any{map[string]any{"type": "message", "content": []any{
					map[string]any{"type": "output_text", "text": `[{"url":"https://example.com/a","title":"A","snippet":"alpha","page_age":"1 day"}]`},
				}}},
				"usage": map[string]any{"input_tokens": 7, "output_tokens": 3},
			})
		case providerEndpointMessages:
			body, _ := io.ReadAll(r.Body)
			u.mu.Lock()
			u.messageBodies = append(u.messageBodies, body)
			index := len(u.messageBodies) - 1
			turn := ""
			if index < len(u.turns) {
				turn = u.turns[index]
			}
			u.mu.Unlock()
			if turn == "" {
				http.Error(w, "unexpected extra messages dispatch", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			// Per-turn value so a test can prove the emitted turn carries the LAST
			// turn's passthrough headers, the way real rate-limit counters move.
			w.Header().Set("Anthropic-Ratelimit-Requests-Remaining", strconv.Itoa(index))
			w.Header().Set("Content-Length", strconv.Itoa(len(turn)))
			_, _ = io.WriteString(w, turn)
		default:
			http.NotFound(w, r)
		}
	})
}

// newWebSearchLoopTestHandler builds a handler whose conversation model is owned
// by an explicit /v1/messages route (the only route shape a Messages request can
// dispatch through an explicit routeOperation: providerKindSupportsExplicitEndpoint
// rejects /v1/messages for Copilot targets) and whose delegate model lives on a
// Copilot provider serving /responses.
func newWebSearchLoopTestHandler(t *testing.T, baseURL string) *ProxyHandler {
	t.Helper()
	h, err := NewProxyHandler(
		auth.NewTestAuthenticator("fixture-token"),
		logger.NewWithWriter(logger.LevelError, io.Discard),
		WithCopilotBaseURL(baseURL),
		WithProvidersConfig(ProvidersConfig{
			SchemaVersion: 2,
			Providers: []ProviderConfig{
				{
					ID: "copilot", Type: string(providerTypeCopilot), Default: true,
					TrustDomain: "github-copilot", IncludeModels: []string{"gpt-delegate"},
				},
				{
					ID: "conversation", Type: string(providerTypeAnthropicCompatible),
					BaseURL: baseURL, AuthType: "none",
				},
			},
			ModelRoutes: []ModelRouteConfig{{
				ID: "claude-route", PublicID: "claude-mediated",
				Endpoints: []string{providerEndpointMessages},
				Targets:   []ModelRouteTargetConfig{{ID: "primary", Provider: "conversation", UpstreamModel: "claude-mediated"}},
			}},
		}),
		WithDeferredDynamicProviderModelValidation(true),
	)
	if err != nil {
		t.Fatalf("NewProxyHandler() error = %v", err)
	}
	t.Cleanup(h.BeginShutdown)
	if err := h.ValidateDynamicProviderModels(t.Context()); err != nil {
		t.Fatalf("ValidateDynamicProviderModels() error = %v", err)
	}
	return h
}

func newWebSearchLoopMediation(t *testing.T, h *ProxyHandler, maxSearches int) *webSearchMediation {
	t.Helper()
	provider, delegate, known := h.resolveProviderModelForRequest("gpt-delegate", providerEndpointResponses)
	if !known || provider == nil {
		t.Fatal("delegate model was not discoverable on the conversation provider")
	}
	return &webSearchMediation{
		cfg:         WebSearchConfig{Enabled: true, MaxSearches: maxSearches, TimeoutMS: 30000, SearchContextSize: "low", MaxResults: 10},
		tool:        models.AnthropicTool{Type: "web_search_20250305", Name: "web_search"},
		delegate:    delegate,
		provider:    provider,
		maxSearches: maxSearches,
		publicModel: "claude-mediated",
	}
}

func TestRunWebSearchLoopKeepsClientUpstreamSendBudgetIntact(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_up1","name":"web_search","input":{"query":"alpha"}}],"usage":{"input_tokens":11,"output_tokens":5}}`,
		`{"id":"msg_2","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"end_turn","stop_sequence":null,"content":[{"type":"text","text":"alpha is a letter"}],"usage":{"input_tokens":29,"output_tokens":9}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchLoopTestHandler(t, upstream.URL)
	ctx, operation, _, err := h.withExplicitRouteOperation(context.Background(), context.Background(), "claude-mediated", providerEndpointMessages)
	if err != nil {
		t.Fatalf("withExplicitRouteOperation() error = %v", err)
	}
	if operation == nil {
		t.Fatal("explicit route operation was not created")
	}

	m := newWebSearchLoopMediation(t, h, 5)
	body := []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"what is alpha"}],"max_tokens":64,"tools":[{"name":"web_search","input_schema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}`)
	message, err := h.runWebSearchLoop(ctx, body, m)
	if err != nil {
		t.Fatalf("runWebSearchLoop() error = %v", err)
	}

	if len(fake.messageBodies) != 2 || fake.responsesCalls != 1 {
		t.Fatalf("messages dispatches = %d, delegated searches = %d; want 2 and 1", len(fake.messageBodies), fake.responsesCalls)
	}
	operation.mu.Lock()
	remaining := operation.remainingUpstreamSends
	operation.mu.Unlock()
	if remaining != 1 {
		t.Fatalf("remainingUpstreamSends = %d; mediated dispatches must not consume the client's fallback budget", remaining)
	}

	if len(message.Content) != 3 ||
		message.Content[0].Type != "server_tool_use" ||
		message.Content[1].Type != "web_search_tool_result" ||
		message.Content[2].Type != "text" {
		t.Fatalf("content = %#v", message.Content)
	}
	if !strings.HasPrefix(message.Content[0].ID, "srvtoolu_") || message.Content[1].ToolUseID != message.Content[0].ID {
		t.Fatalf("synthesized call IDs are not paired: %#v", message.Content[:2])
	}
	if message.Usage.ServerToolUse == nil || message.Usage.ServerToolUse.WebSearchRequests != 1 {
		t.Fatalf("server_tool_use usage = %#v; want 1 delegated request", message.Usage.ServerToolUse)
	}
	if message.Usage.InputTokens != 29 || m.continuationUsage.InputTokens != 11 || m.continuationUsage.OutputTokens != 5 {
		t.Fatalf("emitted usage = %#v, continuation usage = %#v", message.Usage, m.continuationUsage)
	}
	// The second dispatch must replay the assistant turn verbatim and answer the
	// upstream tool_use ID, not the client-facing srvtoolu_ ID.
	second := string(fake.messageBodies[1])
	if !strings.Contains(second, `"toolu_up1"`) || strings.Contains(second, "srvtoolu_") {
		t.Fatalf("continuation body = %s", second)
	}
}

func TestRunWebSearchLoopExhaustionEmitsMaxUsesExceeded(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_a","name":"web_search","input":{"query":"alpha"}},{"type":"tool_use","id":"toolu_b","name":"web_search","input":{"query":"beta"}}],"usage":{"input_tokens":11,"output_tokens":5}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchLoopTestHandler(t, upstream.URL)
	ctx, _, _, err := h.withExplicitRouteOperation(context.Background(), context.Background(), "claude-mediated", providerEndpointMessages)
	if err != nil {
		t.Fatalf("withExplicitRouteOperation() error = %v", err)
	}
	m := newWebSearchLoopMediation(t, h, 1)
	message, err := h.runWebSearchLoop(ctx, []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"two things"}],"max_tokens":64}`), m)
	if err != nil {
		t.Fatalf("runWebSearchLoop() error = %v", err)
	}
	if fake.responsesCalls != 1 {
		t.Fatalf("delegated searches = %d; want exactly max_searches", fake.responsesCalls)
	}
	if len(message.Content) != 4 {
		t.Fatalf("content = %#v", message.Content)
	}
	var errContent struct {
		Type      string `json:"type"`
		ErrorCode string `json:"error_code"`
	}
	if err := json.Unmarshal(message.Content[3].Content, &errContent); err != nil {
		t.Fatalf("unserved result content is not a single object: %s", message.Content[3].Content)
	}
	if errContent.Type != "web_search_tool_result_error" || errContent.ErrorCode != "max_uses_exceeded" {
		t.Fatalf("error content = %#v", errContent)
	}
	if message.Usage.ServerToolUse.WebSearchRequests != 1 {
		t.Fatalf("web_search_requests = %d; unserved calls must not be counted", message.Usage.ServerToolUse.WebSearchRequests)
	}
	// Exhaustion must not leave the upstream's tool_use stop_reason: the client
	// cannot answer a server_tool_use call.
	if message.StopReason == nil || *message.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %v, want end_turn", message.StopReason)
	}
}

func TestRunWebSearchLoopMixedTurnResolvesInlineAndStops(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_s","name":"web_search","input":{"query":"alpha"}},{"type":"tool_use","id":"toolu_c","name":"Read","input":{"path":"/tmp/x"}}],"usage":{"input_tokens":11,"output_tokens":5}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchLoopTestHandler(t, upstream.URL)
	ctx, _, _, err := h.withExplicitRouteOperation(context.Background(), context.Background(), "claude-mediated", providerEndpointMessages)
	if err != nil {
		t.Fatalf("withExplicitRouteOperation() error = %v", err)
	}
	m := newWebSearchLoopMediation(t, h, 5)
	message, err := h.runWebSearchLoop(ctx, []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"mixed"}],"max_tokens":64}`), m)
	if err != nil {
		t.Fatalf("runWebSearchLoop() error = %v", err)
	}
	if len(fake.messageBodies) != 1 {
		t.Fatalf("messages dispatches = %d; a mixed turn must stop the loop", len(fake.messageBodies))
	}
	if len(message.Content) != 3 || message.Content[2].Type != "tool_use" || message.Content[2].Name != "Read" {
		t.Fatalf("content = %#v", message.Content)
	}
	if message.StopReason == nil || *message.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %v; want tool_use", message.StopReason)
	}
}
