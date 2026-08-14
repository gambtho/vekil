package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sozercan/vekil/auth"
	"github.com/sozercan/vekil/logger"
)

// newWebSearchIngressTestHandler builds the PRODUCTION mediation shape, which is
// not the shape the loop suite uses. webSearchMediationFor only activates for a
// Copilot-owned conversation model, and providerKindSupportsExplicitEndpoint
// refuses /v1/messages for Copilot targets, so a mediated turn always runs on a
// legacy route with a NIL route operation. Everything on this path must be
// nil-safe; the loop suite's explicit anthropic-compatible route never proves it.
func newWebSearchIngressTestHandler(t *testing.T, baseURL string) *ProxyHandler {
	t.Helper()
	h, err := NewProxyHandler(
		auth.NewTestAuthenticator("fixture-token"),
		logger.NewWithWriter(logger.LevelError, io.Discard),
		WithCopilotBaseURL(baseURL),
	)
	if err != nil {
		t.Fatalf("NewProxyHandler() error = %v", err)
	}
	t.Cleanup(h.BeginShutdown)
	return h
}

func enabledWebSearchIngressConfig() WebSearchConfig {
	return WebSearchConfig{Enabled: true, DelegateModel: "gpt-delegate", MaxSearches: 5, TimeoutMS: 30000, SearchContextSize: "low", MaxResults: 10}
}

func TestHandleAnthropicMessagesWebSearchDisabledForwardsExactBytes(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_x","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"end_turn","stop_sequence":null,"content":[{"type":"text","text":"no mediation"}],"usage":{"input_tokens":3,"output_tokens":2}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchIngressTestHandler(t, upstream.URL) // web_search disabled by default
	body := []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}`)
	rec := httptest.NewRecorder()
	h.HandleAnthropicMessages(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want = %d, body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if len(fake.messageBodies) != 1 {
		t.Fatalf("messages dispatches got = %d, want = 1", len(fake.messageBodies))
	}
	if !bytes.Equal(fake.messageBodies[0], body) {
		t.Fatalf("disabled mediation rewrote the body: got = %s, want = %s", fake.messageBodies[0], body)
	}
}

func TestHandleAnthropicMessagesWebSearchDelegationFailureFallsBackToOriginalBody(t *testing.T) {
	var messageBodies [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case providerEndpointModels:
			_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{
				map[string]any{"id": "claude-mediated", "supported_endpoints": []string{providerEndpointMessages}},
				map[string]any{"id": "gpt-delegate", "supported_endpoints": []string{providerEndpointResponses}},
			}})
		case providerEndpointResponses:
			http.Error(w, "delegation is unavailable", http.StatusServiceUnavailable)
		case providerEndpointMessages:
			read, _ := io.ReadAll(r.Body)
			messageBodies = append(messageBodies, read)
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_t","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_a","name":"web_search","input":{"query":"alpha"}}],"usage":{"input_tokens":11,"output_tokens":5}}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	h := newWebSearchIngressTestHandler(t, upstream.URL)
	h.webSearch = enabledWebSearchIngressConfig()
	body := []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"alpha"}],"max_tokens":64,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}`)
	rec := httptest.NewRecorder()
	h.HandleAnthropicMessages(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want = %d, body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}
	if len(messageBodies) != 2 {
		t.Fatalf("messages dispatches got = %d, want = 2 (mediated turn plus passthrough fallback)", len(messageBodies))
	}
	if !bytes.Equal(messageBodies[1], body) {
		t.Fatalf("fallback did not resend the original body: got = %s, want = %s", messageBodies[1], body)
	}
	if !strings.Contains(rec.Body.String(), `"tool_use"`) {
		t.Fatalf("fallback response got = %s, want = the untouched passthrough turn", rec.Body.String())
	}
}

func TestHandleAnthropicMessagesWebSearchHappyPathEmitsSynthesizedBlocks(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_a","name":"web_search","input":{"query":"alpha"}}],"usage":{"input_tokens":11,"output_tokens":5}}`,
		`{"id":"msg_2","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"end_turn","stop_sequence":null,"content":[{"type":"text","text":"alpha is a letter"}],"usage":{"input_tokens":29,"output_tokens":9}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchIngressTestHandler(t, upstream.URL)
	h.webSearch = enabledWebSearchIngressConfig()
	body := []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"alpha"}],"max_tokens":64,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}`)
	rec := httptest.NewRecorder()
	h.HandleAnthropicMessages(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want = %d, body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}

	var message struct {
		Model   string `json:"model"`
		Content []struct {
			Type      string `json:"type"`
			ID        string `json:"id"`
			ToolUseID string `json:"tool_use_id"`
			Name      string `json:"name"`
		} `json:"content"`
		Usage struct {
			ServerToolUse *struct {
				WebSearchRequests int `json:"web_search_requests"`
			} `json:"server_tool_use"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &message); err != nil {
		t.Fatalf("response is not a valid Anthropic message: got err = %v, body = %s", err, rec.Body.String())
	}
	if len(message.Content) != 3 ||
		message.Content[0].Type != "server_tool_use" || message.Content[0].Name != "web_search" ||
		message.Content[1].Type != "web_search_tool_result" || message.Content[1].ToolUseID != message.Content[0].ID ||
		message.Content[2].Type != "text" {
		t.Fatalf("content got = %#v, want = server_tool_use, web_search_tool_result, text", message.Content)
	}
	if message.Usage.ServerToolUse == nil || message.Usage.ServerToolUse.WebSearchRequests != 1 {
		t.Fatalf("server_tool_use usage got = %#v, want = web_search_requests 1", message.Usage.ServerToolUse)
	}
	if message.Model != "claude-mediated" {
		t.Fatalf("model got = %q, want = %q", message.Model, "claude-mediated")
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type got = %q, want = %q", got, "application/json")
	}
	// The upstream never saw the hosted tool; it saw the client stand-in.
	if bytes.Contains(fake.messageBodies[0], []byte("web_search_20250305")) {
		t.Fatalf("mediated dispatch forwarded the hosted tool: got = %s", fake.messageBodies[0])
	}
}

func TestHandleAnthropicMessagesWebSearchStreamingClientGetsSSEReplay(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_a","name":"web_search","input":{"query":"alpha"}}],"usage":{"input_tokens":11,"output_tokens":5}}`,
		`{"id":"msg_2","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"end_turn","stop_sequence":null,"content":[{"type":"text","text":"alpha is a letter"}],"usage":{"input_tokens":29,"output_tokens":9}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchIngressTestHandler(t, upstream.URL)
	h.webSearch = enabledWebSearchIngressConfig()
	body := []byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"alpha"}],"max_tokens":64,"stream":true,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}`)
	rec := httptest.NewRecorder()
	h.HandleAnthropicMessages(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want = %d", rec.Code, http.StatusOK)
	}
	out := rec.Body.String()
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type got = %q, want = %q", got, "text/event-stream")
	}
	for _, want := range []string{"event: message_start", "event: content_block_start", "event: message_delta", "event: message_stop", `"type":"web_search_tool_result"`} {
		if !strings.Contains(out, want) {
			t.Fatalf("stream got = %s, want = a stream containing %q", out, want)
		}
	}
	// The upstream must still have been driven non-streaming.
	for _, sent := range fake.messageBodies {
		if strings.Contains(string(sent), `"stream":true`) {
			t.Fatalf("mediated dispatch got = %s, want = a dispatch without stream:true", sent)
		}
	}
}

func TestHandleAnthropicMessagesWebSearchAccountsInternalSpend(t *testing.T) {
	fake := &webSearchLoopUpstream{turns: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"tool_use","stop_sequence":null,"content":[{"type":"tool_use","id":"toolu_a","name":"web_search","input":{"query":"alpha"}}],"usage":{"input_tokens":11,"output_tokens":5}}`,
		`{"id":"msg_2","type":"message","role":"assistant","model":"claude-mediated","stop_reason":"end_turn","stop_sequence":null,"content":[{"type":"text","text":"alpha is a letter"}],"usage":{"input_tokens":29,"output_tokens":9}}`,
	}}
	upstream := httptest.NewServer(fake.handler())
	defer upstream.Close()

	h := newWebSearchIngressTestHandler(t, upstream.URL)
	h.webSearch = enabledWebSearchIngressConfig()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader([]byte(`{"model":"claude-mediated","messages":[{"role":"user","content":"alpha"}],"max_tokens":64,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}]}`)))
	summaryCtx, summary := WithRequestSummary(req.Context())
	req = req.WithContext(summaryCtx)
	rec := httptest.NewRecorder()
	h.HandleAnthropicMessages(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want = %d, body = %s", rec.Code, http.StatusOK, rec.Body.String())
	}
	// Turn 1 (11/5) plus the delegated /responses call (7/3) are internal spend;
	// the emitted turn (29/9) rides on the response body instead.
	summary.mu.Lock()
	extraPrompt, extraCompletion := summary.extraPromptTokens, summary.extraCompletionTokens
	summary.mu.Unlock()
	if extraPrompt != 18 || extraCompletion != 8 {
		t.Fatalf("internal usage got = %d/%d, want = 18/8", extraPrompt, extraCompletion)
	}
	// Only the delegated /responses send is counted here. The two mediated
	// Messages dispatches run on a legacy Copilot route, which has no route
	// operation and therefore never reaches RecordUpstreamAttempt — the same as
	// any other legacy Copilot dispatch today. Counting them separately would
	// double-count the moment Copilot gains an explicit /v1/messages route.
	if got := summary.UpstreamSendCount(); got != 1 {
		t.Fatalf("upstream_sends got = %d, want = 1", got)
	}
}
