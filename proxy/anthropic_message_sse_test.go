package proxy

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sozercan/vekil/models"
)

func sseEventNames(t *testing.T, body string) []string {
	t.Helper()
	names := make([]string, 0, 16)
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, "event: ") {
			names = append(names, strings.TrimPrefix(line, "event: "))
		}
	}
	return names
}

func sseDataForEvent(t *testing.T, body string, occurrence int) string {
	t.Helper()
	lines := strings.Split(body, "\n")
	seen := 0
	for i, line := range lines {
		if !strings.HasPrefix(line, "event: ") {
			continue
		}
		if seen == occurrence {
			if i+1 >= len(lines) || !strings.HasPrefix(lines[i+1], "data: ") {
				t.Fatalf("event %d has no data line", occurrence)
			}
			return strings.TrimPrefix(lines[i+1], "data: ")
		}
		seen++
	}
	t.Fatalf("event %d not found in %s", occurrence, body)
	return ""
}

func TestWriteAnthropicMessageAsSSEEmitsExactEventOrder(t *testing.T) {
	stop := "end_turn"
	text := "alpha is a letter"
	message := &models.AnthropicResponse{
		ID:         "msg_replay",
		Type:       "message",
		Role:       "assistant",
		Model:      "claude-mediated",
		StopReason: &stop,
		Content: []models.ContentBlock{
			{Type: "server_tool_use", ID: "srvtoolu_AAAAAAAAAAAAAAAAAAAAAA", Name: "web_search", Input: json.RawMessage(`{"query":"alpha"}`)},
			{Type: "web_search_tool_result", ToolUseID: "srvtoolu_AAAAAAAAAAAAAAAAAAAAAA", Content: json.RawMessage(`[{"type":"web_search_result","url":"https://example.com/a","title":"Alpha","encrypted_content":"v1.abc","page_age":"1 day"}]`)},
			{Type: "text", Text: &text},
		},
		Usage: models.AnthropicUsage{InputTokens: 29, OutputTokens: 9, ServerToolUse: &models.AnthropicServerToolUse{WebSearchRequests: 1}},
	}

	rec := httptest.NewRecorder()
	if err := writeAnthropicMessageAsSSE(rec, message); err != nil {
		t.Fatalf("writeAnthropicMessageAsSSE() error = %v", err)
	}
	body := rec.Body.String()

	want := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_stop",
		"content_block_start", "content_block_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}
	got := sseEventNames(t, body)
	if len(got) != len(want) {
		t.Fatalf("event order = %v; want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event[%d] = %q, want %q (full order %v)", i, got[i], want[i], got)
		}
	}
	if rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("Content-Type = %q", rec.Header().Get("Content-Type"))
	}

	// server_tool_use streams its input as input_json_delta.
	var toolDelta models.AnthropicStreamEvent
	if err := json.Unmarshal([]byte(sseDataForEvent(t, body, 2)), &toolDelta); err != nil {
		t.Fatal(err)
	}
	if toolDelta.Delta == nil || toolDelta.Delta.Type != "input_json_delta" || toolDelta.Delta.PartialJSON != `{"query":"alpha"}` {
		t.Fatalf("server_tool_use delta = %#v", toolDelta.Delta)
	}

	// web_search_tool_result is emitted whole inside content_block_start, no deltas.
	var resultStart models.AnthropicStreamEvent
	if err := json.Unmarshal([]byte(sseDataForEvent(t, body, 4)), &resultStart); err != nil {
		t.Fatal(err)
	}
	if resultStart.ContentBlock == nil || resultStart.ContentBlock.Type != "web_search_tool_result" {
		t.Fatalf("result block start = %#v", resultStart.ContentBlock)
	}
	if !strings.Contains(string(resultStart.ContentBlock.Content), "https://example.com/a") {
		t.Fatalf("result block was not emitted whole: %s", resultStart.ContentBlock.Content)
	}
	if resultStart.Index == nil || *resultStart.Index != 1 {
		t.Fatalf("result block index = %v; want 1", resultStart.Index)
	}

	var textDelta models.AnthropicStreamEvent
	if err := json.Unmarshal([]byte(sseDataForEvent(t, body, 7)), &textDelta); err != nil {
		t.Fatal(err)
	}
	if textDelta.Delta == nil || textDelta.Delta.Type != "text_delta" || textDelta.Delta.Text != text {
		t.Fatalf("text delta = %#v", textDelta.Delta)
	}

	var final models.AnthropicStreamEvent
	if err := json.Unmarshal([]byte(sseDataForEvent(t, body, 9)), &final); err != nil {
		t.Fatal(err)
	}
	if final.Delta == nil || final.Delta.StopReason != "end_turn" {
		t.Fatalf("message_delta = %#v", final.Delta)
	}
	if final.Usage == nil || final.Usage.OutputTokens != 9 || final.Usage.ServerToolUse == nil || final.Usage.ServerToolUse.WebSearchRequests != 1 {
		t.Fatalf("message_delta usage = %#v", final.Usage)
	}

	var start models.AnthropicStreamEvent
	if err := json.Unmarshal([]byte(sseDataForEvent(t, body, 0)), &start); err != nil {
		t.Fatal(err)
	}
	if start.Message == nil || len(start.Message.Content) != 0 || start.Message.StopReason != nil {
		t.Fatalf("message_start = %#v", start.Message)
	}
}

func TestWriteAnthropicMessageAsSSEEmitsEmptyTextBlockWithoutDelta(t *testing.T) {
	empty := ""
	rec := httptest.NewRecorder()
	if err := writeAnthropicMessageAsSSE(rec, &models.AnthropicResponse{
		ID: "msg_empty", Type: "message", Role: "assistant", Model: "claude-mediated",
		Content: []models.ContentBlock{{Type: "text", Text: &empty}},
	}); err != nil {
		t.Fatalf("writeAnthropicMessageAsSSE() error = %v", err)
	}
	got := sseEventNames(t, rec.Body.String())
	want := []string{"message_start", "content_block_start", "content_block_stop", "message_delta", "message_stop"}
	if len(got) != len(want) {
		t.Fatalf("event order = %v; want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("event[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}
