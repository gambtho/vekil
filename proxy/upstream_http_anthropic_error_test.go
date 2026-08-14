package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
)

func newUpstreamErrorResponse(t *testing.T, status int, contentType, body string) *http.Response {
	t.Helper()

	header := make(http.Header)
	if contentType != "" {
		header.Set("Content-Type", contentType)
	}
	header.Set("Content-Length", strconv.Itoa(len(body)))
	header.Set("X-Request-Id", "upstream-req-1")
	return &http.Response{
		StatusCode: status,
		Header:     header,
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestWriteAnthropicUpstreamErrorResponseWrapsOpenAIShapedBody(t *testing.T) {
	t.Parallel()

	body := `{"error":{"message":"The use of the web search tool is not supported.","code":"unsupported_value"}}`
	resp := newUpstreamErrorResponse(t, http.StatusBadRequest, "application/json", body)
	recorder := httptest.NewRecorder()

	if err := writeAnthropicUpstreamErrorResponse(recorder, resp); err != nil {
		t.Fatalf("writeAnthropicUpstreamErrorResponse: got err=%v, want nil", err)
	}
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status: got=%d, want=%d", recorder.Code, http.StatusBadRequest)
	}

	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode wrapped body %s: %v", recorder.Body.Bytes(), err)
	}
	if envelope.Type != "error" {
		t.Fatalf("type: got=%q, want=%q", envelope.Type, "error")
	}
	if envelope.Error.Type != "invalid_request_error" {
		t.Fatalf("error.type: got=%q, want=%q", envelope.Error.Type, "invalid_request_error")
	}
	if envelope.Error.Message != "The use of the web search tool is not supported." {
		t.Fatalf("error.message: got=%q, want=%q", envelope.Error.Message,
			"The use of the web search tool is not supported.")
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type: got=%q, want=%q", got, "application/json")
	}
	if got := recorder.Header().Get("Content-Length"); got != strconv.Itoa(recorder.Body.Len()) {
		t.Fatalf("Content-Length: got=%q, want=%q", got, strconv.Itoa(recorder.Body.Len()))
	}
	if got := recorder.Header().Get("X-Request-Id"); got != "upstream-req-1" {
		t.Fatalf("X-Request-Id: got=%q, want=%q", got, "upstream-req-1")
	}
}

func TestWriteAnthropicUpstreamErrorResponseRelaysAnthropicEnvelopeVerbatim(t *testing.T) {
	t.Parallel()

	body := `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"},"request_id":"req_9"}`
	resp := newUpstreamErrorResponse(t, http.StatusTooManyRequests, "application/json", body)
	recorder := httptest.NewRecorder()

	if err := writeAnthropicUpstreamErrorResponse(recorder, resp); err != nil {
		t.Fatalf("writeAnthropicUpstreamErrorResponse: got err=%v, want nil", err)
	}
	if recorder.Code != http.StatusTooManyRequests {
		t.Fatalf("status: got=%d, want=%d", recorder.Code, http.StatusTooManyRequests)
	}
	if recorder.Body.String() != body {
		t.Fatalf("body: got=%s, want=%s", recorder.Body.String(), body)
	}
}

func TestWriteAnthropicUpstreamErrorResponseWrapsNonJSONBody(t *testing.T) {
	t.Parallel()

	resp := newUpstreamErrorResponse(t, http.StatusBadGateway, "text/plain", "  upstream exploded\n")
	recorder := httptest.NewRecorder()

	if err := writeAnthropicUpstreamErrorResponse(recorder, resp); err != nil {
		t.Fatalf("writeAnthropicUpstreamErrorResponse: got err=%v, want nil", err)
	}
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status: got=%d, want=%d", recorder.Code, http.StatusBadGateway)
	}

	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode wrapped body %s: %v", recorder.Body.Bytes(), err)
	}
	if envelope.Error.Type != "api_error" {
		t.Fatalf("error.type: got=%q, want=%q", envelope.Error.Type, "api_error")
	}
	if envelope.Error.Message != "upstream exploded" {
		t.Fatalf("error.message: got=%q, want=%q", envelope.Error.Message, "upstream exploded")
	}
	if got := recorder.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type: got=%q, want=%q", got, "application/json")
	}
}

func TestWriteAnthropicUpstreamErrorResponseWrapsEmptyBody(t *testing.T) {
	t.Parallel()

	resp := newUpstreamErrorResponse(t, http.StatusServiceUnavailable, "", "")
	recorder := httptest.NewRecorder()

	if err := writeAnthropicUpstreamErrorResponse(recorder, resp); err != nil {
		t.Fatalf("writeAnthropicUpstreamErrorResponse: got err=%v, want nil", err)
	}

	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode wrapped body %s: %v", recorder.Body.Bytes(), err)
	}
	if envelope.Error.Type != "overloaded_error" {
		t.Fatalf("error.type: got=%q, want=%q", envelope.Error.Type, "overloaded_error")
	}
	if envelope.Error.Message != "upstream returned HTTP 503" {
		t.Fatalf("error.message: got=%q, want=%q", envelope.Error.Message, "upstream returned HTTP 503")
	}
}

func TestWriteAnthropicUpstreamErrorResponseRelaysOversizedBodyVerbatim(t *testing.T) {
	t.Parallel()

	oversized := append([]byte(`{"error":{"message":"`), bytes.Repeat([]byte("x"), anthropicUpstreamErrorMaxBytes+1)...)
	oversized = append(oversized, []byte(`"}}`)...)
	resp := newUpstreamErrorResponse(t, http.StatusBadRequest, "application/json", string(oversized))
	recorder := httptest.NewRecorder()

	if err := writeAnthropicUpstreamErrorResponse(recorder, resp); err != nil {
		t.Fatalf("writeAnthropicUpstreamErrorResponse: got err=%v, want nil", err)
	}
	if recorder.Body.Len() != len(oversized) {
		t.Fatalf("body length: got=%d, want=%d", recorder.Body.Len(), len(oversized))
	}
	if !bytes.Equal(recorder.Body.Bytes(), oversized) {
		t.Fatalf("oversized body was not relayed verbatim")
	}
}
