package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/sozercan/vekil/models"
)

// writeAnthropicMessageAsSSE replays a finished Anthropic message as a valid
// Anthropic SSE stream. anthropicStreamState is not reusable here: its input
// surface is OpenAI-shaped and it knows only text and tool_use blocks.
//
// Block emission follows the Anthropic contract: text streams via text_delta,
// server_tool_use streams its input via input_json_delta, and every other block
// type — web_search_tool_result in particular — is emitted whole inside
// content_block_start with no deltas.
func writeAnthropicMessageAsSSE(w http.ResponseWriter, resp *models.AnthropicResponse) error {
	if resp == nil {
		return fmt.Errorf("anthropic message is required")
	}
	setSSEHeaders(w)
	w.WriteHeader(http.StatusOK)

	start := *resp
	start.Content = []models.ContentBlock{}
	start.StopReason = nil
	start.StopSequence = nil
	startUsage := resp.Usage
	startUsage.OutputTokens = 0
	start.Usage = startUsage
	if err := writeSSEEvent(w, "message_start", models.AnthropicStreamEvent{
		Type:    "message_start",
		Message: &start,
	}); err != nil {
		return err
	}

	for index, block := range resp.Content {
		if err := writeAnthropicContentBlockAsSSE(w, index, block); err != nil {
			return err
		}
	}

	delta := &models.AnthropicDelta{}
	if resp.StopReason != nil {
		delta.StopReason = *resp.StopReason
	}
	if resp.StopSequence != nil {
		delta.StopSequence = *resp.StopSequence
	}
	usage := resp.Usage
	if err := writeSSEEvent(w, "message_delta", models.AnthropicStreamEvent{
		Type:  "message_delta",
		Delta: delta,
		Usage: &usage,
	}); err != nil {
		return err
	}
	return writeSSEEvent(w, "message_stop", models.AnthropicStreamEvent{Type: "message_stop"})
}

func writeAnthropicContentBlockAsSSE(w http.ResponseWriter, index int, block models.ContentBlock) error {
	switch block.Type {
	case "text":
		text := ""
		if block.Text != nil {
			text = *block.Text
		}
		empty := ""
		opening := block
		opening.Text = &empty
		if err := writeSSEEvent(w, "content_block_start", models.AnthropicStreamEvent{
			Type:         "content_block_start",
			Index:        intVal(index),
			ContentBlock: &opening,
		}); err != nil {
			return err
		}
		if text != "" {
			if err := writeSSEEvent(w, "content_block_delta", models.AnthropicStreamEvent{
				Type:  "content_block_delta",
				Index: intVal(index),
				Delta: &models.AnthropicDelta{Type: "text_delta", Text: text},
			}); err != nil {
				return err
			}
		}
	case "server_tool_use", "tool_use":
		input := string(block.Input)
		opening := block
		opening.Input = json.RawMessage(`{}`)
		if err := writeSSEEvent(w, "content_block_start", models.AnthropicStreamEvent{
			Type:         "content_block_start",
			Index:        intVal(index),
			ContentBlock: &opening,
		}); err != nil {
			return err
		}
		if input != "" && input != "{}" {
			if err := writeSSEEvent(w, "content_block_delta", models.AnthropicStreamEvent{
				Type:  "content_block_delta",
				Index: intVal(index),
				Delta: &models.AnthropicDelta{Type: "input_json_delta", PartialJSON: input},
			}); err != nil {
				return err
			}
		}
	default:
		// web_search_tool_result and any other complete block: whole, no deltas.
		whole := block
		if err := writeSSEEvent(w, "content_block_start", models.AnthropicStreamEvent{
			Type:         "content_block_start",
			Index:        intVal(index),
			ContentBlock: &whole,
		}); err != nil {
			return err
		}
	}
	return writeSSEEvent(w, "content_block_stop", models.AnthropicStreamEvent{
		Type:  "content_block_stop",
		Index: intVal(index),
	})
}
