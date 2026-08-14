package proxy

import (
	"strings"

	"github.com/sozercan/vekil/models"
)

const (
	// anthropicWebSearchToolName is the tool name Anthropic uses for both the
	// hosted server tool and the client-side stand-in the proxy substitutes for
	// it when mediating searches.
	anthropicWebSearchToolName = "web_search"

	// anthropicHostedWebSearchTypePrefix matches Anthropic's versioned hosted
	// search tool types (web_search_20250305 and successors). The bare name
	// "web_search" is deliberately not a hosted type: an unversioned type is not
	// a shape Anthropic emits, and treating it as hosted would misclassify a
	// client tool.
	anthropicHostedWebSearchTypePrefix = "web_search_"

	// anthropicClientToolType is the explicit type Anthropic allows on a
	// client-defined tool. An omitted type means the same thing.
	anthropicClientToolType = "custom"
)

// hostedWebSearchTool returns the first hosted web-search tool in tools along
// with its index, or false when the request carries none.
func hostedWebSearchTool(tools []models.AnthropicTool) (models.AnthropicTool, int, bool) {
	for index, tool := range tools {
		if strings.HasPrefix(strings.TrimSpace(tool.Type), anthropicHostedWebSearchTypePrefix) {
			return tool, index, true
		}
	}
	return models.AnthropicTool{}, -1, false
}

// clientDefinesWebSearchTool reports whether the client declared its own tool
// named web_search. Mediation declines in that case rather than inventing a
// renaming scheme to disambiguate the intercepted call. Client tools are those
// isAnthropicClientTool accepts — no type, or an explicit "custom" — so the two
// functions must agree on what a client tool is; a request carrying both a
// hosted web_search and a "custom"-typed web_search would otherwise slip past
// this guard and enter mediation with two identically named tools.
func clientDefinesWebSearchTool(tools []models.AnthropicTool) bool {
	for _, tool := range tools {
		if !isAnthropicClientTool(tool) {
			continue
		}
		if tool.Name == anthropicWebSearchToolName {
			return true
		}
	}
	return false
}

// isAnthropicClientTool reports whether a tool is one the client itself will
// answer. Anthropic marks those with no type or with "custom"; every other type
// names a hosted server tool that the upstream, not the client, executes.
func isAnthropicClientTool(tool models.AnthropicTool) bool {
	toolType := strings.TrimSpace(tool.Type)
	return toolType == "" || strings.EqualFold(toolType, anthropicClientToolType)
}
