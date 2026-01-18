// llm/client.go
package llm

import "context"

// Client is the interface for LLM backends
type Client interface {
	Complete(ctx context.Context, req Request) (Response, error)
}

// Request holds the input for an LLM completion
type Request struct {
	System   string
	Messages []Message
	Config   ModelConfig
}

// Message is a single message in the conversation
type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

// ModelConfig holds model-specific settings
type ModelConfig struct {
	Model       string
	MaxTokens   int
	Temperature float64
	JSONSchema  *JSONSchema
}

// JSONSchema defines the expected response structure
type JSONSchema struct {
	Name   string
	Schema map[string]any
}

// Response holds the LLM output
type Response struct {
	Content string
	Usage   Usage
}

// Usage tracks token consumption
type Usage struct {
	PromptTokens     int
	CompletionTokens int
}
