// llm/mock.go
package llm

import (
	"context"
	"sync"
)

// CapturedRequest holds a captured LLM request with its index
type CapturedRequest struct {
	Index   int
	Request Request
}

// MockClient captures all requests and returns predefined responses
type MockClient struct {
	mu        sync.Mutex
	requests  []CapturedRequest
	responses []Response
	respIndex int
}

// NewMockClient creates a mock client with the given responses
func NewMockClient(responses ...Response) *MockClient {
	return &MockClient{
		responses: responses,
	}
}

// Complete captures the request and returns the next predefined response
func (m *MockClient) Complete(ctx context.Context, req Request) (Response, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.requests = append(m.requests, CapturedRequest{
		Index:   len(m.requests),
		Request: req,
	})

	if m.respIndex < len(m.responses) {
		resp := m.responses[m.respIndex]
		m.respIndex++
		return resp, nil
	}

	// Default empty response
	return Response{Content: `{"issues": []}`}, nil
}

// Requests returns all captured requests
func (m *MockClient) Requests() []CapturedRequest {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]CapturedRequest{}, m.requests...)
}

// Prompts returns just the prompt content from all captured requests
func (m *MockClient) Prompts() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	prompts := make([]string, len(m.requests))
	for i, req := range m.requests {
		if len(req.Request.Messages) > 0 {
			prompts[i] = req.Request.Messages[0].Content
		}
	}
	return prompts
}

// Reset clears all captured requests
func (m *MockClient) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests = nil
	m.respIndex = 0
}
