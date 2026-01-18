// analyze/schema.go
package analyze

import "github.com/loov/reviewmod/llm"

// SummarySchema is the JSON schema for summary responses
var SummarySchema = &llm.JSONSchema{
	Name: "summary",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"purpose": map[string]any{
				"type":        "string",
				"description": "A brief description of what the function does",
			},
			"behavior": map[string]any{
				"type":        "string",
				"description": "Detailed description of the function's behavior",
			},
			"invariants": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "List of invariants the function maintains",
			},
			"security": map[string]any{
				"type":        "array",
				"items":       map[string]any{"type": "string"},
				"description": "List of security properties",
			},
		},
		"required":             []string{"purpose", "behavior", "invariants", "security"},
		"additionalProperties": false,
	},
}

// IssuesSchema is the JSON schema for analysis pass responses
var IssuesSchema = &llm.JSONSchema{
	Name: "issues",
	Schema: map[string]any{
		"type": "object",
		"properties": map[string]any{
			"issues": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"function": map[string]any{
							"type":        "string",
							"description": "Name of the function where the issue occurs",
						},
						"code": map[string]any{
							"type":        "string",
							"description": "The exact line of code where the issue occurs, copied verbatim from the code block",
						},
						"severity": map[string]any{
							"type":        "string",
							"enum":        []string{"critical", "important", "minor"},
							"description": "Severity of the issue",
						},
						"message": map[string]any{
							"type":        "string",
							"description": "Description of the issue",
						},
						"suggestion": map[string]any{
							"type":        "string",
							"description": "Suggested fix for the issue",
						},
					},
					"required":             []string{"function", "code", "severity", "message"},
					"additionalProperties": false,
				},
			},
		},
		"required":             []string{"issues"},
		"additionalProperties": false,
	},
}
