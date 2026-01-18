// analyze/parse_test.go
package analyze

import (
	"testing"
)

func TestParseSummaryResponse(t *testing.T) {
	response := `{
  "purpose": "Fetches user by ID from database",
  "behavior": "Returns nil, ErrNotFound if user doesn't exist",
  "invariants": ["db must not be nil"],
  "security": ["input id is used in SQL query"]
}`

	summary, err := ParseSummaryResponse(response)
	if err != nil {
		t.Fatalf("ParseSummaryResponse: %v", err)
	}

	if summary.Purpose != "Fetches user by ID from database" {
		t.Errorf("purpose = %q", summary.Purpose)
	}

	if len(summary.Invariants) != 1 {
		t.Errorf("invariants count = %d, want 1", len(summary.Invariants))
	}
}

func TestParseIssuesResponse(t *testing.T) {
	response := `{
  "issues": [
    {
      "line": 42,
      "severity": "high",
      "message": "SQL query built with string concatenation",
      "suggestion": "Use parameterized query"
    }
  ]
}`

	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}

	if len(issues) != 1 {
		t.Fatalf("issues count = %d, want 1", len(issues))
	}

	if issues[0].Severity != "high" {
		t.Errorf("severity = %q, want high", issues[0].Severity)
	}

	if issues[0].Line != 42 {
		t.Errorf("line = %d, want 42", issues[0].Line)
	}
}

func TestParseIssuesResponse_WithMarkdown(t *testing.T) {
	// LLMs sometimes wrap JSON in markdown code blocks
	response := "```json\n" + `{
  "issues": [
    {"line": 10, "severity": "medium", "message": "test"}
  ]
}` + "\n```"

	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}

	if len(issues) != 1 {
		t.Fatalf("issues count = %d, want 1", len(issues))
	}
}

func TestParseIssuesResponse_Empty(t *testing.T) {
	response := `{"issues": []}`
	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}
	if len(issues) != 0 {
		t.Errorf("expected 0 issues, got %d", len(issues))
	}
}

func TestParseSummaryResponse_InvalidJSON(t *testing.T) {
	response := `{invalid json}`
	_, err := ParseSummaryResponse(response)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseIssuesResponse_InvalidJSON(t *testing.T) {
	response := `not json at all`
	_, err := ParseIssuesResponse(response)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestParseIssuesResponse_UnescapedTab(t *testing.T) {
	// LLMs sometimes produce unescaped tabs in strings
	response := "{\"issues\": [{\"line\": 1, \"severity\": \"low\", \"message\": \"use\ttabs\"}]}"
	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}
	if len(issues) != 1 {
		t.Fatalf("issues count = %d, want 1", len(issues))
	}
	if issues[0].Message != "use\ttabs" {
		t.Errorf("message = %q, want %q", issues[0].Message, "use\ttabs")
	}
}

func TestParseIssuesResponse_UnescapedNewline(t *testing.T) {
	// LLMs sometimes produce unescaped newlines in strings
	response := "{\"issues\": [{\"line\": 1, \"severity\": \"low\", \"message\": \"line1\nline2\"}]}"
	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}
	if len(issues) != 1 {
		t.Fatalf("issues count = %d, want 1", len(issues))
	}
}

func TestParseError_Context(t *testing.T) {
	response := `{invalid`
	_, err := ParseSummaryResponse(response)
	if err == nil {
		t.Fatal("expected error")
	}

	parseErr, ok := err.(*ParseError)
	if !ok {
		t.Fatalf("expected *ParseError, got %T", err)
	}

	if parseErr.Response != response {
		t.Errorf("Response = %q, want %q", parseErr.Response, response)
	}
}
