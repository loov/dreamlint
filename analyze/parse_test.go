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
      "function": "GetUser",
      "code": "query := \"SELECT * FROM users WHERE id=\" + id",
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

	if issues[0].Code != "query := \"SELECT * FROM users WHERE id=\" + id" {
		t.Errorf("code = %q", issues[0].Code)
	}

	if issues[0].Function != "GetUser" {
		t.Errorf("function = %q, want GetUser", issues[0].Function)
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
