package analyze

import (
	"encoding/json"
	"fmt"
)

// SummaryResponse is the expected JSON structure for summary pass
type SummaryResponse struct {
	Purpose    string   `json:"purpose"`
	Behavior   string   `json:"behavior"`
	Invariants []string `json:"invariants"`
	Security   []string `json:"security"`
}

// IssueResponse is a single issue from the LLM
type IssueResponse struct {
	Function   string `json:"function"`
	Line       int    `json:"line"`
	Code       string `json:"code"`
	Severity   string `json:"severity"`
	Message    string `json:"message"`
	Suggestion string `json:"suggestion,omitempty"`
}

// IssuesResponse is the expected JSON structure for analysis passes
type IssuesResponse struct {
	Issues []IssueResponse `json:"issues"`
}

// ParseSummaryResponse parses the LLM response for a summary pass
func ParseSummaryResponse(response string) (*SummaryResponse, error) {
	var summary SummaryResponse
	if err := json.Unmarshal([]byte(response), &summary); err != nil {
		return nil, &ParseError{
			Err:      err,
			Response: response,
		}
	}

	return &summary, nil
}

// ParseIssuesResponse parses the LLM response for an analysis pass
func ParseIssuesResponse(response string) ([]IssueResponse, error) {
	var issues IssuesResponse
	if err := json.Unmarshal([]byte(response), &issues); err != nil {
		return nil, &ParseError{
			Err:      err,
			Response: response,
		}
	}

	return issues.Issues, nil
}

// ParseError provides context when JSON parsing fails
type ParseError struct {
	Err      error
	Response string
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("failed to parse LLM response: %v\nResponse:\n%s", e.Err, e.Response)
}

func (e *ParseError) Unwrap() error {
	return e.Err
}
