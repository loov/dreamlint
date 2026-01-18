// analyze/parse.go
package analyze

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// codeBlockRegex matches markdown code blocks with any language specifier
var codeBlockRegex = regexp.MustCompile("(?s)```\\w*\\s*(.+?)\\s*```")

// SummaryResponse is the expected JSON structure for summary pass
type SummaryResponse struct {
	Purpose    string   `json:"purpose"`
	Behavior   string   `json:"behavior"`
	Invariants []string `json:"invariants"`
	Security   []string `json:"security"`
}

// IssueResponse is a single issue from the LLM
type IssueResponse struct {
	Line       int    `json:"line"`
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
	cleaned := cleanJSON(response)

	var summary SummaryResponse
	if err := json.Unmarshal([]byte(cleaned), &summary); err != nil {
		return nil, &ParseError{
			Err:      err,
			Response: response,
			Cleaned:  cleaned,
		}
	}

	return &summary, nil
}

// ParseIssuesResponse parses the LLM response for an analysis pass
func ParseIssuesResponse(response string) ([]IssueResponse, error) {
	cleaned := cleanJSON(response)

	// Handle empty response
	if cleaned == "" || cleaned == "{}" {
		return nil, nil
	}

	var issues IssuesResponse
	if err := json.Unmarshal([]byte(cleaned), &issues); err != nil {
		return nil, &ParseError{
			Err:      err,
			Response: response,
			Cleaned:  cleaned,
		}
	}

	return issues.Issues, nil
}

// cleanJSON extracts JSON from markdown code blocks and fixes common LLM issues
func cleanJSON(s string) string {
	s = strings.TrimSpace(s)

	// Extract from markdown code blocks
	if matches := codeBlockRegex.FindStringSubmatch(s); len(matches) > 1 {
		s = strings.TrimSpace(matches[1])
	}

	// Fix common LLM JSON issues:
	// 1. Replace literal tabs/newlines in string values with escaped versions
	s = fixStringLiterals(s)

	return s
}

// fixStringLiterals attempts to fix unescaped control characters in JSON strings
func fixStringLiterals(s string) string {
	var result strings.Builder
	inString := false
	escaped := false

	for i := 0; i < len(s); i++ {
		c := s[i]

		if escaped {
			result.WriteByte(c)
			escaped = false
			continue
		}

		if c == '\\' && inString {
			result.WriteByte(c)
			escaped = true
			continue
		}

		if c == '"' {
			inString = !inString
			result.WriteByte(c)
			continue
		}

		if inString {
			// Replace unescaped control characters
			switch c {
			case '\t':
				result.WriteString("\\t")
			case '\n':
				result.WriteString("\\n")
			case '\r':
				result.WriteString("\\r")
			default:
				result.WriteByte(c)
			}
		} else {
			result.WriteByte(c)
		}
	}

	return result.String()
}

// ParseError provides context when JSON parsing fails
type ParseError struct {
	Err      error
	Response string
	Cleaned  string
}

func (e *ParseError) Error() string {
	// Truncate long responses for readable errors
	preview := e.Cleaned
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	return fmt.Sprintf("failed to parse LLM response: %v\nResponse preview: %s", e.Err, preview)
}

func (e *ParseError) Unwrap() error {
	return e.Err
}
