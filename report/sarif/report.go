package sarif

import (
	"encoding/json"
	"os"

	"github.com/loov/dreamlint/report"
)

// Write serializes the report to SARIF JSON
func Write(r *report.Report) ([]byte, error) {
	s := FromReport(r)
	return json.MarshalIndent(s, "", "  ")
}

// WriteFile writes the report to a SARIF file
func WriteFile(r *report.Report, path string) error {
	data, err := Write(r)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// FromReport converts a Report to SARIF format
func FromReport(r *report.Report) *Report {
	// Collect unique categories as rules
	categories := make(map[string]bool)
	for _, unit := range r.Units {
		for _, issue := range unit.Issues {
			categories[issue.Category] = true
		}
	}

	rules := make([]Rule, 0, len(categories))
	for cat := range categories {
		rules = append(rules, Rule{
			ID:               cat,
			Name:             cat,
			ShortDescription: Message{Text: cat + " analysis"},
		})
	}

	// Convert issues to results
	var results []Result
	for _, unit := range r.Units {
		for _, issue := range unit.Issues {
			result := Result{
				RuleID:  issue.Category,
				Level:   severityToLevel(issue.Severity),
				Message: Message{Text: issue.Message},
			}

			if issue.Position.Filename != "" {
				result.Locations = []Location{{
					PhysicalLocation: PhysicalLocation{
						ArtifactLocation: ArtifactLocation{
							URI: issue.Position.Filename,
						},
						Region: &Region{
							StartLine:   issue.Position.Line,
							StartColumn: issue.Position.Column,
						},
					},
				}}
			}

			if issue.Suggestion != "" {
				result.Fixes = []Fix{{
					Description: Message{Text: issue.Suggestion},
				}}
			}

			results = append(results, result)
		}
	}

	return &Report{
		Schema:  "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
		Version: "2.1.0",
		Runs: []Run{{
			Tool: Tool{
				Driver: Driver{
					Name:    "dreamlint",
					Version: "0.1.0",
					Rules:   rules,
				},
			},
			Results: results,
		}},
	}
}

// severityToLevel converts our severity to SARIF level
func severityToLevel(s report.Severity) string {
	switch s {
	case report.SeverityCritical, report.SeverityHigh:
		return "error"
	case report.SeverityMedium:
		return "warning"
	case report.SeverityLow, report.SeverityInfo:
		return "note"
	default:
		return "note"
	}
}
