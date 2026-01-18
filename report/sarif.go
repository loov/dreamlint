// report/sarif.go
package report

import (
	"encoding/json"
	"os"
)

// SARIF format version 2.1.0
// https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html

// SARIFReport is the root SARIF object
type SARIFReport struct {
	Schema  string     `json:"$schema"`
	Version string     `json:"version"`
	Runs    []SARIFRun `json:"runs"`
}

// SARIFRun represents a single analysis run
type SARIFRun struct {
	Tool    SARIFTool     `json:"tool"`
	Results []SARIFResult `json:"results"`
}

// SARIFTool describes the analysis tool
type SARIFTool struct {
	Driver SARIFDriver `json:"driver"`
}

// SARIFDriver describes the tool driver
type SARIFDriver struct {
	Name           string      `json:"name"`
	Version        string      `json:"version"`
	InformationURI string      `json:"informationUri,omitempty"`
	Rules          []SARIFRule `json:"rules,omitempty"`
}

// SARIFRule describes a rule/category
type SARIFRule struct {
	ID               string       `json:"id"`
	Name             string       `json:"name"`
	ShortDescription SARIFMessage `json:"shortDescription"`
}

// SARIFResult represents a single issue
type SARIFResult struct {
	RuleID    string          `json:"ruleId"`
	Level     string          `json:"level"`
	Message   SARIFMessage    `json:"message"`
	Locations []SARIFLocation `json:"locations,omitempty"`
	Fixes     []SARIFFix      `json:"fixes,omitempty"`
}

// SARIFMessage holds a text message
type SARIFMessage struct {
	Text string `json:"text"`
}

// SARIFLocation describes where an issue occurs
type SARIFLocation struct {
	PhysicalLocation SARIFPhysicalLocation `json:"physicalLocation"`
}

// SARIFPhysicalLocation describes file location
type SARIFPhysicalLocation struct {
	ArtifactLocation SARIFArtifactLocation `json:"artifactLocation"`
	Region           *SARIFRegion          `json:"region,omitempty"`
}

// SARIFArtifactLocation describes the file
type SARIFArtifactLocation struct {
	URI string `json:"uri"`
}

// SARIFRegion describes the position in the file
type SARIFRegion struct {
	StartLine   int `json:"startLine"`
	StartColumn int `json:"startColumn,omitempty"`
}

// SARIFFix describes a suggested fix
type SARIFFix struct {
	Description SARIFMessage `json:"description"`
}

// severityToLevel converts our severity to SARIF level
func severityToLevel(s Severity) string {
	switch s {
	case SeverityCritical:
		return "error"
	case "important":
		return "warning"
	case "minor":
		return "note"
	default:
		return "note"
	}
}

// ToSARIF converts a Report to SARIF format
func ToSARIF(r *Report) *SARIFReport {
	// Collect unique categories as rules
	categories := make(map[string]bool)
	for _, unit := range r.Units {
		for _, issue := range unit.Issues {
			categories[issue.Category] = true
		}
	}

	rules := make([]SARIFRule, 0, len(categories))
	for cat := range categories {
		rules = append(rules, SARIFRule{
			ID:               cat,
			Name:             cat,
			ShortDescription: SARIFMessage{Text: cat + " analysis"},
		})
	}

	// Convert issues to results
	var results []SARIFResult
	for _, unit := range r.Units {
		for _, issue := range unit.Issues {
			result := SARIFResult{
				RuleID:  issue.Category,
				Level:   severityToLevel(issue.Severity),
				Message: SARIFMessage{Text: issue.Message},
			}

			if issue.Position.Filename != "" {
				result.Locations = []SARIFLocation{{
					PhysicalLocation: SARIFPhysicalLocation{
						ArtifactLocation: SARIFArtifactLocation{
							URI: issue.Position.Filename,
						},
						Region: &SARIFRegion{
							StartLine:   issue.Position.Line,
							StartColumn: issue.Position.Column,
						},
					},
				}}
			}

			if issue.Suggestion != "" {
				result.Fixes = []SARIFFix{{
					Description: SARIFMessage{Text: issue.Suggestion},
				}}
			}

			results = append(results, result)
		}
	}

	return &SARIFReport{
		Schema:  "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
		Version: "2.1.0",
		Runs: []SARIFRun{{
			Tool: SARIFTool{
				Driver: SARIFDriver{
					Name:    "reviewmod",
					Version: "0.1.0",
					Rules:   rules,
				},
			},
			Results: results,
		}},
	}
}

// WriteSARIF serializes the report to SARIF JSON
func WriteSARIF(r *Report) ([]byte, error) {
	sarif := ToSARIF(r)
	return json.MarshalIndent(sarif, "", "  ")
}

// WriteSARIFFile writes the report to a SARIF file
func WriteSARIFFile(r *Report, path string) error {
	data, err := WriteSARIF(r)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
