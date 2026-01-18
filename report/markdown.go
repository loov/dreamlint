// report/markdown.go
package report

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// WriteMarkdown renders the report as markdown
func WriteMarkdown(r *Report) string {
	var b strings.Builder

	// Title
	b.WriteString("# Code Review Report\n\n")
	b.WriteString(fmt.Sprintf("Generated: %s | Modules: %s\n\n",
		r.Metadata.GeneratedAt.Format("2006-01-02 15:04"),
		strings.Join(r.Metadata.Modules, ", ")))

	// Summary table
	b.WriteString("## Summary\n\n")
	b.WriteString("| Severity | Count |\n")
	b.WriteString("|----------|-------|\n")

	for _, sev := range []string{"critical", "high", "medium", "low", "info"} {
		if count, ok := r.Summary.BySeverity[sev]; ok && count > 0 {
			b.WriteString(fmt.Sprintf("| %s | %d |\n", titleCase(sev), count))
		}
	}
	b.WriteString("\n")

	// Critical issues first
	if len(r.Summary.CriticalUnits) > 0 {
		b.WriteString("## Critical Issues\n\n")
		for _, unitID := range r.Summary.CriticalUnits {
			unit := r.Units[unitID]
			writeUnitIssues(&b, unitID, unit, SeverityCritical)
		}
	}

	// High issues
	highUnits := findUnitsWithSeverity(r, SeverityHigh)
	if len(highUnits) > 0 {
		b.WriteString("## High Priority Issues\n\n")
		for _, unitID := range highUnits {
			unit := r.Units[unitID]
			writeUnitIssues(&b, unitID, unit, SeverityHigh)
		}
	}

	// All functions
	b.WriteString("## All Functions\n\n")

	// Sort unit IDs for deterministic output
	unitIDs := make([]string, 0, len(r.Units))
	for id := range r.Units {
		unitIDs = append(unitIDs, id)
	}
	sort.Strings(unitIDs)

	for _, unitID := range unitIDs {
		unit := r.Units[unitID]
		writeUnitSummary(&b, unitID, unit)
	}

	return b.String()
}

// titleCase capitalizes the first letter of a string
func titleCase(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

func writeUnitIssues(b *strings.Builder, unitID string, unit UnitReport, severity Severity) {
	pos := ""
	if len(unit.Functions) > 0 {
		pos = fmt.Sprintf(" (%s:%d)", unit.Functions[0].Position.Filename, unit.Functions[0].Position.Line)
	}

	b.WriteString(fmt.Sprintf("### %s%s\n\n", unitID, pos))

	for _, issue := range unit.Issues {
		if issue.Severity != severity {
			continue
		}
		b.WriteString(fmt.Sprintf("**[%s] [%s]** %s\n",
			strings.ToUpper(string(issue.Severity)),
			issue.Category,
			issue.Message))

		if issue.Suggestion != "" {
			b.WriteString(fmt.Sprintf("> Suggestion: %s\n", issue.Suggestion))
		}
		b.WriteString("\n")
	}
	b.WriteString("---\n\n")
}

func writeUnitSummary(b *strings.Builder, unitID string, unit UnitReport) {
	b.WriteString(fmt.Sprintf("### %s\n", unitID))

	if unit.Summary.Purpose != "" {
		b.WriteString(fmt.Sprintf("**Purpose:** %s\n", unit.Summary.Purpose))
	}
	if unit.Summary.Behavior != "" {
		b.WriteString(fmt.Sprintf("**Behavior:** %s\n", unit.Summary.Behavior))
	}
	b.WriteString("\n")

	if len(unit.Issues) > 0 {
		b.WriteString("Issues:\n")
		for _, issue := range unit.Issues {
			b.WriteString(fmt.Sprintf("- [%s] %s\n", issue.Severity, issue.Message))
		}
		b.WriteString("\n")
	}
}

func findUnitsWithSeverity(r *Report, severity Severity) []string {
	var units []string
	for unitID, unit := range r.Units {
		for _, issue := range unit.Issues {
			if issue.Severity == severity {
				units = append(units, unitID)
				break
			}
		}
	}
	sort.Strings(units)
	return units
}

// WriteMarkdownFile writes the markdown report to a file
func WriteMarkdownFile(r *Report, path string) error {
	md := WriteMarkdown(r)
	return os.WriteFile(path, []byte(md), 0644)
}
