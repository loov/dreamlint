package report

import (
	"go/token"
	"time"
)

// Severity levels
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
	SeverityInfo     Severity = "info"
)

// SeverityStrings is a list of all severity levels
var SeverityStrings = []string{
	string(SeverityCritical),
	string(SeverityHigh),
	string(SeverityMedium),
	string(SeverityLow),
	string(SeverityInfo),
}

// Report is the complete analysis report
type Report struct {
	Metadata Metadata              `json:"metadata"`
	Units    map[string]UnitReport `json:"units"`
	Summary  Summary               `json:"summary"`
}

// Metadata holds report metadata
type Metadata struct {
	GeneratedAt   time.Time `json:"generated_at"`
	Modules       []string  `json:"modules"`
	ConfigFiles   []string  `json:"config_files"`
	InlineConfigs []string  `json:"inline_configs,omitempty"`
	TotalUnits    int       `json:"total_units"`
	CacheHits     int       `json:"cache_hits"`
}

// UnitReport holds analysis results for a single unit
type UnitReport struct {
	Functions []FunctionInfo  `json:"functions"`
	Summary   FunctionSummary `json:"summary"`
	Issues    []Issue         `json:"issues"`
}

// FunctionInfo holds function metadata
type FunctionInfo struct {
	Package   string         `json:"package"`
	Name      string         `json:"name"`
	Receiver  string         `json:"receiver,omitempty"`
	Signature string         `json:"signature"`
	Position  token.Position `json:"position"`
}

// FunctionSummary describes function behavior
type FunctionSummary struct {
	Purpose    string   `json:"purpose"`
	Behavior   string   `json:"behavior"`
	Invariants []string `json:"invariants,omitempty"`
	Security   []string `json:"security,omitempty"`
}

// Issue represents a found problem
type Issue struct {
	Position   token.Position `json:"position"`
	Severity   Severity       `json:"severity"`
	Category   string         `json:"category"`
	Message    string         `json:"message"`
	Snippet    string         `json:"snippet,omitempty"`
	Suggestion string         `json:"suggestion,omitempty"`
}

// Summary aggregates issue counts
type Summary struct {
	TotalIssues   int            `json:"total_issues"`
	BySeverity    map[string]int `json:"by_severity"`
	ByCategory    map[string]int `json:"by_category"`
	CriticalUnits []string       `json:"critical_units"`
}

// NewReport creates a new empty report
func NewReport() *Report {
	return &Report{
		Metadata: Metadata{
			GeneratedAt: time.Now(),
		},
		Units: make(map[string]UnitReport),
		Summary: Summary{
			BySeverity: make(map[string]int),
			ByCategory: make(map[string]int),
		},
	}
}

// AddIssue adds an issue to a unit and updates summary
func (r *Report) AddIssue(unitID string, issue Issue) {
	unit := r.Units[unitID]
	unit.Issues = append(unit.Issues, issue)
	r.Units[unitID] = unit

	r.Summary.TotalIssues++
	r.Summary.BySeverity[string(issue.Severity)]++
	r.Summary.ByCategory[issue.Category]++

	if issue.Severity == SeverityCritical {
		found := false
		for _, u := range r.Summary.CriticalUnits {
			if u == unitID {
				found = true
				break
			}
		}
		if !found {
			r.Summary.CriticalUnits = append(r.Summary.CriticalUnits, unitID)
		}
	}
}
