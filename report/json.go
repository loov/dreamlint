// report/json.go
package report

import (
	"encoding/json"
	"os"
)

// WriteJSON serializes the report to JSON
func WriteJSON(r *Report) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// WriteJSONFile writes the report to a JSON file
func WriteJSONFile(r *Report, path string) error {
	data, err := WriteJSON(r)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// ReadJSONFile reads a report from a JSON file
func ReadJSONFile(path string) (*Report, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Report
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	// Ensure maps are initialized
	if r.Units == nil {
		r.Units = make(map[string]UnitReport)
	}
	if r.Summary.BySeverity == nil {
		r.Summary.BySeverity = make(map[string]int)
	}
	if r.Summary.ByCategory == nil {
		r.Summary.ByCategory = make(map[string]int)
	}
	return &r, nil
}
