package goextract

import (
	"context"
	"fmt"

	"github.com/loov/dreamlint/extract"
)

// Extractor extracts analysis units from Go source via go/packages and SSA+CHA.
type Extractor struct {
	Dir      string
	Patterns []string
}

// Extract loads Go packages and builds the analysis units.
func (e *Extractor) Extract(ctx context.Context) (*extract.Result, error) {
	pkgs, err := LoadPackages(e.Dir, e.Patterns...)
	if err != nil {
		return nil, fmt.Errorf("load packages: %w", err)
	}

	funcs := ExtractFunctions(pkgs)
	graph := BuildCallgraph(pkgs)
	external := ExtractExternalFuncs(pkgs, graph)
	units := extract.BuildAnalysisUnits(funcs, graph)

	return &extract.Result{
		Units:    units,
		External: external,
		Language: "Go",
	}, nil
}
