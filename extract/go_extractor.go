package extract

import (
	"context"
	"fmt"
)

// GoExtractor extracts analysis units from Go source via go/packages and SSA+CHA.
type GoExtractor struct {
	Dir      string
	Patterns []string
}

// Extract loads Go packages and builds the analysis units.
func (g *GoExtractor) Extract(ctx context.Context) (*Result, error) {
	pkgs, err := LoadPackages(g.Dir, g.Patterns...)
	if err != nil {
		return nil, fmt.Errorf("load packages: %w", err)
	}

	funcs := ExtractFunctions(pkgs)
	graph := BuildCallgraph(pkgs)
	external := ExtractExternalFuncs(pkgs, graph)
	units := BuildAnalysisUnits(funcs, graph)

	return &Result{
		Units:    units,
		External: external,
		Language: "Go",
	}, nil
}
