package extract

import "context"

// Result is the output of an Extractor: the analysis units (callgraph-ordered),
// a map of external function metadata, the user-defined types, and the
// source language.
type Result struct {
	Units    []*AnalysisUnit
	External map[string]*ExternalFunc
	Types    map[string]*TypeInfo
	Language string
	Warnings []string
}

// Extractor produces analysis units from a source tree or index.
// Implementations may parse source directly (goextract) or consume
// a pre-built index such as SCIP (scipextract).
type Extractor interface {
	Extract(ctx context.Context) (*Result, error)
}
