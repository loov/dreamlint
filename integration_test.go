//go:build integration

package main

import (
	"slices"
	"testing"

	"github.com/loov/dreamlint/extract"
	"github.com/loov/dreamlint/extract/goextract"
)

func TestIntegration_FullPipeline(t *testing.T) {
	// Load packages from testdata
	pkgs, err := goextract.LoadPackages("testdata/testpkg", "./...")
	if err != nil {
		t.Fatalf("LoadPackages: %v", err)
	}

	// Extract functions
	funcs := goextract.ExtractFunctions(pkgs)

	if len(funcs) != 2 {
		t.Fatalf("got %d functions, want 2", len(funcs))
	}

	// Build callgraph
	graph := goextract.BuildCallgraph(pkgs)

	// Multiply should call Add
	if !slices.Contains(graph["testpkg.Multiply"], "testpkg.Add") {
		t.Errorf("Multiply should call Add")
	}

	// Build analysis units
	units := extract.BuildAnalysisUnits(funcs, graph)

	if len(units) != 2 {
		t.Fatalf("got %d units, want 2", len(units))
	}

	// First unit should be Add (no dependencies)
	if units[0].Functions[0].Name != "Add" {
		t.Errorf("first unit should be Add, got %s", units[0].Functions[0].Name)
	}

	// Second unit should be Multiply
	if units[1].Functions[0].Name != "Multiply" {
		t.Errorf("second unit should be Multiply, got %s", units[1].Functions[0].Name)
	}
}
