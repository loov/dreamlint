//go:build integration

package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/loov/reviewmod/extract"
)

func TestIntegration_FullPipeline(t *testing.T) {
	dir := t.TempDir()

	// Create test Go module
	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

// Add adds two numbers.
func Add(a, b int) int {
	return a + b
}

// Multiply multiplies by calling Add repeatedly.
func Multiply(a, b int) int {
	result := 0
	for i := 0; i < b; i++ {
		result = Add(result, a)
	}
	return result
}
`
	if err := os.WriteFile(filepath.Join(dir, "math.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	// Load packages once
	pkgs, err := extract.LoadPackages(dir, "./...")
	if err != nil {
		t.Fatalf("LoadPackages: %v", err)
	}

	// Extract functions
	funcs := extract.ExtractFunctions(pkgs)

	if len(funcs) != 2 {
		t.Fatalf("got %d functions, want 2", len(funcs))
	}

	// Build callgraph
	graph := extract.BuildCallgraph(pkgs)

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
