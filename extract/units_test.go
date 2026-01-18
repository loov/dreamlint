// extract/units_test.go
package extract

import (
	"testing"
)

func TestBuildAnalysisUnits(t *testing.T) {
	funcs := []*FunctionInfo{
		{Package: "pkg", Name: "A"},
		{Package: "pkg", Name: "B"},
		{Package: "pkg", Name: "C"},
	}

	graph := map[string][]string{
		"pkg.A": {"pkg.B"},
		"pkg.B": {"pkg.C"},
		"pkg.C": {},
	}

	units := BuildAnalysisUnits(funcs, graph)

	// Should have 3 units in topological order: C, B, A
	if len(units) != 3 {
		t.Fatalf("got %d units, want 3", len(units))
	}

	// First unit should be C (no dependencies)
	if units[0].ID != "pkg.C" {
		t.Errorf("first unit should be C, got %s", units[0].ID)
	}

	// Last unit should be A
	if units[2].ID != "pkg.A" {
		t.Errorf("last unit should be A, got %s", units[2].ID)
	}

	// A should have B as callee
	if len(units[2].Callees) != 1 || units[2].Callees[0] != "pkg.B" {
		t.Errorf("A callees should be [pkg.B], got %v", units[2].Callees)
	}
}

func TestBuildAnalysisUnits_SCC(t *testing.T) {
	funcs := []*FunctionInfo{
		{Package: "pkg", Name: "A"},
		{Package: "pkg", Name: "B"},
		{Package: "pkg", Name: "C"},
	}

	// B and C form a cycle
	graph := map[string][]string{
		"pkg.A": {"pkg.B"},
		"pkg.B": {"pkg.C"},
		"pkg.C": {"pkg.B"},
	}

	units := BuildAnalysisUnits(funcs, graph)

	// Should have 2 units: {B,C} SCC, then A
	if len(units) != 2 {
		t.Fatalf("got %d units, want 2", len(units))
	}

	// First unit should be the SCC with B and C
	if len(units[0].Functions) != 2 {
		t.Errorf("first unit should have 2 functions, got %d", len(units[0].Functions))
	}
}
