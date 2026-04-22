package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/loov/dreamlint/extract"
)

// TestGolden_GoExample pins the extraction of the committed
// testdata/go-example/index.scip fixture. The fixture is regenerated via
// testdata/go-example/generate.sh (Docker + scip-go); regenerate and
// update this test when the Go sources change.
func TestGolden_GoExample(t *testing.T) {
	root, err := filepath.Abs("testdata/go-example")
	if err != nil {
		t.Fatal(err)
	}

	ex := &Extractor{
		IndexPath:   filepath.Join(root, "index.scip"),
		ProjectRoot: root,
	}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if res.Language != "Go" {
		t.Errorf("Language = %q, want Go", res.Language)
	}

	sccID := "example/example.IsEven+example/example.IsOdd"
	wantUnits := map[string]bool{
		"example/example.Add":               true,
		"example/example.Multiply":          true,
		"example/example.(Counter).Bump":    true,
		"example/example.(Counter).Value":   true,
		sccID:                               true,
	}
	gotUnits := map[string]bool{}
	for _, u := range res.Units {
		gotUnits[u.ID] = true
	}
	for id := range wantUnits {
		if !gotUnits[id] {
			t.Errorf("missing unit %q", id)
		}
	}
	for id := range gotUnits {
		if !wantUnits[id] {
			t.Errorf("unexpected unit %q", id)
		}
	}

	byID := map[string]*extract.AnalysisUnit{}
	for _, u := range res.Units {
		byID[u.ID] = u
	}

	// Free function: body, doc, signature.
	add := byID["example/example.Add"]
	if len(add.Callees) != 0 {
		t.Errorf("Add.Callees = %v, want empty", add.Callees)
	}
	assertBodyContains(t, "Add", add.Functions[0].Body,
		"func Add(a, b int) int",
		"return a + b")

	// Method: receiver set correctly.
	bump := byID["example/example.(Counter).Bump"]
	if got := bump.Functions[0].Receiver; got != "Counter" {
		t.Errorf("Bump.Receiver = %q, want Counter", got)
	}
	if !slices.Contains(bump.Callees, "example/example.Add") {
		t.Errorf("Bump.Callees = %v, want to contain example/example.Add", bump.Callees)
	}
	assertBodyContains(t, "Bump", bump.Functions[0].Body,
		"func (c *Counter) Bump() int",
		"Add(c.n, 1)")

	// Value method: pointer vs value receiver.
	val := byID["example/example.(Counter).Value"]
	if got := val.Functions[0].Receiver; got != "Counter" {
		t.Errorf("Value.Receiver = %q, want Counter", got)
	}

	// Multiply calls Add.
	mul := byID["example/example.Multiply"]
	if !slices.Contains(mul.Callees, "example/example.Add") {
		t.Errorf("Multiply.Callees = %v, want to contain example/example.Add", mul.Callees)
	}

	// Mutual recursion SCC.
	scc := byID[sccID]
	if scc == nil {
		t.Fatalf("SCC unit %q not found", sccID)
	}
	if len(scc.Functions) != 2 {
		t.Fatalf("SCC unit has %d functions, want 2", len(scc.Functions))
	}
	names := map[string]bool{}
	for _, f := range scc.Functions {
		names[f.Name] = true
	}
	if !names["IsEven"] || !names["IsOdd"] {
		t.Errorf("SCC functions = %v, want {IsEven, IsOdd}", names)
	}

	// Type: Counter struct with methods linked.
	ti, ok := res.Types["example/example.Counter"]
	if !ok {
		t.Fatalf("Counter type missing; got keys: %v", typeKeys(res.Types))
	}
	if ti.Kind != "struct" {
		t.Errorf("Counter.Kind = %q, want struct", ti.Kind)
	}
	if !strings.Contains(ti.Body, "Counter") {
		t.Errorf("Counter body missing type name; body=%q", ti.Body)
	}
	if len(ti.Methods) != 2 {
		t.Errorf("Counter.Methods = %v, want 2 entries", ti.Methods)
	}
}

func typeKeys(m map[string]*extract.TypeInfo) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
