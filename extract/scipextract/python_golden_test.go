package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestGolden_PythonExample pins the extraction of the committed
// testdata/python-example/index.scip fixture. The fixture is
// regenerated via testdata/python-example/generate.sh (Docker +
// scip-python); regenerate and update this test when the Python
// sources change.
func TestGolden_PythonExample(t *testing.T) {
	root, err := filepath.Abs("testdata/python-example")
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

	if res.Language != "Python" {
		t.Errorf("Language = %q, want Python", res.Language)
	}

	pkg := "example/math_helpers"
	sccID := pkg + ".is_even+" + pkg + ".is_odd"

	wantUnits := map[string]bool{
		pkg + ".add":                true,
		pkg + ".multiply":           true,
		pkg + ".(Counter).__init__": true,
		pkg + ".(Counter).bump":     true,
		pkg + ".(Counter).value":    true,
		sccID:                       true,
		"example/main.main":         true,
	}
	gotUnits := map[string]bool{}
	for _, u := range res.Units {
		gotUnits[u.ID] = true
	}
	for id := range wantUnits {
		if !gotUnits[id] {
			keys := make([]string, 0, len(gotUnits))
			for k := range gotUnits {
				keys = append(keys, k)
			}
			t.Errorf("missing unit %q; got %v", id, keys)
		}
	}
	for id := range gotUnits {
		if !wantUnits[id] {
			t.Errorf("unexpected unit %q", id)
		}
	}

	byID := map[string]struct {
		callees []string
		funcs   []struct{ name, recv, body string }
	}{}
	for _, u := range res.Units {
		e := struct {
			callees []string
			funcs   []struct{ name, recv, body string }
		}{callees: u.Callees}
		for _, fn := range u.Functions {
			e.funcs = append(e.funcs, struct{ name, recv, body string }{fn.Name, fn.Receiver, fn.Body})
		}
		byID[u.ID] = e
	}

	// Free function body.
	add := byID[pkg+".add"]
	if !strings.Contains(add.funcs[0].body, "def add") {
		t.Errorf("add body missing signature; body=%q", add.funcs[0].body)
	}

	// Method receiver.
	bump := byID[pkg+".(Counter).bump"]
	if bump.funcs[0].recv != "Counter" {
		t.Errorf("bump.Receiver = %q, want Counter", bump.funcs[0].recv)
	}
	if !slices.Contains(bump.callees, pkg+".add") {
		t.Errorf("bump.Callees = %v, want to contain add", bump.callees)
	}

	// __init__ (constructor).
	init := byID[pkg+".(Counter).__init__"]
	if init.funcs[0].recv != "Counter" {
		t.Errorf("__init__.Receiver = %q, want Counter", init.funcs[0].recv)
	}

	// Mutual recursion SCC.
	scc := byID[sccID]
	if len(scc.funcs) != 2 {
		t.Fatalf("SCC has %d functions, want 2", len(scc.funcs))
	}
	names := map[string]bool{}
	for _, f := range scc.funcs {
		names[f.name] = true
	}
	if !names["is_even"] || !names["is_odd"] {
		t.Errorf("SCC functions = %v, want {is_even, is_odd}", names)
	}

	// main in a separate module calls across files.
	main := byID["example/main.main"]
	for _, want := range []string{
		pkg + ".add",
		pkg + ".multiply",
		pkg + ".(Counter).bump",
		pkg + ".(Counter).value",
		sccID,
	} {
		if !slices.Contains(main.callees, want) {
			t.Errorf("main.Callees missing %q; got %v", want, main.callees)
		}
	}

	// Counter type.
	ti, ok := res.Types[pkg+".Counter"]
	if !ok {
		t.Fatalf("Counter type missing; types=%v", typeKeys(res.Types))
	}
	if len(ti.Methods) != 3 {
		t.Errorf("Counter.Methods = %v, want 3 entries", ti.Methods)
	}

	// External: builtins.print.
	sawPrint := false
	for _, e := range res.External {
		if e.Name == "print" {
			sawPrint = true
		}
	}
	if !sawPrint {
		t.Errorf("expected external entry for builtins.print, got %d externals", len(res.External))
	}
}
