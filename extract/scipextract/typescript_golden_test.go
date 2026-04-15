package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/loov/dreamlint/extract"
)

// TestGolden_TypeScriptExample pins the extraction of the committed
// testdata/typescript-example/index.scip fixture. The fixture is
// regenerated via testdata/typescript-example/generate.sh
// (Docker + @sourcegraph/scip-typescript); regenerate and update this test
// when the TypeScript sources change.
func TestGolden_TypeScriptExample(t *testing.T) {
	root, err := filepath.Abs("testdata/typescript-example")
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

	if res.Language != "TypeScript" {
		t.Errorf("Language = %q, want TypeScript", res.Language)
	}

	const (
		addID  = "typescript-example/src/math.ts.add"
		mulID  = "typescript-example/src/math.ts.multiply"
		ctorID = "typescript-example/src/math.ts.(Counter).<constructor>"
		bumpID = "typescript-example/src/math.ts.(Counter).bump"
		valID  = "typescript-example/src/math.ts.(Counter).value"
		sccID  = "typescript-example/src/math.ts.isEven+typescript-example/src/math.ts.isOdd"
		mainID = "typescript-example/src/main.ts.main"
	)
	want := map[string]bool{
		addID:  true,
		mulID:  true,
		ctorID: true,
		bumpID: true,
		valID:  true,
		sccID:  true,
		mainID: true,
	}
	got := map[string]bool{}
	byID := map[string]*extract.AnalysisUnit{}
	for _, u := range res.Units {
		got[u.ID] = true
		byID[u.ID] = u
	}
	for id := range want {
		if !got[id] {
			t.Errorf("missing unit %q; got %v", id, keys(got))
		}
	}
	for id := range got {
		if !want[id] {
			t.Errorf("unexpected unit %q", id)
		}
	}

	// Topology: callees precede callers.
	indexOf := func(id string) int {
		for i, u := range res.Units {
			if u.ID == id {
				return i
			}
		}
		return -1
	}
	for _, pair := range [][2]string{
		{addID, mulID},
		{addID, bumpID},
		{ctorID, mainID},
		{sccID, mainID},
	} {
		if indexOf(pair[0]) >= indexOf(pair[1]) {
			t.Errorf("topology: %q should precede %q", pair[0], pair[1])
		}
	}

	// Class methods: Counter descriptor becomes the Receiver.
	for _, id := range []string{ctorID, bumpID, valID} {
		if r := byID[id].Functions[0].Receiver; r != "Counter" {
			t.Errorf("%s receiver = %q, want Counter", id, r)
		}
	}

	// Mutual recursion SCC.
	scc := byID[sccID]
	if len(scc.Functions) != 2 {
		t.Fatalf("SCC unit has %d functions, want 2", len(scc.Functions))
	}
	names := map[string]bool{}
	for _, f := range scc.Functions {
		names[f.Name] = true
	}
	if !names["isEven"] || !names["isOdd"] {
		t.Errorf("SCC functions = %v, want {isEven, isOdd}", names)
	}
	if len(scc.Callees) != 0 {
		t.Errorf("SCC.Callees = %v, want empty (mutual recursion is internal)", scc.Callees)
	}

	// Body assertions for a couple of representative units.
	assertBodyContains(t, "add", byID[addID].Functions[0].Body,
		"export function add(a: number, b: number): number",
		"return a + b;")
	assertBodyContains(t, "bump", byID[bumpID].Functions[0].Body,
		"bump(): number",
		"this.n = add(this.n, 1);")
	for _, f := range scc.Functions {
		if !strings.Contains(f.Body, "return "+flipTS(f.Name)+"(n - 1);") {
			t.Errorf("%s body missing mutual-recursion call; got %q", f.Name, f.Body)
		}
	}

	// main callgraph pulls everything together.
	mainCallees := byID[mainID].Callees
	for _, w := range []string{addID, mulID, ctorID, bumpID, valID, sccID} {
		if !slices.Contains(mainCallees, w) {
			t.Errorf("main.Callees missing %q; got %v", w, mainCallees)
		}
	}

	// console.log lands in externals.
	sawConsoleLog := false
	for _, e := range res.External {
		if e.Name == "log" {
			sawConsoleLog = true
			break
		}
	}
	if !sawConsoleLog {
		t.Errorf("expected external entry for console.log, got %d externals", len(res.External))
	}
}

func flipTS(name string) string {
	switch name {
	case "isEven":
		return "isOdd"
	case "isOdd":
		return "isEven"
	}
	return name
}
