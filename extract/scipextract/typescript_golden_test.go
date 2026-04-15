package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
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

	// scip-typescript leaves Document.Language empty; we recover it from
	// the symbol scheme.
	if res.Language != "TypeScript" {
		t.Errorf("Language = %q, want TypeScript", res.Language)
	}

	wantUnits := []string{
		"typescript-example/src/math.ts.add",
		"typescript-example/src/math.ts.multiply",
		"typescript-example/src/main.ts.main",
	}
	if len(res.Units) != len(wantUnits) {
		var got []string
		for _, u := range res.Units {
			got = append(got, u.ID)
		}
		t.Fatalf("units = %v, want %v", got, wantUnits)
	}
	for i, want := range wantUnits {
		if got := res.Units[i].ID; got != want {
			t.Errorf("units[%d].ID = %q, want %q", i, got, want)
		}
	}

	byID := map[string]*unitSummary{}
	for _, u := range res.Units {
		byID[u.ID] = &unitSummary{
			callees: u.Callees,
			body:    u.Functions[0].Body,
			godoc:   u.Functions[0].Godoc,
			sig:     u.Functions[0].Signature,
		}
	}

	addID := "typescript-example/src/math.ts.add"
	mulID := "typescript-example/src/math.ts.multiply"
	mainID := "typescript-example/src/main.ts.main"

	// add: leaf, body extracted via scip-typescript's EnclosingRange.
	if got := byID[addID].callees; len(got) != 0 {
		t.Errorf("add.Callees = %v, want empty", got)
	}
	assertBodyContains(t, "add", byID[addID].body,
		"export function add(a: number, b: number): number",
		"return a + b;")
	if godoc := byID[addID].godoc; !strings.Contains(godoc, "Adds two numbers") {
		t.Errorf("add.Godoc = %q, want to mention 'Adds two numbers'", godoc)
	}

	// multiply calls add.
	if !slices.Contains(byID[mulID].callees, addID) {
		t.Errorf("multiply.Callees = %v, want to contain %s", byID[mulID].callees, addID)
	}
	assertBodyContains(t, "multiply", byID[mulID].body,
		"export function multiply(a: number, b: number): number",
		"result = add(result, a);")

	// main calls both add and multiply.
	mainCallees := byID[mainID].callees
	if !slices.Contains(mainCallees, addID) {
		t.Errorf("main.Callees = %v, want to contain %s", mainCallees, addID)
	}
	if !slices.Contains(mainCallees, mulID) {
		t.Errorf("main.Callees = %v, want to contain %s", mainCallees, mulID)
	}
	assertBodyContains(t, "main", byID[mainID].body,
		"function main(): void",
		"const sum = add(2, 3);",
		"console.log(product);")

	// console.log comes from lib.dom.d.ts and should appear as external.
	sawConsoleLog := false
	for _, e := range res.External {
		if e.Name == "log" {
			sawConsoleLog = true
			break
		}
	}
	if !sawConsoleLog {
		t.Errorf("expected an external entry for console.log, got %d externals", len(res.External))
	}
}
