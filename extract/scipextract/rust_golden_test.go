package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestGolden_RustExample pins the extraction of the committed
// testdata/rust-example/index.scip fixture. The fixture is regenerated via
// testdata/rust-example/generate.sh (Docker + rust-analyzer scip);
// regenerate and update this test when the Rust sources change.
func TestGolden_RustExample(t *testing.T) {
	root, err := filepath.Abs("testdata/rust-example")
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

	if res.Language != "Rust" {
		t.Errorf("Language = %q, want Rust", res.Language)
	}

	wantUnits := []string{
		"rust_example.add",
		"rust_example.multiply",
		"rust_example.main",
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

	// add: leaf, body extracted via rust-analyzer's EnclosingRange, doc comment preserved.
	if got := byID["rust_example.add"].callees; len(got) != 0 {
		t.Errorf("add.Callees = %v, want empty", got)
	}
	assertBodyContains(t, "add", byID["rust_example.add"].body,
		"pub fn add(a: i32, b: i32) -> i32",
		"a + b")
	if sig := byID["rust_example.add"].sig; !strings.Contains(sig, "pub fn add") {
		t.Errorf("add.Signature = %q, want to contain 'pub fn add'", sig)
	}
	if godoc := byID["rust_example.add"].godoc; !strings.Contains(godoc, "Adds two numbers") {
		t.Errorf("add.Godoc = %q, want to mention 'Adds two numbers'", godoc)
	}

	// multiply: calls add.
	if !slices.Contains(byID["rust_example.multiply"].callees, "rust_example.add") {
		t.Errorf("multiply.Callees = %v, want to contain rust_example.add",
			byID["rust_example.multiply"].callees)
	}
	assertBodyContains(t, "multiply", byID["rust_example.multiply"].body,
		"pub fn multiply(a: i32, b: i32) -> i32",
		"result = add(result, a);")

	// main: calls add and multiply.
	mainCallees := byID["rust_example.main"].callees
	if !slices.Contains(mainCallees, "rust_example.add") {
		t.Errorf("main.Callees = %v, want to contain rust_example.add", mainCallees)
	}
	if !slices.Contains(mainCallees, "rust_example.multiply") {
		t.Errorf("main.Callees = %v, want to contain rust_example.multiply", mainCallees)
	}
	assertBodyContains(t, "main", byID["rust_example.main"].body,
		"fn main()",
		"let sum = add(2, 3);",
		`println!("{}", product);`)

	// println! macro and the core i32::add operator implementation should
	// land in the externals via rust-analyzer's cross-crate references.
	sawPrintln := false
	for _, e := range res.External {
		if e.Name == "println" {
			sawPrintln = true
			break
		}
	}
	if !sawPrintln {
		t.Errorf("expected an external entry for println!, got %d externals", len(res.External))
	}
}
