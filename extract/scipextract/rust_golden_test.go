package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/loov/dreamlint/extract"
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

	sccID := "rust_example.is_even+rust_example.is_odd"
	wantUnits := []string{
		"rust_example.add",
		"rust_example.(Counter).bump",
		"rust_example.(Counter).new",
		"rust_example.(Counter).default",
		"rust_example.(Counter).value",
		sccID,
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

	byID := map[string]*extract.AnalysisUnit{}
	for _, u := range res.Units {
		byID[u.ID] = u
	}

	// Free function: body, signature, doc comment all preserved.
	add := byID["rust_example.add"]
	if len(add.Callees) != 0 {
		t.Errorf("add.Callees = %v, want empty", add.Callees)
	}
	assertBodyContains(t, "add", add.Functions[0].Body,
		"pub fn add(a: i32, b: i32) -> i32",
		"a + b")
	if sig := add.Functions[0].Signature; !strings.Contains(sig, "pub fn add") {
		t.Errorf("add.Signature = %q, want to contain 'pub fn add'", sig)
	}

	// Struct method: Receiver set to the impl target, not "impl".
	bump := byID["rust_example.(Counter).bump"]
	if got := bump.Functions[0].Receiver; got != "Counter" {
		t.Errorf("bump.Receiver = %q, want Counter", got)
	}
	if !slices.Contains(bump.Callees, "rust_example.add") {
		t.Errorf("bump.Callees = %v, want to contain rust_example.add", bump.Callees)
	}
	assertBodyContains(t, "bump", bump.Functions[0].Body,
		"pub fn bump(&mut self) -> i32",
		"self.n = add(self.n, 1);")

	// Default trait impl for Counter still gets Receiver "Counter" (the
	// impl target), with the trait name available on the descriptor chain
	// but not bleeding into Receiver.
	def := byID["rust_example.(Counter).default"]
	if got := def.Functions[0].Receiver; got != "Counter" {
		t.Errorf("default.Receiver = %q, want Counter", got)
	}
	if !slices.Contains(def.Callees, "rust_example.(Counter).new") {
		t.Errorf("default.Callees = %v, want to contain rust_example.(Counter).new", def.Callees)
	}

	// Mutual recursion: is_even + is_odd land in a single SCC unit.
	scc := byID[sccID]
	if len(scc.Functions) != 2 {
		t.Fatalf("SCC unit has %d functions, want 2", len(scc.Functions))
	}
	names := map[string]bool{}
	for _, f := range scc.Functions {
		names[f.Name] = true
	}
	if !names["is_even"] || !names["is_odd"] {
		t.Errorf("SCC functions = %v, want {is_even, is_odd}", names)
	}
	if len(scc.Callees) != 0 {
		t.Errorf("SCC.Callees = %v, want empty (mutual recursion is internal)", scc.Callees)
	}

	// main pulls the SCC, both helpers, and the Counter methods together.
	main := byID["rust_example.main"]
	for _, want := range []string{
		"rust_example.add",
		"rust_example.multiply",
		"rust_example.(Counter).new",
		"rust_example.(Counter).bump",
		"rust_example.(Counter).value",
		sccID,
	} {
		if !slices.Contains(main.Callees, want) {
			t.Errorf("main.Callees missing %q; got %v", want, main.Callees)
		}
	}

	// println! still shows up as external.
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
