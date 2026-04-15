package scipextract

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/scip-code/scip/bindings/go/scip"
	"google.golang.org/protobuf/proto"
)

func TestExtract_SingleDocument(t *testing.T) {
	dir := t.TempDir()
	rel := "src/lib.rs"
	abs := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	source := "fn foo() -> u32 {\n    1 + 2\n}\n\nfn bar() -> u32 {\n    foo()\n}\n"
	if err := os.WriteFile(abs, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}

	index := &scip.Index{
		Metadata: &scip.Metadata{
			ProjectRoot: "file://" + dir,
		},
		Documents: []*scip.Document{{
			Language:     "Rust",
			RelativePath: rel,
			Symbols: []*scip.SymbolInformation{
				{
					Symbol:      "rust-analyzer cargo example 0.1.0 foo().",
					Kind:        scip.SymbolInformation_Function,
					DisplayName: "foo",
				},
				{
					Symbol:      "rust-analyzer cargo example 0.1.0 bar().",
					Kind:        scip.SymbolInformation_Function,
					DisplayName: "bar",
				},
			},
			Occurrences: []*scip.Occurrence{
				{
					Symbol:         "rust-analyzer cargo example 0.1.0 foo().",
					Range:          []int32{0, 3, 0, 6},
					EnclosingRange: []int32{0, 0, 2, 1},
					SymbolRoles:    int32(scip.SymbolRole_Definition),
				},
				{
					Symbol:         "rust-analyzer cargo example 0.1.0 bar().",
					Range:          []int32{4, 3, 4, 6},
					EnclosingRange: []int32{4, 0, 6, 1},
					SymbolRoles:    int32(scip.SymbolRole_Definition),
				},
			},
		}},
	}

	data, err := proto.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	idxPath := filepath.Join(dir, "index.scip")
	if err := os.WriteFile(idxPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	ex := &Extractor{IndexPath: idxPath}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if res.Language != "Rust" {
		t.Errorf("Language = %q, want Rust", res.Language)
	}
	if len(res.Units) != 2 {
		t.Fatalf("got %d units, want 2", len(res.Units))
	}

	names := map[string]bool{}
	for _, unit := range res.Units {
		for _, f := range unit.Functions {
			names[f.Name] = true
			if f.Body == "" {
				t.Errorf("function %s has empty body", f.Name)
			}
		}
		if len(unit.Callees) != 0 {
			t.Errorf("step 4 should emit empty callees, got %v", unit.Callees)
		}
	}
	if !names["foo"] || !names["bar"] {
		t.Errorf("missing expected functions: got %v", names)
	}
}
