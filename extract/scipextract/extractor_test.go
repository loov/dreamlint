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

// TestExtract_MethodReceiverTypeLink checks that the extractor
// collects type symbols, extracts their declaration body, and wires
// FunctionInfo.ReceiverType / TypeInfo.Methods together.
func TestExtract_MethodReceiverTypeLink(t *testing.T) {
	dir := t.TempDir()
	rel := "src/lib.rs"
	abs := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	// Lines (0-indexed):
	//   0: pub struct Counter {
	//   1:     n: i32,
	//   2: }
	//   3:
	//   4: impl Counter {
	//   5:     pub fn bump(&mut self) -> i32 {
	//   6:         self.n + 1
	//   7:     }
	//   8: }
	source := "pub struct Counter {\n    n: i32,\n}\n\nimpl Counter {\n    pub fn bump(&mut self) -> i32 {\n        self.n + 1\n    }\n}\n"
	if err := os.WriteFile(abs, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}

	counterSym := "rust-analyzer cargo example 0.1.0 Counter#"
	bumpSym := "rust-analyzer cargo example 0.1.0 Counter#bump()."

	index := &scip.Index{
		Metadata: &scip.Metadata{ProjectRoot: "file://" + dir},
		Documents: []*scip.Document{{
			Language:     "Rust",
			RelativePath: rel,
			Symbols: []*scip.SymbolInformation{
				{
					Symbol:      counterSym,
					Kind:        scip.SymbolInformation_Struct,
					DisplayName: "Counter",
				},
				{
					Symbol:      bumpSym,
					Kind:        scip.SymbolInformation_Method,
					DisplayName: "bump",
				},
			},
			Occurrences: []*scip.Occurrence{
				{
					Symbol:         counterSym,
					Range:          []int32{0, 11, 18},
					EnclosingRange: []int32{0, 0, 2, 1},
					SymbolRoles:    int32(scip.SymbolRole_Definition),
				},
				{
					Symbol:         bumpSym,
					Range:          []int32{5, 11, 5, 15},
					EnclosingRange: []int32{5, 4, 7, 5},
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

	if len(res.Types) != 1 {
		t.Fatalf("got %d types, want 1: %+v", len(res.Types), res.Types)
	}
	ti, ok := res.Types["example.Counter"]
	if !ok {
		gotKeys := make([]string, 0, len(res.Types))
		for k := range res.Types {
			gotKeys = append(gotKeys, k)
		}
		t.Fatalf("Counter type missing; got keys=%v", gotKeys)
	}
	if ti.Kind != "struct" {
		t.Errorf("Kind = %q, want struct", ti.Kind)
	}
	if ti.Body == "" {
		t.Errorf("Counter body is empty")
	}
	if len(ti.Methods) != 1 || ti.Methods[0] != "example.(Counter).bump" {
		t.Errorf("Counter.Methods = %v, want [example.(Counter).bump]", ti.Methods)
	}

	var bump *extractFunctionInfoView
	for _, unit := range res.Units {
		for _, fn := range unit.Functions {
			if fn.Name == "bump" {
				bump = &extractFunctionInfoView{ReceiverType: fn.ReceiverType, Receiver: fn.Receiver}
			}
		}
	}
	if bump == nil {
		t.Fatal("bump method not found")
	}
	if bump.Receiver != "Counter" {
		t.Errorf("Receiver = %q, want Counter", bump.Receiver)
	}
	if bump.ReceiverType != "example.Counter" {
		t.Errorf("ReceiverType = %q, want example.Counter", bump.ReceiverType)
	}
}

type extractFunctionInfoView struct {
	Receiver     string
	ReceiverType string
}

func TestStripFileURI(t *testing.T) {
	cases := map[string]string{
		"":                                "",
		"/plain/path":                     "/plain/path",
		"file:///home/user/project":      "/home/user/project",
		"file:///C:/src/project":         "C:/src/project",
		"file:///c:/src/project":         "c:/src/project",
		"file://C:/src/project":          "C:/src/project",
		"file:///Z:/with/space dir":      "Z:/with/space dir",
		"file://host/share/path":         "host/share/path",
	}
	for in, want := range cases {
		if got := stripFileURI(in); got != want {
			t.Errorf("stripFileURI(%q) = %q, want %q", in, got, want)
		}
	}
}
