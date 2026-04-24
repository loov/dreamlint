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

func TestFilterDocuments(t *testing.T) {
	docs := []*scip.Document{
		{RelativePath: "src/main.rs"},
		{RelativePath: "src/lib.rs"},
		{RelativePath: "tests/integration.rs"},
		{RelativePath: "build.rs"},
	}

	t.Run("empty filters keeps all", func(t *testing.T) {
		got := filterDocuments(docs, nil)
		if len(got) != len(docs) {
			t.Errorf("got %d docs, want %d", len(got), len(docs))
		}
	})

	t.Run("single glob", func(t *testing.T) {
		got := filterDocuments(docs, []string{"src/*.rs"})
		if len(got) != 2 {
			t.Errorf("got %d docs, want 2", len(got))
		}
	})

	t.Run("multiple globs", func(t *testing.T) {
		got := filterDocuments(docs, []string{"src/*.rs", "build.rs"})
		if len(got) != 3 {
			t.Errorf("got %d docs, want 3", len(got))
		}
	})

	t.Run("no match", func(t *testing.T) {
		got := filterDocuments(docs, []string{"*.go"})
		if len(got) != 0 {
			t.Errorf("got %d docs, want 0", len(got))
		}
	})
}

func TestExtract_ProjectRootOverride(t *testing.T) {
	dir := t.TempDir()
	rel := "src/lib.rs"
	abs := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	source := "fn hello() -> u32 {\n    42\n}\n"
	if err := os.WriteFile(abs, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}

	index := &scip.Index{
		Metadata: &scip.Metadata{
			ProjectRoot: "file:///nonexistent/bogus/path",
		},
		Documents: []*scip.Document{{
			Language:     "Rust",
			RelativePath: rel,
			Symbols: []*scip.SymbolInformation{{
				Symbol:      "rust-analyzer cargo example 0.1.0 hello().",
				Kind:        scip.SymbolInformation_Function,
				DisplayName: "hello",
			}},
			Occurrences: []*scip.Occurrence{{
				Symbol:         "rust-analyzer cargo example 0.1.0 hello().",
				Range:          []int32{0, 3, 0, 8},
				EnclosingRange: []int32{0, 0, 2, 1},
				SymbolRoles:    int32(scip.SymbolRole_Definition),
			}},
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

	ex := &Extractor{IndexPath: idxPath, ProjectRoot: dir}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if len(res.Units) != 1 {
		t.Fatalf("got %d units, want 1", len(res.Units))
	}
	fn := res.Units[0].Functions[0]
	if fn.Body == "" {
		t.Error("body empty — ProjectRoot override did not resolve source file")
	}
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

func TestDisplayLanguage(t *testing.T) {
	cases := map[string]string{
		"CPP":           "C++",
		"cpp":           "C++",
		"Cpp":           "C++",
		"CSharp":        "C#",
		"csharp":        "C#",
		"ObjectiveC":    "Objective-C",
		"ObjectiveCPP":  "Objective-C++",
		"JavaScript":    "JavaScript",
		"typescript":    "TypeScript",
		"Go":            "Go",
		"go":            "Go",
		"Rust":          "Rust",
		"rust":          "Rust",
		"Java":          "Java",
		"Kotlin":        "Kotlin",
		"Python":        "Python",
		"Ruby":          "Ruby",
		"c":             "C",
		"C":             "C",
		"UnknownLang":   "UnknownLang",
	}
	for in, want := range cases {
		if got := displayLanguage(in); got != want {
			t.Errorf("displayLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPickLanguage(t *testing.T) {
	t.Run("majority wins", func(t *testing.T) {
		docs := []*scip.Document{
			{Language: "Rust"},
			{Language: "Rust"},
			{Language: "CPP"},
		}
		if got := pickLanguage(docs); got != "Rust" {
			t.Errorf("got %q, want Rust", got)
		}
	})

	t.Run("tie broken alphabetically", func(t *testing.T) {
		docs := []*scip.Document{
			{Language: "Rust"},
			{Language: "Go"},
		}
		if got := pickLanguage(docs); got != "Go" {
			t.Errorf("got %q, want Go (alphabetically first)", got)
		}
	})

	t.Run("empty language falls back to scheme", func(t *testing.T) {
		docs := []*scip.Document{
			{
				Language: "",
				Symbols: []*scip.SymbolInformation{
					{Symbol: "scip-typescript npm pkg 1.0.0 foo()."},
				},
			},
		}
		if got := pickLanguage(docs); got != "TypeScript" {
			t.Errorf("got %q, want TypeScript", got)
		}
	})

	t.Run("no docs returns empty", func(t *testing.T) {
		if got := pickLanguage(nil); got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("normalizes display name", func(t *testing.T) {
		docs := []*scip.Document{{Language: "CPP"}}
		if got := pickLanguage(docs); got != "C++" {
			t.Errorf("got %q, want C++", got)
		}
	})
}

func writeIndex(t *testing.T, dir string, index *scip.Index) string {
	t.Helper()
	data, err := proto.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "index.scip")
	if err := os.WriteFile(p, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestExtract_EmptyIndex(t *testing.T) {
	dir := t.TempDir()
	p := writeIndex(t, dir, &scip.Index{})
	ex := &Extractor{IndexPath: p, ProjectRoot: dir}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Units) != 0 {
		t.Errorf("got %d units, want 0", len(res.Units))
	}
}

func TestExtract_DocumentWithNoSymbols(t *testing.T) {
	dir := t.TempDir()
	p := writeIndex(t, dir, &scip.Index{
		Metadata: &scip.Metadata{ProjectRoot: "file://" + dir},
		Documents: []*scip.Document{
			{Language: "Rust", RelativePath: "src/lib.rs"},
		},
	})
	ex := &Extractor{IndexPath: p}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Units) != 0 {
		t.Errorf("got %d units, want 0", len(res.Units))
	}
}

func TestExtract_UnparseableSymbol(t *testing.T) {
	dir := t.TempDir()
	p := writeIndex(t, dir, &scip.Index{
		Metadata: &scip.Metadata{ProjectRoot: "file://" + dir},
		Documents: []*scip.Document{{
			Language:     "Rust",
			RelativePath: "src/lib.rs",
			Symbols: []*scip.SymbolInformation{
				{Symbol: "not a valid scip symbol!!!", Kind: scip.SymbolInformation_Function},
			},
		}},
	})
	ex := &Extractor{IndexPath: p}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if len(res.Units) != 0 {
		t.Errorf("got %d units, want 0", len(res.Units))
	}
}

func TestExtract_ShortRangeArray(t *testing.T) {
	dir := t.TempDir()
	abs := filepath.Join(dir, "src/lib.rs")
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, []byte("fn f() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	p := writeIndex(t, dir, &scip.Index{
		Metadata: &scip.Metadata{ProjectRoot: "file://" + dir},
		Documents: []*scip.Document{{
			Language:     "Rust",
			RelativePath: "src/lib.rs",
			Symbols: []*scip.SymbolInformation{
				{
					Symbol:      "rust-analyzer cargo x 0.1.0 f().",
					Kind:        scip.SymbolInformation_Function,
					DisplayName: "f",
				},
			},
			Occurrences: []*scip.Occurrence{
				{
					Symbol:      "rust-analyzer cargo x 0.1.0 f().",
					Range:       []int32{}, // empty Range array
					SymbolRoles: int32(scip.SymbolRole_Definition),
				},
			},
		}},
	})
	ex := &Extractor{IndexPath: p}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract should not panic: %v", err)
	}
	_ = res
}

// TestExtract_MultiDocCrossModule exercises the multi-document path:
// a type defined in one file, a method in another, and a free function
// in a third file that calls the method. This verifies that the
// callgraph, type-method linking, and body extraction all work when
// symbols span multiple documents.
func TestExtract_MultiDocCrossModule(t *testing.T) {
	dir := t.TempDir()

	// types.rs: defines Counter struct
	typesRel := "src/types.rs"
	typesAbs := filepath.Join(dir, typesRel)
	if err := os.MkdirAll(filepath.Dir(typesAbs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(typesAbs, []byte("pub struct Counter {\n    n: i32,\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// impl.rs: defines Counter::bump method
	implRel := "src/impl.rs"
	implAbs := filepath.Join(dir, implRel)
	if err := os.WriteFile(implAbs, []byte("impl Counter {\n    pub fn bump(&mut self) -> i32 {\n        self.n += 1;\n        self.n\n    }\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// main.rs: calls Counter::bump
	mainRel := "src/main.rs"
	mainAbs := filepath.Join(dir, mainRel)
	if err := os.WriteFile(mainAbs, []byte("fn main() {\n    let mut c = Counter { n: 0 };\n    c.bump();\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	counterSym := "rust-analyzer cargo example 0.1.0 Counter#"
	bumpSym := "rust-analyzer cargo example 0.1.0 Counter#bump()."
	mainSym := "rust-analyzer cargo example 0.1.0 main()."

	index := &scip.Index{
		Metadata: &scip.Metadata{ProjectRoot: "file://" + dir},
		Documents: []*scip.Document{
			{
				Language:     "Rust",
				RelativePath: typesRel,
				Symbols: []*scip.SymbolInformation{
					{Symbol: counterSym, Kind: scip.SymbolInformation_Struct, DisplayName: "Counter"},
				},
				Occurrences: []*scip.Occurrence{
					{
						Symbol:         counterSym,
						Range:          []int32{0, 11, 0, 18},
						EnclosingRange: []int32{0, 0, 2, 1},
						SymbolRoles:    int32(scip.SymbolRole_Definition),
					},
				},
			},
			{
				Language:     "Rust",
				RelativePath: implRel,
				Symbols: []*scip.SymbolInformation{
					{Symbol: bumpSym, Kind: scip.SymbolInformation_Method, DisplayName: "bump"},
				},
				Occurrences: []*scip.Occurrence{
					{
						Symbol:         bumpSym,
						Range:          []int32{1, 11, 1, 15},
						EnclosingRange: []int32{1, 4, 4, 5},
						SymbolRoles:    int32(scip.SymbolRole_Definition),
					},
				},
			},
			{
				Language:     "Rust",
				RelativePath: mainRel,
				Symbols: []*scip.SymbolInformation{
					{Symbol: mainSym, Kind: scip.SymbolInformation_Function, DisplayName: "main"},
				},
				Occurrences: []*scip.Occurrence{
					{
						Symbol:         mainSym,
						Range:          []int32{0, 3, 0, 7},
						EnclosingRange: []int32{0, 0, 3, 1},
						SymbolRoles:    int32(scip.SymbolRole_Definition),
					},
					// main references bump on line 2
					{
						Symbol:      bumpSym,
						Range:       []int32{2, 6, 2, 10},
						SymbolRoles: int32(scip.SymbolRole_ReadAccess),
					},
				},
			},
		},
	}

	idxPath := writeIndex(t, dir, index)
	ex := &Extractor{IndexPath: idxPath}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	// Type defined in types.rs should be linked to method in impl.rs.
	ti, ok := res.Types["example.Counter"]
	if !ok {
		keys := make([]string, 0, len(res.Types))
		for k := range res.Types {
			keys = append(keys, k)
		}
		t.Fatalf("Counter type missing; got %v", keys)
	}
	if ti.Body == "" {
		t.Error("Counter body is empty")
	}
	if len(ti.Methods) != 1 {
		t.Errorf("Counter.Methods = %v, want 1 entry", ti.Methods)
	}

	// bump method should have ReceiverType linking back.
	var bumpFn *extractFunctionInfoView
	for _, u := range res.Units {
		for _, fn := range u.Functions {
			if fn.Name == "bump" {
				bumpFn = &extractFunctionInfoView{Receiver: fn.Receiver, ReceiverType: fn.ReceiverType}
			}
		}
	}
	if bumpFn == nil {
		t.Fatal("bump not found in units")
	}
	if bumpFn.ReceiverType != "example.Counter" {
		t.Errorf("bump.ReceiverType = %q, want example.Counter", bumpFn.ReceiverType)
	}

	// main should call bump (cross-document edge).
	var mainUnit *struct{ callees []string }
	for _, u := range res.Units {
		for _, fn := range u.Functions {
			if fn.Name == "main" {
				mainUnit = &struct{ callees []string }{callees: u.Callees}
			}
		}
	}
	if mainUnit == nil {
		t.Fatal("main unit not found")
	}
	foundBumpCallee := false
	for _, c := range mainUnit.callees {
		if c == "example.(Counter).bump" {
			foundBumpCallee = true
		}
	}
	if !foundBumpCallee {
		t.Errorf("main.Callees = %v, want to contain bump", mainUnit.callees)
	}
}

// TestExtract_PathFilterExcludesType verifies that when PathFilters
// include a method's document but exclude the type's document, the
// method is still extracted but its ReceiverType link is empty and the
// type is absent from Result.Types.
func TestExtract_PathFilterExcludesType(t *testing.T) {
	dir := t.TempDir()

	typesRel := "src/types.rs"
	typesAbs := filepath.Join(dir, typesRel)
	if err := os.MkdirAll(filepath.Dir(typesAbs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(typesAbs, []byte("pub struct Counter {\n    n: i32,\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	implRel := "src/impl.rs"
	implAbs := filepath.Join(dir, implRel)
	if err := os.WriteFile(implAbs, []byte("impl Counter {\n    pub fn bump(&mut self) -> i32 {\n        self.n += 1;\n        self.n\n    }\n}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	counterSym := "rust-analyzer cargo example 0.1.0 Counter#"
	bumpSym := "rust-analyzer cargo example 0.1.0 Counter#bump()."

	index := &scip.Index{
		Metadata: &scip.Metadata{ProjectRoot: "file://" + dir},
		Documents: []*scip.Document{
			{
				Language:     "Rust",
				RelativePath: typesRel,
				Symbols: []*scip.SymbolInformation{
					{Symbol: counterSym, Kind: scip.SymbolInformation_Struct, DisplayName: "Counter"},
				},
				Occurrences: []*scip.Occurrence{
					{
						Symbol:         counterSym,
						Range:          []int32{0, 11, 0, 18},
						EnclosingRange: []int32{0, 0, 2, 1},
						SymbolRoles:    int32(scip.SymbolRole_Definition),
					},
				},
			},
			{
				Language:     "Rust",
				RelativePath: implRel,
				Symbols: []*scip.SymbolInformation{
					{Symbol: bumpSym, Kind: scip.SymbolInformation_Method, DisplayName: "bump"},
				},
				Occurrences: []*scip.Occurrence{
					{
						Symbol:         bumpSym,
						Range:          []int32{1, 11, 1, 15},
						EnclosingRange: []int32{1, 4, 4, 5},
						SymbolRoles:    int32(scip.SymbolRole_Definition),
					},
				},
			},
		},
	}

	idxPath := writeIndex(t, dir, index)
	// Only include impl.rs — types.rs (and its Counter struct) are excluded.
	ex := &Extractor{IndexPath: idxPath, PathFilters: []string{"src/impl.rs"}}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	// Counter type should be absent.
	if len(res.Types) != 0 {
		t.Errorf("expected no types, got %v", res.Types)
	}

	// bump method should be extracted but unlinked.
	if len(res.Units) != 1 {
		t.Fatalf("got %d units, want 1", len(res.Units))
	}
	fn := res.Units[0].Functions[0]
	if fn.Name != "bump" {
		t.Errorf("Name = %q, want bump", fn.Name)
	}
	if fn.Receiver != "Counter" {
		t.Errorf("Receiver = %q, want Counter", fn.Receiver)
	}
	if fn.ReceiverType != "" {
		t.Errorf("ReceiverType = %q, want empty (type excluded by filter)", fn.ReceiverType)
	}
}
