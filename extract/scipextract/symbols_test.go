package scipextract

import (
	"testing"

	"github.com/scip-code/scip/bindings/go/scip"
)

func TestIsFunctionSymbol(t *testing.T) {
	cases := []struct {
		name string
		info *scip.SymbolInformation
		want bool
	}{
		{
			name: "function by kind",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo example 0.1.0 foo/bar().",
				Kind:   scip.SymbolInformation_Function,
			},
			want: true,
		},
		{
			name: "method by kind",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo example 0.1.0 foo/Bar#baz().",
				Kind:   scip.SymbolInformation_Method,
			},
			want: true,
		},
		{
			name: "function by descriptor suffix only",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo example 0.1.0 foo/bar().",
			},
			want: true,
		},
		{
			name: "type symbol is not a function",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo example 0.1.0 foo/Bar#",
				Kind:   scip.SymbolInformation_Struct,
			},
			want: false,
		},
		{
			name: "local symbol excluded",
			info: &scip.SymbolInformation{
				Symbol: "local 12",
				Kind:   scip.SymbolInformation_Function,
			},
			want: false,
		},
		{
			name: "empty symbol excluded",
			info: &scip.SymbolInformation{},
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isFunctionSymbol(tc.info); got != tc.want {
				t.Errorf("isFunctionSymbol() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestBuildFunctionInfo(t *testing.T) {
	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Occurrences: []*scip.Occurrence{
			{
				Symbol:      "rust-analyzer cargo example 0.1.0 foo/Bar#baz().",
				Range:       []int32{10, 4, 10, 7},
				SymbolRoles: int32(scip.SymbolRole_Definition),
			},
		},
	}
	info := &scip.SymbolInformation{
		Symbol:      "rust-analyzer cargo example 0.1.0 foo/Bar#baz().",
		Kind:        scip.SymbolInformation_Method,
		DisplayName: "baz",
		Documentation: []string{
			"baz adds spice.",
		},
		SignatureDocumentation: &scip.Document{
			Language: "Rust",
			Text:     "fn baz(self) -> u32",
		},
	}

	fn := buildFunctionInfo(info, doc, "/repo/src/lib.rs")
	if fn == nil {
		t.Fatal("buildFunctionInfo returned nil")
	}
	if fn.Name != "baz" {
		t.Errorf("Name = %q, want baz", fn.Name)
	}
	if fn.Receiver != "Bar" {
		t.Errorf("Receiver = %q, want Bar", fn.Receiver)
	}
	if fn.Package != "example/foo" {
		t.Errorf("Package = %q, want example/foo", fn.Package)
	}
	if fn.Signature != "fn baz(self) -> u32" {
		t.Errorf("Signature = %q, want fn baz(self) -> u32", fn.Signature)
	}
	if fn.Doc != "baz adds spice." {
		t.Errorf("Doc = %q, want baz adds spice.", fn.Doc)
	}
	if fn.Position.Filename != "/repo/src/lib.rs" {
		t.Errorf("Position.Filename = %q, want /repo/src/lib.rs", fn.Position.Filename)
	}
	if fn.Position.Line != 11 || fn.Position.Column != 5 {
		t.Errorf("Position = %d:%d, want 11:5 (1-based)", fn.Position.Line, fn.Position.Column)
	}
}

func TestDefinitionPosition_MultipleDefinitions(t *testing.T) {
	sym := "scip-clang pkg example 0.1.0 foo()."
	doc := &scip.Document{
		RelativePath: "src/main.cpp",
		Occurrences: []*scip.Occurrence{
			{
				Symbol:      sym,
				Range:       []int32{2, 5, 2, 8},
				SymbolRoles: int32(scip.SymbolRole_Definition),
			},
			{
				Symbol:      sym,
				Range:       []int32{10, 5, 10, 8},
				SymbolRoles: int32(scip.SymbolRole_Definition),
			},
		},
	}
	pos := definitionPosition(sym, doc, "/repo/src/main.cpp")
	if pos.Line != 3 || pos.Column != 6 {
		t.Errorf("got %d:%d, want 3:6 (first-wins, 1-based)", pos.Line, pos.Column)
	}
}

func TestDefinitionPosition_NoDefinition(t *testing.T) {
	sym := "rust-analyzer cargo example 0.1.0 bar()."
	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Occurrences: []*scip.Occurrence{
			{
				Symbol:      sym,
				Range:       []int32{5, 4, 5, 7},
				SymbolRoles: int32(scip.SymbolRole_ReadAccess),
			},
		},
	}
	pos := definitionPosition(sym, doc, "/repo/src/lib.rs")
	if pos.Line != 0 {
		t.Errorf("got Line=%d, want 0 (no definition)", pos.Line)
	}
	if pos.Filename != "/repo/src/lib.rs" {
		t.Errorf("Filename = %q, want /repo/src/lib.rs", pos.Filename)
	}
}

func TestIsCallableDescriptor(t *testing.T) {
	cases := []struct {
		name string
		d    *scip.Descriptor
		want bool
	}{
		{"Method", &scip.Descriptor{Suffix: scip.Descriptor_Method}, true},
		{"Macro", &scip.Descriptor{Suffix: scip.Descriptor_Macro}, true},
		{"Type", &scip.Descriptor{Suffix: scip.Descriptor_Type}, false},
		{"Namespace", &scip.Descriptor{Suffix: scip.Descriptor_Namespace}, false},
		{"nil", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isCallableDescriptor(tc.d); got != tc.want {
				t.Errorf("isCallableDescriptor() = %v, want %v", got, tc.want)
			}
		})
	}
}
