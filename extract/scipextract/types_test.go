package scipextract

import (
	"testing"

	"github.com/scip-code/scip/bindings/go/scip"
)

func TestIsTypeSymbol(t *testing.T) {
	cases := []struct {
		name string
		info *scip.SymbolInformation
		want bool
	}{
		{
			name: "class by kind",
			info: &scip.SymbolInformation{
				Symbol: "scip-typescript npm ts 1.0.0 src/m.ts`Foo#",
				Kind:   scip.SymbolInformation_Class,
			},
			want: true,
		},
		{
			name: "struct by kind",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo ex 0.1.0 Foo#",
				Kind:   scip.SymbolInformation_Struct,
			},
			want: true,
		},
		{
			name: "interface by kind",
			info: &scip.SymbolInformation{
				Symbol: "scip-java maven org.e e 1.0 Foo#",
				Kind:   scip.SymbolInformation_Interface,
			},
			want: true,
		},
		{
			name: "trait by kind",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo ex 0.1.0 Foo#",
				Kind:   scip.SymbolInformation_Trait,
			},
			want: true,
		},
		{
			name: "enum by kind",
			info: &scip.SymbolInformation{
				Symbol: "scip-java maven org.e e 1.0 Foo#",
				Kind:   scip.SymbolInformation_Enum,
			},
			want: true,
		},
		{
			name: "descriptor fallback for type-suffix symbol",
			info: &scip.SymbolInformation{
				Symbol: "scip-clang . . . math/Counter#",
			},
			want: true,
		},
		{
			name: "function rejected",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo ex 0.1.0 foo/bar().",
				Kind:   scip.SymbolInformation_Function,
			},
			want: false,
		},
		{
			name: "method rejected",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo ex 0.1.0 foo/Foo#bar().",
				Kind:   scip.SymbolInformation_Method,
			},
			want: false,
		},
		{
			name: "local symbol rejected",
			info: &scip.SymbolInformation{
				Symbol: "local 1",
				Kind:   scip.SymbolInformation_Class,
			},
			want: false,
		},
		{
			name: "empty rejected",
			info: &scip.SymbolInformation{},
			want: false,
		},
		{
			name: "parameter descriptor rejected",
			info: &scip.SymbolInformation{
				Symbol: "rust-analyzer cargo example 0.1.0 foo/bar().(x)",
			},
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isTypeSymbol(tc.info); got != tc.want {
				t.Errorf("isTypeSymbol(%q) = %v, want %v", tc.info.Symbol, got, tc.want)
			}
		})
	}
}

func TestBuildTypeInfo(t *testing.T) {
	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Occurrences: []*scip.Occurrence{
			{
				Symbol:      "rust-analyzer cargo example 0.1.0 Counter#",
				Range:       []int32{5, 11, 18},
				SymbolRoles: int32(scip.SymbolRole_Definition),
			},
		},
	}
	info := &scip.SymbolInformation{
		Symbol:                 "rust-analyzer cargo example 0.1.0 Counter#",
		Kind:                   scip.SymbolInformation_Struct,
		DisplayName:            "Counter",
		Documentation:          []string{"A running counter."},
		SignatureDocumentation: &scip.Document{Language: "Rust", Text: "pub struct Counter"},
	}

	ti := buildTypeInfo(info, doc, "/repo/src/lib.rs")
	if ti == nil {
		t.Fatal("buildTypeInfo returned nil")
	}
	if ti.Name != "Counter" {
		t.Errorf("Name = %q, want Counter", ti.Name)
	}
	if ti.Package != "example" {
		t.Errorf("Package = %q, want example", ti.Package)
	}
	if ti.Kind != "struct" {
		t.Errorf("Kind = %q, want struct", ti.Kind)
	}
	if ti.Signature != "pub struct Counter" {
		t.Errorf("Signature = %q, want pub struct Counter", ti.Signature)
	}
	if ti.Doc != "A running counter." {
		t.Errorf("Doc = %q, want 'A running counter.'", ti.Doc)
	}
	if ti.Position.Filename != "/repo/src/lib.rs" {
		t.Errorf("Position.Filename = %q", ti.Position.Filename)
	}
	if ti.Position.Line != 6 || ti.Position.Column != 12 {
		t.Errorf("Position = %d:%d, want 6:12 (1-based)", ti.Position.Line, ti.Position.Column)
	}
}

func TestBuildTypeInfo_RejectsRustImplWrapper(t *testing.T) {
	// rust-analyzer emits an impl block as a synthetic TypeAlias whose
	// descriptor chain starts with Type "impl". These carry the impl'd
	// type's signature/doc and would collide with the real type entry.
	info := &scip.SymbolInformation{
		Symbol:        "rust-analyzer cargo example 0.1.0 impl#[Counter][Default]",
		Kind:          scip.SymbolInformation_TypeAlias,
		Documentation: []string{"Counter accumulates bumps into a running total."},
	}
	doc := &scip.Document{}
	if ti := buildTypeInfo(info, doc, "/repo/src/lib.rs"); ti != nil {
		t.Errorf("buildTypeInfo should reject impl wrapper, got %+v", ti)
	}
}

func TestKindString(t *testing.T) {
	cases := map[scip.SymbolInformation_Kind]string{
		scip.SymbolInformation_Class:     "class",
		scip.SymbolInformation_Struct:    "struct",
		scip.SymbolInformation_Interface: "interface",
		scip.SymbolInformation_Trait:     "trait",
		scip.SymbolInformation_Enum:      "enum",
		scip.SymbolInformation_TypeClass: "typeclass",
		scip.SymbolInformation_TypeAlias: "type",
		scip.SymbolInformation_Function:  "type", // not a type kind → fallback
	}
	for k, want := range cases {
		if got := kindString(k); got != want {
			t.Errorf("kindString(%v) = %q, want %q", k, got, want)
		}
	}
}
