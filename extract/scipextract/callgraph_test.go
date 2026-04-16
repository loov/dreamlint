package scipextract

import (
	"slices"
	"testing"

	"github.com/scip-code/scip/bindings/go/scip"
)

func TestBuildCallgraph_InternalEdge(t *testing.T) {
	aSym := "rust-analyzer cargo example 0.1.0 a()."
	bSym := "rust-analyzer cargo example 0.1.0 b()."

	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Symbols: []*scip.SymbolInformation{
			{Symbol: aSym, Kind: scip.SymbolInformation_Function, DisplayName: "a"},
			{Symbol: bSym, Kind: scip.SymbolInformation_Function, DisplayName: "b"},
		},
		Occurrences: []*scip.Occurrence{
			// A defined, enclosing lines 0..2.
			{
				Symbol:         aSym,
				Range:          []int32{0, 3, 0, 4},
				EnclosingRange: []int32{0, 0, 2, 1},
				SymbolRoles:    int32(scip.SymbolRole_Definition),
			},
			// B defined, enclosing lines 4..6.
			{
				Symbol:         bSym,
				Range:          []int32{4, 3, 4, 4},
				EnclosingRange: []int32{4, 0, 6, 1},
				SymbolRoles:    int32(scip.SymbolRole_Definition),
			},
			// A references B on line 1.
			{
				Symbol:      bSym,
				Range:       []int32{1, 4, 1, 5},
				SymbolRoles: int32(scip.SymbolRole_ReadAccess),
			},
		},
	}

	symToID := map[string]string{
		aSym: "example.a",
		bSym: "example.b",
	}
	docRanges := map[*scip.Document]map[string]scip.Range{
		doc: definitionRanges(doc, symToID),
	}

	graph, external := buildCallgraph([]*scip.Document{doc}, &scip.Index{}, docRanges, symToID)
	if got := graph["example.a"]; !slices.Contains(got, "example.b") {
		t.Errorf("expected a -> b edge, got %v", got)
	}
	if len(graph["example.b"]) != 0 {
		t.Errorf("b should have no callees, got %v", graph["example.b"])
	}
	if len(external) != 0 {
		t.Errorf("expected no external funcs, got %v", external)
	}
}

func TestBuildCallgraph_ExternalRef(t *testing.T) {
	aSym := "rust-analyzer cargo example 0.1.0 a()."
	extSym := "rust-analyzer cargo std 1.0.0 io/println()."

	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Symbols: []*scip.SymbolInformation{
			{Symbol: aSym, Kind: scip.SymbolInformation_Function, DisplayName: "a"},
		},
		Occurrences: []*scip.Occurrence{
			{
				Symbol:         aSym,
				Range:          []int32{0, 3, 0, 4},
				EnclosingRange: []int32{0, 0, 2, 1},
				SymbolRoles:    int32(scip.SymbolRole_Definition),
			},
			{
				Symbol:      extSym,
				Range:       []int32{1, 4, 1, 11},
				SymbolRoles: int32(scip.SymbolRole_ReadAccess),
			},
		},
	}

	index := &scip.Index{
		ExternalSymbols: []*scip.SymbolInformation{
			{
				Symbol:      extSym,
				Kind:        scip.SymbolInformation_Function,
				DisplayName: "println",
				Documentation: []string{
					"Prints to stdout.",
				},
				SignatureDocumentation: &scip.Document{Text: "fn println(s: &str)"},
			},
		},
	}

	symToID := map[string]string{aSym: "example.a"}
	docRanges := map[*scip.Document]map[string]scip.Range{
		doc: definitionRanges(doc, symToID),
	}

	graph, external := buildCallgraph([]*scip.Document{doc}, index, docRanges, symToID)

	id := externalID(extSym)
	if !slices.Contains(graph["example.a"], id) {
		t.Errorf("expected external edge, got %v", graph["example.a"])
	}
	ext, ok := external[id]
	if !ok {
		t.Fatalf("external %s not recorded", id)
	}
	if ext.Name != "println" {
		t.Errorf("ExternalFunc.Name = %q, want println", ext.Name)
	}
	if ext.Signature != "fn println(s: &str)" {
		t.Errorf("ExternalFunc.Signature = %q", ext.Signature)
	}
	if ext.Doc != "Prints to stdout." {
		t.Errorf("ExternalFunc.Doc = %q", ext.Doc)
	}
}
