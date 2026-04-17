package scipextract

import (
	"sort"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"

	"github.com/loov/dreamlint/extract"
)

const maxTypeBodyLines = 200

// isTypeSymbol reports whether a SymbolInformation describes a
// user-defined type (class, struct, interface, trait, enum, type alias).
// Both the SCIP Descriptor suffix and the higher-level Kind are consulted
// because indexers are inconsistent about which they populate.
func isTypeSymbol(info *scip.SymbolInformation) bool {
	if info == nil || info.Symbol == "" {
		return false
	}
	if scip.IsLocalSymbol(info.Symbol) {
		return false
	}
	switch info.Kind {
	case scip.SymbolInformation_Class,
		scip.SymbolInformation_Struct,
		scip.SymbolInformation_Interface,
		scip.SymbolInformation_Trait,
		scip.SymbolInformation_Enum,
		scip.SymbolInformation_Type,
		scip.SymbolInformation_TypeAlias,
		scip.SymbolInformation_TypeClass,
		scip.SymbolInformation_Protocol,
		scip.SymbolInformation_Union:
		return true
	}
	// Fall back to descriptor suffix for indexers that leave Kind unset.
	// A type symbol ends in a Type descriptor with no following Method or
	// TypeParameter.
	sym, err := scip.ParseSymbol(info.Symbol)
	if err != nil || len(sym.Descriptors) == 0 {
		return false
	}
	last := sym.Descriptors[len(sym.Descriptors)-1]
	return last.Suffix == scip.Descriptor_Type
}

// kindString maps a SCIP Kind to the short tag the prompt renders. For
// kinds not specifically recognized we return "type".
func kindString(kind scip.SymbolInformation_Kind) string {
	switch kind {
	case scip.SymbolInformation_Class:
		return "class"
	case scip.SymbolInformation_Struct:
		return "struct"
	case scip.SymbolInformation_Interface:
		return "interface"
	case scip.SymbolInformation_Trait:
		return "trait"
	case scip.SymbolInformation_Enum:
		return "enum"
	case scip.SymbolInformation_Protocol:
		return "protocol"
	case scip.SymbolInformation_Union:
		return "union"
	case scip.SymbolInformation_TypeAlias:
		return "type"
	}
	return "type"
}

// buildTypeInfo constructs an extract.TypeInfo from a type-like
// SymbolInformation. Returns nil if the symbol can't be parsed or
// describes a rust-analyzer-style synthetic "impl" wrapper (these
// carry the impl'd type's signature/doc but are keyed by the
// trait name, which would collide with the real type entry).
func buildTypeInfo(info *scip.SymbolInformation, doc *scip.Document, absPath string) *extract.TypeInfo {
	sym, err := scip.ParseSymbol(info.Symbol)
	if err != nil {
		return nil
	}
	if len(sym.Descriptors) > 0 && sym.Descriptors[0].Suffix == scip.Descriptor_Type &&
		sym.Descriptors[0].Name == "impl" {
		return nil
	}

	pkg := packageName(sym)
	name := info.DisplayName
	if name == "" && len(sym.Descriptors) > 0 {
		last := sym.Descriptors[len(sym.Descriptors)-1]
		if last.Suffix != scip.Descriptor_Type {
			return nil
		}
		name = last.Name
	}
	if name == "" {
		return nil
	}

	sig := ""
	if info.SignatureDocumentation != nil && info.SignatureDocumentation.Text != "" {
		sig = info.SignatureDocumentation.Text
	}

	pos := definitionPosition(info.Symbol, doc, absPath)

	return &extract.TypeInfo{
		Package:   pkg,
		Name:      name,
		Kind:      kindString(info.Kind),
		Signature: sig,
		Doc:       strings.Join(info.Documentation, "\n\n"),
		Position:  pos,
	}
}

// typeID is the canonical ID for a type in the analysis pipeline.
// Matches the receiver-linking convention used in FunctionInfo and the
// unit ID builder.
func typeID(t *extract.TypeInfo) string {
	return t.Package + "." + t.Name
}

// typeDefinitionRanges returns a per-symbol range that covers each
// type's declaration body. It mirrors definitionRanges but is kept
// separate so it doesn't interfere with the function-body span
// heuristic (a type's body usually encloses its methods).
func typeDefinitionRanges(doc *scip.Document, typeSymbols map[string]bool) map[string]scip.Range {
	type entry struct {
		sym          string
		r            scip.Range
		hasEnclosing bool
	}
	var entries []entry
	seen := make(map[string]bool)
	for _, occ := range doc.Occurrences {
		if occ.SymbolRoles&int32(scip.SymbolRole_Definition) == 0 {
			continue
		}
		if occ.Symbol == "" || !typeSymbols[occ.Symbol] || seen[occ.Symbol] {
			continue
		}
		seen[occ.Symbol] = true

		var r scip.Range
		hasEnc := false
		if len(occ.EnclosingRange) > 0 {
			if rr, err := scip.NewRange(occ.EnclosingRange); err == nil {
				r = rr
				hasEnc = true
			}
		}
		if !hasEnc {
			rr, err := scip.NewRange(occ.Range)
			if err != nil {
				continue
			}
			r = rr
		}
		entries = append(entries, entry{sym: occ.Symbol, r: r, hasEnclosing: hasEnc})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].r.Start.Less(entries[j].r.Start)
	})

	out := make(map[string]scip.Range, len(entries))
	for i, e := range entries {
		if !e.hasEnclosing {
			if i+1 < len(entries) {
				e.r.End = scip.Position{Line: entries[i+1].r.Start.Line}
			} else {
				e.r.End = scip.Position{Line: e.r.Start.Line + maxTypeBodyLines}
			}
		}
		out[e.sym] = e.r
	}
	return out
}

// extractTypeBody slices the source text for a type's declaration. Falls
// back to the type's signature text when the range can't be resolved.
func extractTypeBody(info *scip.SymbolInformation, doc *scip.Document, absPath string, ranges map[string]scip.Range, cache *sourceCache) string {
	r, ok := ranges[info.Symbol]
	if !ok {
		return ""
	}
	body, ok := sliceRange(r, doc, absPath, cache)
	if !ok {
		return ""
	}
	return body
}
