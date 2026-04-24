package scipextract

import (
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
	case scip.SymbolInformation_TypeClass:
		return "typeclass"
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
	return receiverKey(t.Package, t.Name)
}

// receiverKey is the package-qualified identity used both to key types
// and to link methods to their receivers. Keeping both call sites on
// one helper prevents them from drifting (e.g. if the separator or
// ordering ever needs to change).
func receiverKey(pkg, name string) string {
	return pkg + "." + name
}

// typeDefinitionRanges returns a per-symbol range that covers each
// type's declaration body. It mirrors definitionRanges but is kept
// separate so it doesn't interfere with the function-body span
// heuristic (a type's body usually encloses its methods).
func typeDefinitionRanges(doc *scip.Document, typeSymbols map[string]bool) map[string]scip.Range {
	return collectDefinitionRanges(doc, func(sym string) bool {
		return typeSymbols[sym]
	}, func(startLine int32) int32 { return startLine + maxTypeBodyLines })
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
