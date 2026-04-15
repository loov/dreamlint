package scipextract

import (
	"go/token"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"

	"github.com/loov/dreamlint/extract"
)

// isFunctionSymbol reports whether a SymbolInformation describes a callable
// (function, method, constructor, ...). Both the SCIP Descriptor suffix
// and the higher-level Kind are consulted, because older indexers sometimes
// only emit one.
func isFunctionSymbol(info *scip.SymbolInformation) bool {
	if info == nil || info.Symbol == "" {
		return false
	}
	if scip.IsLocalSymbol(info.Symbol) {
		return false
	}
	switch info.Kind {
	case scip.SymbolInformation_Function,
		scip.SymbolInformation_Method,
		scip.SymbolInformation_AbstractMethod,
		scip.SymbolInformation_StaticMethod,
		scip.SymbolInformation_Constructor,
		scip.SymbolInformation_MethodSpecification,
		scip.SymbolInformation_ProtocolMethod,
		scip.SymbolInformation_PureVirtualMethod,
		scip.SymbolInformation_TraitMethod,
		scip.SymbolInformation_TypeClassMethod,
		scip.SymbolInformation_SingletonMethod,
		scip.SymbolInformation_MethodAlias,
		scip.SymbolInformation_Macro:
		return true
	}
	sym, err := scip.ParseSymbol(info.Symbol)
	if err != nil || len(sym.Descriptors) == 0 {
		return false
	}
	return sym.Descriptors[len(sym.Descriptors)-1].Suffix == scip.Descriptor_Method
}

// buildFunctionInfo constructs an extract.FunctionInfo from a function-like
// SymbolInformation. Returns nil if the symbol can't be parsed.
func buildFunctionInfo(info *scip.SymbolInformation, doc *scip.Document, absPath string) *extract.FunctionInfo {
	sym, err := scip.ParseSymbol(info.Symbol)
	if err != nil {
		return nil
	}

	pkg := packageName(sym)
	name := info.DisplayName
	receiver := ""
	if len(sym.Descriptors) > 0 {
		last := sym.Descriptors[len(sym.Descriptors)-1]
		if name == "" {
			name = last.Name
		}
		// A type descriptor one level up from the method typically represents
		// the enclosing class / trait / impl target.
		for i := len(sym.Descriptors) - 2; i >= 0; i-- {
			d := sym.Descriptors[i]
			if d.Suffix == scip.Descriptor_Type {
				receiver = d.Name
				break
			}
			if d.Suffix != scip.Descriptor_TypeParameter {
				break
			}
		}
	}

	sig := ""
	if info.SignatureDocumentation != nil && info.SignatureDocumentation.Text != "" {
		sig = info.SignatureDocumentation.Text
	}

	pos := definitionPosition(info.Symbol, doc, absPath)

	return &extract.FunctionInfo{
		Package:   pkg,
		Name:      name,
		Receiver:  receiver,
		Signature: sig,
		Godoc:     strings.Join(info.Documentation, "\n\n"),
		Position:  pos,
	}
}

// packageName builds a package identifier from the SCIP Symbol's Package
// metadata plus any namespace descriptors. Languages that don't express
// packages (e.g. C) leave Package nil; we then fall back to joined
// namespace descriptor names.
func packageName(sym *scip.Symbol) string {
	var parts []string
	if sym.Package != nil && sym.Package.Name != "" {
		parts = append(parts, sym.Package.Name)
	}
	for _, d := range sym.Descriptors {
		if d.Suffix == scip.Descriptor_Namespace {
			parts = append(parts, d.Name)
		}
	}
	return strings.Join(parts, "/")
}

// definitionPosition finds the first occurrence of sym in doc with the
// Definition role and converts it to a 1-based token.Position.
func definitionPosition(symbol string, doc *scip.Document, absPath string) token.Position {
	for _, occ := range doc.Occurrences {
		if occ.Symbol != symbol {
			continue
		}
		if occ.SymbolRoles&int32(scip.SymbolRole_Definition) == 0 {
			continue
		}
		if len(occ.Range) == 0 {
			continue
		}
		return token.Position{
			Filename: absPath,
			Line:     int(occ.Range[0]) + 1,
			Column:   int(occ.Range[1]) + 1,
		}
	}
	return token.Position{Filename: absPath}
}
