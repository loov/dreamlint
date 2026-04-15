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
	switch sym.Descriptors[len(sym.Descriptors)-1].Suffix {
	case scip.Descriptor_Method, scip.Descriptor_Macro:
		return true
	}
	return false
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
		receiver = receiverName(sym.Descriptors)
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

// receiverName returns the enclosing type (class / struct / impl target)
// for a function-like descriptor chain.
//
// Languages encode this differently:
//
//   - Java / C++ / TypeScript: `Foo#bar().` → descriptors
//     [Type "Foo", Method "bar"]. The last Type before the method is the
//     receiver.
//   - Rust (rust-analyzer): `impl#[Counter]bump().` → descriptors
//     [Type "impl", TypeParameter "Counter", Method "bump"]. The outer
//     Type "impl" is a synthetic wrapper; the impl target is the first
//     TypeParameter after it. Traits (`impl Default for Counter`) add
//     further TypeParameters; we still want the first one.
//
// Returns "" for free functions.
func receiverName(descriptors []*scip.Descriptor) string {
	if len(descriptors) < 2 {
		return ""
	}
	// rust-analyzer's impl wrapper convention.
	if descriptors[0].Suffix == scip.Descriptor_Type && descriptors[0].Name == "impl" &&
		descriptors[1].Suffix == scip.Descriptor_TypeParameter {
		return descriptors[1].Name
	}
	// Generic case: last Type descriptor before the method descriptor.
	for i := len(descriptors) - 2; i >= 0; i-- {
		d := descriptors[i]
		if d.Suffix == scip.Descriptor_Type {
			return d.Name
		}
		if d.Suffix != scip.Descriptor_TypeParameter {
			return ""
		}
	}
	return ""
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
