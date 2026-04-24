package scipextract

import (
	"go/token"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"

	"github.com/loov/dreamlint/extract"
)

// symbolClass is the high-level category this package assigns to a
// SymbolInformation. Centralising the Kind-switch + descriptor-fallback
// logic in classifySymbol prevents isFunctionSymbol and isTypeSymbol
// from drifting (see c658717, which realigned the external-callable
// filter with isFunctionSymbol after they diverged).
type symbolClass int

const (
	classOther symbolClass = iota
	classFunction
	classType
)

// classifySymbol maps a SymbolInformation to its class. Both the SCIP
// Kind and the descriptor suffix are consulted because indexers are
// inconsistent about which they populate.
func classifySymbol(info *scip.SymbolInformation) symbolClass {
	if info == nil || info.Symbol == "" {
		return classOther
	}
	if scip.IsLocalSymbol(info.Symbol) {
		return classOther
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
		return classFunction
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
		return classType
	}
	sym, err := scip.ParseSymbol(info.Symbol)
	if err != nil || len(sym.Descriptors) == 0 {
		return classOther
	}
	last := sym.Descriptors[len(sym.Descriptors)-1]
	if isCallableDescriptor(last) {
		return classFunction
	}
	if last.Suffix == scip.Descriptor_Type {
		return classType
	}
	return classOther
}

// isFunctionSymbol reports whether a SymbolInformation describes a callable.
func isFunctionSymbol(info *scip.SymbolInformation) bool {
	return classifySymbol(info) == classFunction
}

// isCallableDescriptor reports whether a SCIP descriptor identifies a
// callable entity (function, method, or macro). Shared between internal
// symbol filtering (isFunctionSymbol) and external-symbol filtering
// (buildExternalFunc) so both agree on the descriptor-only fallback.
func isCallableDescriptor(d *scip.Descriptor) bool {
	if d == nil {
		return false
	}
	switch d.Suffix {
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
	disambiguator := ""
	if len(sym.Descriptors) > 0 {
		last := sym.Descriptors[len(sym.Descriptors)-1]
		if name == "" {
			name = last.Name
		}
		disambiguator = last.Disambiguator
		receiver = receiverName(sym.Descriptors)
	}

	sig := ""
	if info.SignatureDocumentation != nil && info.SignatureDocumentation.Text != "" {
		sig = info.SignatureDocumentation.Text
	}

	pos := definitionPosition(info.Symbol, doc, absPath)

	return &extract.FunctionInfo{
		Package:       pkg,
		Name:          name,
		Disambiguator: disambiguator,
		Receiver:      receiver,
		Signature:     sig,
		Doc:           strings.Join(info.Documentation, "\n\n"),
		Position:      pos,
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
// metadata, namespace descriptors, and enclosing type descriptors.
//
// The last Type descriptor is excluded because it represents the
// receiver (for methods) or the type itself (for type symbols). All
// earlier Type descriptors are included so that nested classes in
// different outer types get distinct packages. For example,
// `Outer#Inner#method().` includes "Outer" in the package; "Inner" is
// the receiver, not part of the package.
func packageName(sym *scip.Symbol) string {
	var parts []string
	if sym.Package != nil && sym.Package.Name != "" {
		parts = append(parts, sym.Package.Name)
	}
	// Find the index of the last Type descriptor — it becomes the
	// receiver or type name and must NOT be part of the package.
	lastType := -1
	for i := len(sym.Descriptors) - 1; i >= 0; i-- {
		if sym.Descriptors[i].Suffix == scip.Descriptor_Type {
			lastType = i
			break
		}
	}
	for i, d := range sym.Descriptors {
		switch d.Suffix {
		case scip.Descriptor_Namespace:
			parts = append(parts, d.Name)
		case scip.Descriptor_Type:
			if i != lastType {
				parts = append(parts, d.Name)
			}
		}
	}
	return strings.Join(parts, "/")
}

// definitionPosition finds the first occurrence of sym in doc with the
// Definition role and converts it to a 1-based token.Position.
func definitionPosition(symbol string, doc *scip.Document, absPath string) token.Position {
	occ := findDefinition(symbol, doc)
	if occ == nil {
		return token.Position{Filename: absPath}
	}
	r, _ := scip.NewRange(occ.Range)
	return token.Position{
		Filename: absPath,
		Line:     int(r.Start.Line) + 1,
		Column:   int(r.Start.Character) + 1,
	}
}
