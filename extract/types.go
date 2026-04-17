package extract

import "go/token"

// FunctionInfo holds information about a single function.
//
// The Receiver field carries whatever a language treats as the enclosing
// type or scope of a method (Go receiver, Rust impl target, C++ class,
// Java enclosing class, etc.). It is empty for free functions.
//
// ReceiverType is the ID of the enclosing TypeInfo (Package + "." + Name)
// when the extractor was able to resolve the receiver to a type known to
// the indexed project. Empty for free functions and for methods on types
// defined outside the analyzed code.
type FunctionInfo struct {
	Package        string
	Name           string
	Disambiguator  string
	Receiver       string
	ReceiverType   string
	Signature      string
	Body           string
	Doc            string
	Position       token.Position
}

// ID returns the canonical function ID used to key functions in maps
// and to reference them from unit callgraphs:
//
//	Package + "." + Name                 (free functions)
//	Package + ".(" + Receiver + ")." + Name (methods)
//
// When Disambiguator is non-empty (overloaded methods in Java, C++,
// TypeScript), it is appended as Name + "(" + Disambiguator + ")" so
// each overload gets a unique ID.
func (f *FunctionInfo) ID() string {
	name := f.Name
	if f.Disambiguator != "" {
		name = f.Name + "(" + f.Disambiguator + ")"
	}
	if f.Receiver != "" {
		return f.Package + ".(" + f.Receiver + ")." + name
	}
	return f.Package + "." + name
}

// TypeInfo holds information about a user-defined type: class, struct,
// interface, trait, enum, or type alias. Methods defined on the type
// are collected in Methods as function IDs.
type TypeInfo struct {
	Package   string
	Name      string
	Kind      string // "class" | "struct" | "interface" | "trait" | "enum" | "type"
	Signature string
	Body      string
	Doc       string
	Position  token.Position
	Methods   []string
}

// ExternalFunc holds shallow info about a function defined outside the
// analyzed code (a dependency).
type ExternalFunc struct {
	Package   string
	Name      string
	Signature string
	Doc       string
}
