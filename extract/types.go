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
	Package      string
	Name         string
	Receiver     string
	ReceiverType string
	Signature    string
	Body         string
	Godoc        string
	Position     token.Position
}

// ID returns the canonical function ID used to key functions in maps
// and to reference them from unit callgraphs:
//
//	Package + "." + Name                 (free functions)
//	Package + ".(" + Receiver + ")." + Name (methods)
func (f *FunctionInfo) ID() string {
	if f.Receiver != "" {
		return f.Package + ".(" + f.Receiver + ")." + f.Name
	}
	return f.Package + "." + f.Name
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
	Godoc     string
	Position  token.Position
	Methods   []string
}

// ExternalFunc holds shallow info about a function defined outside the
// analyzed code (a dependency).
type ExternalFunc struct {
	Package   string
	Name      string
	Signature string
	Godoc     string
}
