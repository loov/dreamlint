package extract

import "go/token"

// FunctionInfo holds information about a single function.
//
// The Receiver field carries whatever a language treats as the enclosing
// type or scope of a method (Go receiver, Rust impl target, C++ class,
// Java enclosing class, etc.). It is empty for free functions.
type FunctionInfo struct {
	Package   string
	Name      string
	Receiver  string
	Signature string
	Body      string
	Godoc     string
	Position  token.Position
}

// ExternalFunc holds shallow info about a function defined outside the
// analyzed code (a dependency).
type ExternalFunc struct {
	Package   string
	Name      string
	Signature string
	Godoc     string
}
