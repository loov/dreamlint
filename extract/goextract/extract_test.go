package goextract

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/loov/dreamlint/extract"
)

func TestExtractFunctions(t *testing.T) {
	// Create a temp directory with a simple Go file
	dir := t.TempDir()

	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

// Hello returns a greeting.
func Hello(name string) string {
	return "Hello, " + name
}

func helper() {}
`
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	pkgs, err := LoadPackages(dir, "./...")
	if err != nil {
		t.Fatalf("LoadPackages: %v", err)
	}

	funcs := ExtractFunctions(pkgs)

	if len(funcs) != 2 {
		t.Fatalf("got %d functions, want 2", len(funcs))
	}

	// Check Hello function
	var hello *extract.FunctionInfo
	for _, f := range funcs {
		if f.Name == "Hello" {
			hello = f
			break
		}
	}

	if hello == nil {
		t.Fatal("Hello function not found")
	}

	if hello.Signature == "" {
		t.Error("Hello signature is empty")
	}

	if hello.Godoc == "" {
		t.Error("Hello godoc is empty")
	}
}

func TestExtractTypes_LinksMethodsToReceiver(t *testing.T) {
	dir := t.TempDir()

	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

// Counter accumulates a running total.
type Counter struct {
	n int
}

// Bump increments the counter.
func (c *Counter) Bump() int {
	c.n++
	return c.n
}

// Value returns the current total.
func (c Counter) Value() int {
	return c.n
}

type Shape interface {
	Area() float64
}

func free() {}
`
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	pkgs, err := LoadPackages(dir, "./...")
	if err != nil {
		t.Fatalf("LoadPackages: %v", err)
	}

	funcs := ExtractFunctions(pkgs)
	types := ExtractTypes(pkgs)
	byID := LinkMethodsToTypes(funcs, types)

	counter, ok := byID["testpkg.Counter"]
	if !ok {
		t.Fatalf("Counter type missing; got %d types", len(byID))
	}
	if counter.Kind != "struct" {
		t.Errorf("Counter.Kind = %q, want struct", counter.Kind)
	}
	if counter.Godoc == "" {
		t.Errorf("Counter.Godoc is empty")
	}
	wantMethods := map[string]bool{
		"testpkg.(*Counter).Bump": true,
		"testpkg.(Counter).Value": true,
	}
	if len(counter.Methods) != len(wantMethods) {
		t.Errorf("Counter.Methods = %v, want %d entries", counter.Methods, len(wantMethods))
	}
	for _, m := range counter.Methods {
		if !wantMethods[m] {
			t.Errorf("unexpected method ID %q in Counter.Methods", m)
		}
	}

	shape, ok := byID["testpkg.Shape"]
	if !ok {
		t.Fatal("Shape type missing")
	}
	if shape.Kind != "interface" {
		t.Errorf("Shape.Kind = %q, want interface", shape.Kind)
	}

	var bump, free *extract.FunctionInfo
	for _, fn := range funcs {
		switch fn.Name {
		case "Bump":
			bump = fn
		case "free":
			free = fn
		}
	}
	if bump == nil {
		t.Fatal("Bump not found")
	}
	if bump.ReceiverType != "testpkg.Counter" {
		t.Errorf("Bump.ReceiverType = %q, want testpkg.Counter", bump.ReceiverType)
	}
	if free == nil {
		t.Fatal("free not found")
	}
	if free.ReceiverType != "" {
		t.Errorf("free.ReceiverType = %q, want empty", free.ReceiverType)
	}
}

func TestStripReceiverPointer(t *testing.T) {
	cases := map[string]string{
		"Foo":         "Foo",
		"*Foo":        "Foo",
		"*Foo[T]":     "Foo",
		"Foo[T, U]":   "Foo",
		"*Foo[T, U]":  "Foo",
	}
	for in, want := range cases {
		if got := stripReceiverPointer(in); got != want {
			t.Errorf("stripReceiverPointer(%q) = %q, want %q", in, got, want)
		}
	}
}
