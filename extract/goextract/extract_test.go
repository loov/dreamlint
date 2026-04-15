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
