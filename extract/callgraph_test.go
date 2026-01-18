// extract/callgraph_test.go
package extract

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestBuildCallgraph(t *testing.T) {
	dir := t.TempDir()

	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

func A() { B() }
func B() { C() }
func C() {}
`
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	pkgs, err := LoadPackages(dir, "./...")
	if err != nil {
		t.Fatalf("LoadPackages: %v", err)
	}

	graph := BuildCallgraph(pkgs)

	// A calls B
	if !slices.Contains(graph["testpkg.A"], "testpkg.B") {
		t.Errorf("A should call B, got %v", graph["testpkg.A"])
	}

	// B calls C
	if !slices.Contains(graph["testpkg.B"], "testpkg.C") {
		t.Errorf("B should call C, got %v", graph["testpkg.B"])
	}

	// C calls nothing
	if len(graph["testpkg.C"]) != 0 {
		t.Errorf("C should call nothing, got %v", graph["testpkg.C"])
	}
}
