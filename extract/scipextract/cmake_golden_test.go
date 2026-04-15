package scipextract

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestGolden_CMakeExample pins the extraction of the committed
// testdata/cmake-example/index.scip fixture. The fixture is regenerated via
// testdata/cmake-example/generate.sh (Docker + scip-clang); regenerate and
// update this test when the C++ sources change.
func TestGolden_CMakeExample(t *testing.T) {
	root, err := filepath.Abs("testdata/cmake-example")
	if err != nil {
		t.Fatal(err)
	}
	// Silence warnings from scip-clang's missing EnclosingRange support.
	defer redirectStderr(t)()

	ex := &Extractor{
		IndexPath:   filepath.Join(root, "index.scip"),
		ProjectRoot: root,
	}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if res.Language != "C++" {
		t.Errorf("Language = %q, want C++", res.Language)
	}

	wantUnits := []string{"math.add", "math.multiply", ".main"}
	if len(res.Units) != len(wantUnits) {
		var got []string
		for _, u := range res.Units {
			got = append(got, u.ID)
		}
		t.Fatalf("units = %v, want %v", got, wantUnits)
	}
	for i, want := range wantUnits {
		if got := res.Units[i].ID; got != want {
			t.Errorf("units[%d].ID = %q, want %q", i, got, want)
		}
	}

	byID := map[string]*unitSummary{}
	for _, u := range res.Units {
		byID[u.ID] = &unitSummary{
			callees: u.Callees,
			body:    u.Functions[0].Body,
			godoc:   u.Functions[0].Godoc,
		}
	}

	// math.add is a leaf.
	if got := byID["math.add"].callees; len(got) != 0 {
		t.Errorf("math.add.Callees = %v, want empty", got)
	}
	assertBodyContains(t, "math.add", byID["math.add"].body,
		"int add(int a, int b)",
		"return a + b;")

	// math.multiply calls add.
	if got := byID["math.multiply"].callees; !slices.Contains(got, "math.add") {
		t.Errorf("math.multiply.Callees = %v, want to contain math.add", got)
	}
	assertBodyContains(t, "math.multiply", byID["math.multiply"].body,
		"int multiply(int a, int b)",
		"result = add(result, a);")

	// main calls both math.multiply and math.add (scip-clang emits both
	// references; the callgraph over-approximation treats both as edges).
	mainCallees := byID[".main"].callees
	if !slices.Contains(mainCallees, "math.multiply") {
		t.Errorf("main.Callees = %v, want to contain math.multiply", mainCallees)
	}
	if !slices.Contains(mainCallees, "math.add") {
		t.Errorf("main.Callees = %v, want to contain math.add", mainCallees)
	}
	assertBodyContains(t, "main", byID[".main"].body,
		"int main()",
		"math::add(2, 3)",
		"math::multiply(sum, 4)")

	// Godoc from the Doxygen-style comments in math.h/math.cpp.
	if !strings.Contains(byID["math.add"].godoc, "sum of a and b") {
		t.Errorf("math.add.Godoc = %q, want mention of 'sum of a and b'", byID["math.add"].godoc)
	}

	// The only non-internal reference in the sources is std::printf.
	foundPrintf := false
	for _, ext := range res.External {
		if ext.Name == "printf" && ext.Package == "std" {
			foundPrintf = true
		}
	}
	if !foundPrintf {
		t.Errorf("expected an external entry for std::printf, got %d externals", len(res.External))
	}
}

type unitSummary struct {
	callees []string
	body    string
	godoc   string
}

func assertBodyContains(t *testing.T, name, body string, needles ...string) {
	t.Helper()
	for _, n := range needles {
		if !strings.Contains(body, n) {
			t.Errorf("%s body missing %q; body=%q", name, n, body)
		}
	}
}

// redirectStderr silences stderr for the duration of the test so the
// per-function "no enclosing range" warnings don't clutter -v output.
func redirectStderr(t *testing.T) func() {
	t.Helper()
	orig := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	done := make(chan struct{})
	go func() {
		io.Copy(io.Discard, r)
		close(done)
	}()
	return func() {
		w.Close()
		os.Stderr = orig
		<-done
		r.Close()
	}
}
