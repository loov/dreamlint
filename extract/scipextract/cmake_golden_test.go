package scipextract

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/loov/dreamlint/extract"
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
	// Silence per-function "no enclosing range" warnings from scip-clang.
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

	sccID := "math.is_even+math.is_odd"
	want := map[string]bool{
		"math.add":                true,
		"math.multiply":           true,
		"math.(Counter).Counter":  true,
		"math.(Counter).bump":     true,
		"math.(Counter).value":    true,
		sccID:                     true,
		".main":                   true,
	}
	got := map[string]bool{}
	byID := map[string]*extract.AnalysisUnit{}
	for _, u := range res.Units {
		got[u.ID] = true
		byID[u.ID] = u
	}
	for id := range want {
		if !got[id] {
			t.Errorf("missing unit %q; got %v", id, keys(got))
		}
	}
	for id := range got {
		if !want[id] {
			t.Errorf("unexpected unit %q", id)
		}
	}

	// Topology: leaves precede their callers. Check by index-of.
	indexOf := func(id string) int {
		for i, u := range res.Units {
			if u.ID == id {
				return i
			}
		}
		return -1
	}
	for _, pair := range [][2]string{
		{"math.add", "math.multiply"},
		{"math.add", "math.(Counter).bump"},
		{"math.(Counter).Counter", ".main"},
		{sccID, ".main"},
	} {
		if indexOf(pair[0]) >= indexOf(pair[1]) {
			t.Errorf("topology: %q should precede %q", pair[0], pair[1])
		}
	}

	// Class methods carry the Counter receiver (the last Type descriptor
	// before the method, in scip-clang's encoding).
	for _, id := range []string{
		"math.(Counter).Counter",
		"math.(Counter).bump",
		"math.(Counter).value",
	} {
		if r := byID[id].Functions[0].Receiver; r != "Counter" {
			t.Errorf("%s receiver = %q, want Counter", id, r)
		}
	}

	// Mutual recursion lands in a single SCC.
	scc := byID[sccID]
	if len(scc.Functions) != 2 {
		t.Fatalf("SCC unit has %d functions, want 2", len(scc.Functions))
	}
	names := map[string]bool{}
	for _, f := range scc.Functions {
		names[f.Name] = true
	}
	if !names["is_even"] || !names["is_odd"] {
		t.Errorf("SCC functions = %v, want {is_even, is_odd}", names)
	}
	if len(scc.Callees) != 0 {
		t.Errorf("SCC.Callees = %v, want empty (mutual recursion is internal)", scc.Callees)
	}

	// Callgraph checks.
	if !slices.Contains(byID["math.(Counter).bump"].Callees, "math.add") {
		t.Errorf("bump.Callees = %v, want to contain math.add",
			byID["math.(Counter).bump"].Callees)
	}
	if !slices.Contains(byID["math.multiply"].Callees, "math.add") {
		t.Errorf("multiply.Callees = %v, want to contain math.add",
			byID["math.multiply"].Callees)
	}
	mainCallees := byID[".main"].Callees
	for _, want := range []string{
		"math.add",
		"math.multiply",
		"math.(Counter).Counter",
		"math.(Counter).bump",
		"math.(Counter).value",
		sccID,
	} {
		if !slices.Contains(mainCallees, want) {
			t.Errorf("main.Callees missing %q; got %v", want, mainCallees)
		}
	}

	// Bodies contain the expected snippets despite scip-clang's lack of
	// EnclosingRange (the span-to-next-function heuristic takes over).
	assertBodyContains(t, "add", byID["math.add"].Functions[0].Body,
		"int add(int a, int b)", "return a + b;")
	assertBodyContains(t, "bump", byID["math.(Counter).bump"].Functions[0].Body,
		"int Counter::bump()", "n = add(n, 1);")
	for _, f := range scc.Functions {
		if !strings.Contains(f.Body, "return is_"+flip(f.Name)+"(n - 1);") {
			t.Errorf("%s body missing mutual-recursion call; got %q", f.Name, f.Body)
		}
	}

	// std::printf as external.
	sawPrintf := false
	for _, e := range res.External {
		if e.Name == "printf" && e.Package == "std" {
			sawPrintf = true
		}
	}
	if !sawPrintf {
		t.Errorf("expected external entry for std::printf, got %d externals", len(res.External))
	}
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

// keys returns the keys of m as a slice (for stable failure messages).
func keys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// flip returns "odd" when given "even" and vice versa, used to assert
// mutual-recursion call sites inside the is_even / is_odd bodies.
func flip(name string) string {
	switch name {
	case "is_even":
		return "odd"
	case "is_odd":
		return "even"
	}
	return name
}
