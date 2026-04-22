package scipextract

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/loov/dreamlint/extract"
)

// TestGolden_JavaExample pins the extraction of the committed
// testdata/java-example/index.scip fixture. The fixture is regenerated
// via testdata/java-example/generate.sh (Docker + scip-java);
// regenerate and update this test when the Java sources change.
func TestGolden_JavaExample(t *testing.T) {
	root, err := filepath.Abs("testdata/java-example")
	if err != nil {
		t.Fatal(err)
	}

	ex := &Extractor{
		IndexPath:   filepath.Join(root, "index.scip"),
		ProjectRoot: root,
	}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	if res.Language != "Java" {
		t.Errorf("Language = %q, want Java", res.Language)
	}

	pkg := "maven/com.example/java-example/example"
	sccID := pkg + ".(Math).isEven+" + pkg + ".(Math).isOdd"

	wantUnits := map[string]bool{
		pkg + ".(Math).add":          true,
		pkg + ".(Math).multiply":     true,
		pkg + ".(Math).<init>":       true,
		pkg + ".(Counter).<init>":    true,
		pkg + ".(Counter).bump":      true,
		pkg + ".(Counter).value":     true,
		pkg + ".(Main).<init>":       true,
		pkg + ".(Main).main":         true,
		sccID:                        true,
	}
	gotUnits := map[string]bool{}
	for _, u := range res.Units {
		gotUnits[u.ID] = true
	}
	for id := range wantUnits {
		if !gotUnits[id] {
			keys := make([]string, 0, len(gotUnits))
			for k := range gotUnits {
				keys = append(keys, k)
			}
			t.Errorf("missing unit %q; got %v", id, keys)
		}
	}
	for id := range gotUnits {
		if !wantUnits[id] {
			t.Errorf("unexpected unit %q", id)
		}
	}

	byID := map[string]*extract.AnalysisUnit{}
	for _, u := range res.Units {
		byID[u.ID] = u
	}

	// Static method body extracted.
	add := byID[pkg+".(Math).add"]
	assertBodyContains(t, "add", add.Functions[0].Body,
		"public static int add", "return a + b;")

	// Class method calls static method.
	bump := byID[pkg+".(Counter).bump"]
	if got := bump.Functions[0].Receiver; got != "Counter" {
		t.Errorf("bump.Receiver = %q, want Counter", got)
	}
	if !slices.Contains(bump.Callees, pkg+".(Math).add") {
		t.Errorf("bump.Callees = %v, want to contain Math.add", bump.Callees)
	}

	// Mutual recursion SCC.
	scc := byID[sccID]
	if scc == nil {
		t.Fatalf("SCC unit %q not found", sccID)
	}
	if len(scc.Functions) != 2 {
		t.Fatalf("SCC unit has %d functions, want 2", len(scc.Functions))
	}
	names := map[string]bool{}
	for _, f := range scc.Functions {
		names[f.Name] = true
	}
	if !names["isEven"] || !names["isOdd"] {
		t.Errorf("SCC functions = %v, want {isEven, isOdd}", names)
	}

	// main calls most of the project.
	main := byID[pkg+".(Main).main"]
	for _, want := range []string{
		pkg + ".(Math).add",
		pkg + ".(Math).multiply",
		pkg + ".(Counter).<init>",
		pkg + ".(Counter).bump",
		pkg + ".(Counter).value",
		sccID,
	} {
		if !slices.Contains(main.Callees, want) {
			t.Errorf("main.Callees missing %q; got %v", want, main.Callees)
		}
	}

	// Types: Counter and Math are classes.
	for _, name := range []string{"Counter", "Math", "Main"} {
		ti, ok := res.Types[pkg+"."+name]
		if !ok {
			t.Errorf("type %s missing", name)
			continue
		}
		if ti.Kind != "class" {
			t.Errorf("%s.Kind = %q, want class", name, ti.Kind)
		}
		if !strings.Contains(ti.Body, name) {
			t.Errorf("%s body missing type name; body=%q", name, ti.Body)
		}
	}

	// External: println overloads.
	if len(res.External) == 0 {
		t.Error("expected external entries for PrintStream.println, got 0")
	}
}
