package scipextract

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/scip-code/scip/bindings/go/scip"
)

func TestExtractBody_EnclosingRange_FromDocText(t *testing.T) {
	source := "fn foo() -> u32 {\n    1 + 2\n}\n"
	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Text:         source,
		Occurrences: []*scip.Occurrence{
			{
				Symbol:         "rust-analyzer cargo example 0.1.0 foo().",
				Range:          []int32{0, 3, 0, 6},
				EnclosingRange: []int32{0, 0, 2, 1},
				SymbolRoles:    int32(scip.SymbolRole_Definition),
			},
		},
	}
	info := &scip.SymbolInformation{
		Symbol:      "rust-analyzer cargo example 0.1.0 foo().",
		Kind:        scip.SymbolInformation_Function,
		DisplayName: "foo",
	}
	cache := newSourceCache("")
	ranges := definitionRanges(doc, map[string]string{info.Symbol: "example.foo"}).ByID
	body, warn := extractBody(info, doc, "/repo/src/lib.rs", ranges, cache)
	if warn != "" {
		t.Errorf("unexpected warning: %s", warn)
	}
	want := "fn foo() -> u32 {\n    1 + 2\n}"
	if body != want {
		t.Errorf("body mismatch:\n got %q\nwant %q", body, want)
	}
}

func TestExtractBody_FallsBackToDefinitionRange(t *testing.T) {
	source := "fn foo() -> u32 {\n    1 + 2\n}\n"
	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Text:         source,
		Occurrences: []*scip.Occurrence{
			{
				Symbol:      "rust-analyzer cargo example 0.1.0 foo().",
				Range:       []int32{0, 3, 0, 6},
				SymbolRoles: int32(scip.SymbolRole_Definition),
			},
		},
	}
	info := &scip.SymbolInformation{
		Symbol: "rust-analyzer cargo example 0.1.0 foo().",
		Kind:   scip.SymbolInformation_Function,
	}
	cache := newSourceCache("")
	ranges := definitionRanges(doc, map[string]string{info.Symbol: "example.foo"}).ByID
	body, warn := extractBody(info, doc, "/repo/src/lib.rs", ranges, cache)
	if warn == "" {
		t.Error("expected a warning when EnclosingRange is missing")
	}
	if !strings.HasPrefix(body, "fn foo()") {
		t.Errorf("expected signature-line fallback, got %q", body)
	}
}

func TestExtractBody_ReadsFromDisk(t *testing.T) {
	dir := t.TempDir()
	rel := "src/lib.rs"
	abs := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	content := "fn foo() -> u32 {\n    1 + 2\n}\n"
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	doc := &scip.Document{
		RelativePath: rel,
		Occurrences: []*scip.Occurrence{
			{
				Symbol:         "rust-analyzer cargo example 0.1.0 foo().",
				Range:          []int32{0, 3, 0, 6},
				EnclosingRange: []int32{0, 0, 2, 1},
				SymbolRoles:    int32(scip.SymbolRole_Definition),
			},
		},
	}
	info := &scip.SymbolInformation{
		Symbol: "rust-analyzer cargo example 0.1.0 foo().",
		Kind:   scip.SymbolInformation_Function,
	}
	cache := newSourceCache(dir)
	ranges := definitionRanges(doc, map[string]string{info.Symbol: "example.foo"}).ByID
	body, warn := extractBody(info, doc, abs, ranges, cache)
	if warn != "" {
		t.Errorf("unexpected warning: %s", warn)
	}
	if body != "fn foo() -> u32 {\n    1 + 2\n}" {
		t.Errorf("body mismatch: %q", body)
	}
}

func TestSliceLinesHalfOpen_EdgeCases(t *testing.T) {
	text := "line0\nline1\nline2\nline3\n"

	t.Run("empty text line 0", func(t *testing.T) {
		got, ok := sliceLinesHalfOpen("", scip.Range{
			Start: scip.Position{Line: 0},
			End:   scip.Position{Line: 0, Character: 1},
		})
		if !ok {
			t.Fatal("expected ok — SplitAfter yields one empty element")
		}
		if got != "" {
			t.Errorf("got %q, want empty", got)
		}
	})

	t.Run("empty text line 1", func(t *testing.T) {
		_, ok := sliceLinesHalfOpen("", scip.Range{
			Start: scip.Position{Line: 1},
			End:   scip.Position{Line: 1, Character: 1},
		})
		if ok {
			t.Error("expected false for line beyond single empty element")
		}
	})

	t.Run("start beyond EOF", func(t *testing.T) {
		_, ok := sliceLinesHalfOpen(text, scip.Range{
			Start: scip.Position{Line: 99},
			End:   scip.Position{Line: 100, Character: 1},
		})
		if ok {
			t.Error("expected false for start beyond EOF")
		}
	})

	t.Run("end character zero excludes end line", func(t *testing.T) {
		got, ok := sliceLinesHalfOpen(text, scip.Range{
			Start: scip.Position{Line: 0},
			End:   scip.Position{Line: 2, Character: 0},
		})
		if !ok {
			t.Fatal("expected ok")
		}
		if got != "line0\nline1" {
			t.Errorf("got %q, want %q", got, "line0\nline1")
		}
	})

	t.Run("end past EOF clamped", func(t *testing.T) {
		got, ok := sliceLinesHalfOpen(text, scip.Range{
			Start: scip.Position{Line: 2},
			End:   scip.Position{Line: 999, Character: 1},
		})
		if !ok {
			t.Fatal("expected ok")
		}
		if got != "line2\nline3" {
			t.Errorf("got %q, want %q", got, "line2\nline3")
		}
	})

	t.Run("single line", func(t *testing.T) {
		got, ok := sliceLinesHalfOpen(text, scip.Range{
			Start: scip.Position{Line: 1},
			End:   scip.Position{Line: 1, Character: 5},
		})
		if !ok {
			t.Fatal("expected ok")
		}
		if got != "line1" {
			t.Errorf("got %q, want %q", got, "line1")
		}
	})

	t.Run("end before start clamped to start", func(t *testing.T) {
		got, ok := sliceLinesHalfOpen(text, scip.Range{
			Start: scip.Position{Line: 2},
			End:   scip.Position{Line: 0, Character: 0},
		})
		if !ok {
			t.Fatal("expected ok")
		}
		if got != "line2" {
			t.Errorf("got %q, want %q", got, "line2")
		}
	})
}

func TestSliceRange_PreferDocText(t *testing.T) {
	text := "fn a() {}\nfn b() {}\n"
	doc := &scip.Document{
		RelativePath: "src/lib.rs",
		Text:         text,
	}
	cache := newSourceCache("/nonexistent")
	r := scip.Range{
		Start: scip.Position{Line: 0},
		End:   scip.Position{Line: 0, Character: 9},
	}
	got, ok := sliceRange(r, doc, "/nonexistent/src/lib.rs", cache)
	if !ok {
		t.Fatal("expected ok from doc.Text path")
	}
	if got != "fn a() {}" {
		t.Errorf("got %q", got)
	}
}

func TestSliceRange_DiskFallback(t *testing.T) {
	dir := t.TempDir()
	abs := filepath.Join(dir, "main.rs")
	if err := os.WriteFile(abs, []byte("fn x() {}\nfn y() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	doc := &scip.Document{RelativePath: "main.rs"}
	cache := newSourceCache(dir)
	r := scip.Range{
		Start: scip.Position{Line: 1},
		End:   scip.Position{Line: 1, Character: 9},
	}
	got, ok := sliceRange(r, doc, abs, cache)
	if !ok {
		t.Fatal("expected ok from disk path")
	}
	if got != "fn y() {}" {
		t.Errorf("got %q", got)
	}
}

func TestSliceRange_MissingFile(t *testing.T) {
	doc := &scip.Document{RelativePath: "gone.rs"}
	cache := newSourceCache("/nonexistent")
	r := scip.Range{
		Start: scip.Position{Line: 0},
		End:   scip.Position{Line: 0, Character: 5},
	}
	_, ok := sliceRange(r, doc, "/nonexistent/gone.rs", cache)
	if ok {
		t.Error("expected false for missing file")
	}
}
