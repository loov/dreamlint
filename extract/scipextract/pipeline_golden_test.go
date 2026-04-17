package scipextract_test

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/loov/dreamlint/analyze"
	"github.com/loov/dreamlint/config"
	"github.com/loov/dreamlint/extract"
	"github.com/loov/dreamlint/extract/scipextract"
	"github.com/loov/dreamlint/llm"
)

var updatePipelineGoldens = flag.Bool("update-scip-pipeline", false,
	"update SCIP pipeline golden files under testdata/<fixture>/pipeline.golden")

// TestPipelineGolden_SCIP runs each SCIP fixture through the full
// analyze pipeline (summary + baseline passes) with a mock LLM client
// and pins the captured prompts to a per-fixture golden file. This is
// the end-to-end assurance that everything dreamlint builds on top of
// the extracted units — SCC grouping, callee summaries, Receiver /
// Language rendering, external function context — actually reaches
// the LLM in the shape we expect.
//
// Regenerate goldens with:
//
//	go test ./extract/scipextract/... -update-scip-pipeline
func TestPipelineGolden_SCIP(t *testing.T) {
	fixtures := []string{
		"cmake-example",
		"rust-example",
		"typescript-example",
	}

	for _, fixture := range fixtures {
		t.Run(fixture, func(t *testing.T) {
			root, err := filepath.Abs(filepath.Join("testdata", fixture))
			if err != nil {
				t.Fatal(err)
			}
			runPipelineGolden(t, root)
		})
	}
}

func runPipelineGolden(t *testing.T, root string) {
	t.Helper()

	ex := &scipextract.Extractor{
		IndexPath:   filepath.Join(root, "index.scip"),
		ProjectRoot: root,
	}
	res, err := ex.Extract(context.Background())
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}

	cfg := &config.Config{
		LLM: config.LLMConfig{
			Model:       "test-model",
			MaxTokens:   1000,
			Temperature: 0,
		},
		Cache: config.CacheConfig{Enabled: false},
		Analyse: []config.AnalysisPass{
			{Name: "summary", Prompt: "builtin:summary", Enabled: true},
			{Name: "baseline", Prompt: "builtin:baseline", Enabled: true},
		},
	}

	// Pre-seed responses:
	//   - one type-summary response per extracted type (pinned text so
	//     the rendered receiver-type block in downstream prompts is
	//     deterministic).
	//   - two responses per unit: a deterministic function summary
	//     (so callee context blocks render "Purpose: ...") and an
	//     empty issues response.
	// Extra calls beyond this fall back to the mock's default
	// `{"issues": []}`.
	typeSummaryJSON := `{"purpose": "test type purpose", "behavior": "test type behavior", ` +
		`"invariants": ["test type invariant"], "security": []}`
	summaryJSON := `{"purpose": "test purpose", "behavior": "test behavior", ` +
		`"invariants": ["test invariant"], "security": []}`
	var responses []llm.Response
	for range res.Types {
		responses = append(responses, llm.Response{Content: typeSummaryJSON})
	}
	for range res.Units {
		responses = append(responses,
			llm.Response{Content: summaryJSON},
			llm.Response{Content: `{"issues": []}`},
		)
	}
	mock := llm.NewMockClient(responses...)

	// Function lookup for receiver-type sibling rendering.
	funcByID := make(map[string]*extract.FunctionInfo)
	for _, unit := range res.Units {
		for _, fn := range unit.Functions {
			funcByID[fn.ID()] = fn
		}
	}

	pipeline := analyze.NewPipeline(cfg, nil, mock, res.External, res.Types, funcByID)
	pipeline.SetLanguage(res.Language)
	if err := pipeline.LoadPrompts(); err != nil {
		t.Fatalf("load prompts: %v", err)
	}

	ctx := context.Background()
	if err := pipeline.AnalyzeTypes(ctx); err != nil {
		t.Fatalf("analyze types: %v", err)
	}

	calleeSummaries := make(map[string]*analyze.SummaryResponse)
	for _, unit := range res.Units {
		if _, err := pipeline.Analyze(ctx, unit, calleeSummaries); err != nil {
			t.Fatalf("analyze %s: %v", unit.ID, err)
		}
		if s := pipeline.GetSummary(unit.ID); s != nil {
			calleeSummaries[unit.ID] = s
		}
	}

	got := renderPipelinePrompts(res, mock.Prompts())

	goldenPath := filepath.Join(root, "pipeline.golden")
	if *updatePipelineGoldens {
		if err := os.WriteFile(goldenPath, []byte(got), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("wrote %s", goldenPath)
		return
	}

	wantBytes, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden %s: %v (run with -update-scip-pipeline to create)",
			goldenPath, err)
	}
	want := string(wantBytes)
	if got != want {
		t.Errorf("pipeline prompts mismatch for %s.\n"+
			"Re-run with -update-scip-pipeline after verifying the diff.\n"+
			"\n--- got ---\n%s\n--- want ---\n%s",
			filepath.Base(root), got, want)
	}
}

// renderPipelinePrompts produces a deterministic, human-readable
// dump of every LLM call made during the pipeline run. The header
// per prompt names the unit and pass so the golden file stays
// legible even with dozens of prompts.
func renderPipelinePrompts(res *extract.Result, prompts []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fixture language: %s\n", res.Language)
	fmt.Fprintf(&b, "units (%d):\n", len(res.Units))
	for i, u := range res.Units {
		fmt.Fprintf(&b, "  [%d] %s (callees=%v)\n", i, u.ID, u.Callees)
	}
	fmt.Fprintf(&b, "types (%d):\n", len(res.Types))
	typeIDs := make([]string, 0, len(res.Types))
	for id := range res.Types {
		typeIDs = append(typeIDs, id)
	}
	sortStrings(typeIDs)
	for _, id := range typeIDs {
		t := res.Types[id]
		fmt.Fprintf(&b, "  %s (%s, methods=%v)\n", id, t.Kind, t.Methods)
	}
	fmt.Fprintf(&b, "external symbols (%d):\n", len(res.External))
	// Sort for determinism.
	extIDs := make([]string, 0, len(res.External))
	for id := range res.External {
		extIDs = append(extIDs, id)
	}
	sortStrings(extIDs)
	for _, id := range extIDs {
		e := res.External[id]
		fmt.Fprintf(&b, "  %s::%s\n", e.Package, e.Name)
	}
	b.WriteString("\n")

	promptIdx := 0

	// Type summary prompts come first (AnalyzeTypes iterates types in
	// sorted ID order — match that here so the golden is stable).
	for _, id := range typeIDs {
		if promptIdx >= len(prompts) {
			break
		}
		fmt.Fprintf(&b, "== prompt %d :: type=%s :: pass=type_summary ==\n",
			promptIdx, id)
		b.WriteString(prompts[promptIdx])
		if !strings.HasSuffix(prompts[promptIdx], "\n") {
			b.WriteString("\n")
		}
		b.WriteString("\n")
		promptIdx++
	}

	passNames := []string{"summary", "baseline"}
	for _, unit := range res.Units {
		for _, pass := range passNames {
			if promptIdx >= len(prompts) {
				break
			}
			fmt.Fprintf(&b, "== prompt %d :: unit=%s :: pass=%s ==\n",
				promptIdx, unit.ID, pass)
			b.WriteString(prompts[promptIdx])
			if !strings.HasSuffix(prompts[promptIdx], "\n") {
				b.WriteString("\n")
			}
			b.WriteString("\n")
			promptIdx++
		}
	}
	return b.String()
}

// sortStrings sorts a slice in-place without pulling in the
// stdlib sort package at every call site.
func sortStrings(s []string) {
	// Small inputs, insertion sort is fine.
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}

