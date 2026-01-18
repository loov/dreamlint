// pipeline_golden_test.go
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/loov/reviewmod/analyze"
	"github.com/loov/reviewmod/config"
	"github.com/loov/reviewmod/extract"
	"github.com/loov/reviewmod/llm"
	"golang.org/x/tools/txtar"
)

var update = flag.Bool("update", false, "update golden files")

func TestPipelineGolden(t *testing.T) {
	files, err := filepath.Glob("testdata/pipeline/*.txtar")
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		name := strings.TrimSuffix(filepath.Base(file), ".txtar")
		t.Run(name, func(t *testing.T) {
			runPipelineGoldenTest(t, file)
		})
	}
}

func runPipelineGoldenTest(t *testing.T, txtarFile string) {
	archive, err := txtar.ParseFile(txtarFile)
	if err != nil {
		t.Fatalf("parse txtar: %v", err)
	}

	// Parse config from archive comment
	var passName, pkgPattern string
	comment := string(archive.Comment)
	for _, line := range strings.Split(comment, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "pass:") {
			passName = strings.TrimSpace(strings.TrimPrefix(line, "pass:"))
		}
		if strings.HasPrefix(line, "package:") {
			pkgPattern = strings.TrimSpace(strings.TrimPrefix(line, "package:"))
		}
	}

	if passName == "" {
		t.Fatal("missing pass: directive in txtar comment")
	}
	if pkgPattern == "" {
		pkgPattern = "testdata/testpkg"
	}

	// Find want.txt
	var want string
	for _, f := range archive.Files {
		if f.Name == "want.txt" {
			want = string(f.Data)
		}
	}

	// Create minimal config with just summary and the requested pass
	cfg := &config.Config{
		LLM: config.LLMConfig{
			Model:       "test-model",
			MaxTokens:   1000,
			Temperature: 0,
		},
		Cache: config.CacheConfig{
			Enabled: false,
		},
		Analyses: []config.AnalysisPass{
			{Name: "summary", Prompt: "prompts/summary.txt", Enabled: true},
			{Name: passName, Prompt: fmt.Sprintf("prompts/%s.txt", passName), Enabled: true},
		},
	}

	// Load packages from testdata directory
	pkgs, err := extract.LoadPackages(pkgPattern, "./...")
	if err != nil {
		t.Fatalf("load packages: %v", err)
	}

	// Extract functions
	funcs := extract.ExtractFunctions(pkgs)
	if len(funcs) == 0 {
		t.Fatal("no functions found")
	}

	// Build callgraph and units
	graph := extract.BuildCallgraph(pkgs)
	externalFuncs := extract.ExtractExternalFuncs(pkgs, graph)
	units := extract.BuildAnalysisUnits(funcs, graph)

	// Create mock client with summary response
	mockClient := llm.NewMockClient(
		llm.Response{Content: `{"purpose": "test purpose", "behavior": "test behavior", "invariants": [], "security": []}`},
	)

	// Create and run pipeline
	pipeline := analyze.NewPipeline(cfg, nil, mockClient, externalFuncs)
	if err := pipeline.LoadPrompts(); err != nil {
		t.Fatalf("load prompts: %v", err)
	}

	// Analyze first unit only
	ctx := context.Background()
	_, err = pipeline.Analyze(ctx, units[0], nil)
	if err != nil {
		t.Fatalf("analyze: %v", err)
	}

	// Get captured prompts
	prompts := mockClient.Prompts()

	// Build output - one file per prompt
	var got strings.Builder
	for i, prompt := range prompts {
		if i > 0 {
			got.WriteString("\n")
		}
		got.WriteString(fmt.Sprintf("== prompt %d ==\n", i))
		got.WriteString(prompt)
		got.WriteString("\n")
	}

	gotStr := strings.TrimSpace(got.String())
	wantStr := strings.TrimSpace(want)

	if *update {
		// Update the golden file
		archive.Files = []txtar.File{
			{Name: "want.txt", Data: []byte(gotStr + "\n")},
		}
		if err := os.WriteFile(txtarFile, txtar.Format(archive), 0644); err != nil {
			t.Fatalf("update golden: %v", err)
		}
		return
	}

	if gotStr != wantStr {
		t.Errorf("output mismatch.\n\nGot:\n%s\n\nWant:\n%s", gotStr, wantStr)
	}
}
