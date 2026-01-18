// main.go
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/loov/reviewmod/analyze"
	"github.com/loov/reviewmod/cache"
	"github.com/loov/reviewmod/config"
	"github.com/loov/reviewmod/extract"
	"github.com/loov/reviewmod/llm"
	"github.com/loov/reviewmod/report"
)

func main() {
	configPath := flag.String("config", "reviewmod.cue", "path to config file")
	format := flag.String("format", "both", "output format: json, markdown, or both")
	flag.Parse()

	patterns := flag.Args()
	if len(patterns) == 0 {
		patterns = []string{"./..."}
	}

	if err := run(*configPath, *format, patterns); err != nil {
		log.Fatal(err)
	}
}

func run(configPath, format string, patterns []string) error {
	// Load config
	cfg, err := config.LoadConfig(configPath)
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}

	// Create LLM client
	client := llm.NewOpenAIClient(cfg.LLM.BaseURL, cfg.LLM.APIKey)

	// Create cache
	var c *cache.Cache
	if cfg.Cache.Enabled {
		c = cache.New(cfg.Cache.Dir)
	}

	// Load packages once
	fmt.Println("Loading packages...")
	pkgs, err := extract.LoadPackages(".", patterns...)
	if err != nil {
		return fmt.Errorf("load packages: %w", err)
	}

	// Extract functions
	fmt.Println("Extracting functions...")
	funcs := extract.ExtractFunctions(pkgs)
	fmt.Printf("Found %d functions\n", len(funcs))

	// Build callgraph
	fmt.Println("Building callgraph...")
	graph := extract.BuildCallgraph(pkgs)

	// Extract external function info
	fmt.Println("Extracting external functions...")
	externalFuncs := extract.ExtractExternalFuncs(pkgs, graph)
	fmt.Printf("Found %d external functions\n", len(externalFuncs))

	// Build analysis units
	fmt.Println("Building analysis units...")
	units := extract.BuildAnalysisUnits(funcs, graph)
	fmt.Printf("Created %d analysis units\n", len(units))

	// Create pipeline
	pipeline := analyze.NewPipeline(cfg, c, client, externalFuncs)
	if err := pipeline.LoadPrompts(); err != nil {
		return fmt.Errorf("load prompts: %w", err)
	}

	// Create report
	rpt := report.NewReport()
	rpt.Metadata.Modules = patterns
	rpt.Metadata.ConfigFile = configPath
	rpt.Metadata.TotalUnits = len(units)
	rpt.Metadata.GeneratedAt = time.Now()

	// Analyze each unit in order
	ctx := context.Background()
	calleeSummaries := make(map[string]*analyze.SummaryResponse)

	for i, unit := range units {
		fmt.Printf("Analyzing %d/%d: %s\n", i+1, len(units), unit.ID)

		unitReport, err := pipeline.Analyze(ctx, unit, calleeSummaries)
		if err != nil {
			return fmt.Errorf("analyze %s: %w", unit.ID, err)
		}

		rpt.Units[unit.ID] = *unitReport

		// Store summary for callers
		if summary := pipeline.GetSummary(unit.ID); summary != nil {
			calleeSummaries[unit.ID] = summary
		}

		// Update report summary
		for _, issue := range unitReport.Issues {
			rpt.Summary.TotalIssues++
			rpt.Summary.BySeverity[string(issue.Severity)]++
			rpt.Summary.ByCategory[issue.Category]++

			if issue.Severity == report.SeverityCritical {
				found := false
				for _, u := range rpt.Summary.CriticalUnits {
					if u == unit.ID {
						found = true
						break
					}
				}
				if !found {
					rpt.Summary.CriticalUnits = append(rpt.Summary.CriticalUnits, unit.ID)
				}
			}
		}
	}

	// Write output
	if format == "json" || format == "both" {
		if err := report.WriteJSONFile(rpt, cfg.Output.JSON); err != nil {
			return fmt.Errorf("write json: %w", err)
		}
		fmt.Printf("Wrote %s\n", cfg.Output.JSON)
	}

	if format == "markdown" || format == "both" {
		if err := report.WriteMarkdownFile(rpt, cfg.Output.Markdown); err != nil {
			return fmt.Errorf("write markdown: %w", err)
		}
		fmt.Printf("Wrote %s\n", cfg.Output.Markdown)
	}

	// Print summary
	fmt.Printf("\nAnalysis complete: %d issues found\n", rpt.Summary.TotalIssues)
	for sev, count := range rpt.Summary.BySeverity {
		fmt.Printf("  %s: %d\n", sev, count)
	}

	return nil
}
