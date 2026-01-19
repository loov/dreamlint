package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mattn/go-isatty"
	"github.com/zeebo/clingy"

	"github.com/loov/dreamlint/analyze"
	"github.com/loov/dreamlint/cache"
	"github.com/loov/dreamlint/config"
	"github.com/loov/dreamlint/extract"
	"github.com/loov/dreamlint/llm"
	"github.com/loov/dreamlint/report"
	"github.com/loov/dreamlint/report/markdown"
	"github.com/loov/dreamlint/report/sarif"
)

type cmdRun struct {
	configPaths   []string
	inlineConfigs []string
	format        string
	resume        bool
	promptsDir    string
	patterns      []string
}

func (c *cmdRun) Setup(params clingy.Parameters) {
	c.configPaths = params.Flag("config", "path to config file",
		[]string{"dreamlint.cue"},
		clingy.Repeated,
	).([]string)

	c.inlineConfigs = params.Flag("c", "inline CUE config",
		[]string{},
		clingy.Repeated,
	).([]string)

	c.format = params.Flag("format", "output format: json, markdown, sarif, or all", "all").(string)

	c.resume = params.Flag("resume", "resume from existing partial report", false,
		clingy.Boolean,
	).(bool)

	c.promptsDir = params.Flag("prompts", "directory to load prompts from", "").(string)

	c.patterns = params.Arg("patterns", "packages to analyze",
		clingy.Optional,
		clingy.Repeated,
	).([]string)
}

func (c *cmdRun) Execute(ctx context.Context) error {
	patterns := c.patterns
	if len(patterns) == 0 {
		patterns = []string{"./..."}
	}

	return run(c.configPaths, c.inlineConfigs, c.format, c.resume, c.promptsDir, patterns)
}

// isTTY reports whether stdout is a terminal.
var isTTY = isatty.IsTerminal(os.Stdout.Fd()) || isatty.IsCygwinTerminal(os.Stdout.Fd())

// clearLine clears the current line if stdout is a terminal.
func clearLine() {
	if isTTY {
		fmt.Print("\r\033[K")
	}
}

// printProgress prints a progress message, overwriting the current line if stdout is a terminal.
func printProgress(format string, args ...any) {
	if isTTY {
		fmt.Print("\r\033[K")
		fmt.Printf(format, args...)
	}
}

func run(configPaths, inlineConfigs []string, format string, resume bool, promptsDir string, patterns []string) error {
	// Load config
	cfg, err := config.LoadConfig(configPaths, inlineConfigs)
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
	if promptsDir != "" {
		pipeline.SetPromptsFS(os.DirFS(promptsDir))
	}
	if err := pipeline.LoadPrompts(); err != nil {
		return fmt.Errorf("load prompts: %w", err)
	}

	// Load or create report
	var rpt *report.Report
	if resume {
		rpt, err = report.ReadJSONFile(cfg.Output.JSON)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				fmt.Println("No existing report found, starting fresh")
				rpt = nil
			} else {
				return fmt.Errorf("load existing report: %w", err)
			}
		} else {
			fmt.Printf("Resuming from existing report with %d units already analyzed\n", len(rpt.Units))
		}
	}

	if rpt == nil {
		rpt = report.NewReport()
		rpt.Metadata.Modules = patterns
		rpt.Metadata.ConfigFiles = configPaths
		rpt.Metadata.InlineConfigs = inlineConfigs
		rpt.Metadata.TotalUnits = len(units)
		rpt.Metadata.GeneratedAt = time.Now()
	}

	// Analyze each unit in order
	ctx := context.Background()
	calleeSummaries := make(map[string]*analyze.SummaryResponse)

	// Rebuild callee summaries from existing report for resumption
	for unitID, unitReport := range rpt.Units {
		calleeSummaries[unitID] = &analyze.SummaryResponse{
			Purpose:    unitReport.Summary.Purpose,
			Behavior:   unitReport.Summary.Behavior,
			Invariants: unitReport.Summary.Invariants,
			Security:   unitReport.Summary.Security,
		}
	}

	skipped := 0
	analyzed := 0
	var lastErr error

	// Track issues found during analysis for live display
	issuesBySeverity := make(map[string]int)
	var currentPhase string

	pipeline.OnProgress(func(event analyze.ProgressEvent) {
		if event.Phase != currentPhase {
			currentPhase = event.Phase
			printProgress("    → %s", currentPhase)
		}
		if event.IssueFound != nil {
			issuesBySeverity[event.IssueFound.Severity]++
			// Show running tally
			if isTTY {
				fmt.Print("\r\033[K    → ", currentPhase, " [")
				first := true
				for _, sev := range []string{"critical", "important", "minor"} {
					if count := issuesBySeverity[sev]; count > 0 {
						if !first {
							fmt.Print(" ")
						}
						fmt.Printf("%s:%d", sev, count)
						first = false
					}
				}
				fmt.Print("]")
			}
		}
	})

	for i, unit := range units {
		// Skip already analyzed units
		if _, exists := rpt.Units[unit.ID]; exists {
			skipped++
			continue
		}

		// Reset per-unit tracking
		issuesBySeverity = make(map[string]int)
		currentPhase = ""

		fmt.Printf("\n[%d/%d] %s\n", i+1, len(units), unit.ID)

		unitReport, err := pipeline.Analyze(ctx, unit, calleeSummaries)
		clearLine()
		if err != nil {
			lastErr = fmt.Errorf("analyze %s: %w", unit.ID, err)
			fmt.Printf("Error: %v\n", lastErr)
			fmt.Println("Saving progress...")
			saveProgress(rpt, cfg, format)
			fmt.Printf("Progress saved. Run with -resume to continue.\n")
			return lastErr
		}

		rpt.Units[unit.ID] = *unitReport
		analyzed++

		// Print unit summary
		if len(unitReport.Issues) == 0 {
			fmt.Println("    ✓ No issues found")
		} else {
			fmt.Printf("    Found %d issue(s)\n", len(unitReport.Issues))
		}

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

		// Save progress periodically (every 10 units)
		if analyzed%10 == 0 {
			saveProgress(rpt, cfg, format)
		}
	}

	if skipped > 0 {
		fmt.Printf("Skipped %d already analyzed units\n", skipped)
	}

	// Write final output
	if err := writeReport(rpt, cfg, format, true); err != nil {
		return err
	}

	// Print summary
	fmt.Printf("\nAnalysis complete: %d issues found\n", rpt.Summary.TotalIssues)
	for sev, count := range rpt.Summary.BySeverity {
		fmt.Printf("  %s: %d\n", sev, count)
	}

	return nil
}

func writeReport(rpt *report.Report, cfg *config.Config, format string, final bool) error {
	if format == "json" || format == "all" {
		if err := report.WriteJSONFile(rpt, cfg.Output.JSON); err != nil {
			return fmt.Errorf("write json: %w", err)
		}
		if final {
			fmt.Printf("Wrote %s\n", cfg.Output.JSON)
		}
	}
	if format == "markdown" || format == "all" {
		if err := markdown.WriteFile(rpt, cfg.Output.Markdown); err != nil {
			return fmt.Errorf("write markdown: %w", err)
		}
		if final {
			fmt.Printf("Wrote %s\n", cfg.Output.Markdown)
		}
	}
	if format == "sarif" || format == "all" {
		if err := sarif.WriteFile(rpt, cfg.Output.SARIF); err != nil {
			return fmt.Errorf("write sarif: %w", err)
		}
		if final {
			fmt.Printf("Wrote %s\n", cfg.Output.SARIF)
		}
	}
	return nil
}

func saveProgress(rpt *report.Report, cfg *config.Config, format string) {
	if err := writeReport(rpt, cfg, format, false); err != nil {
		fmt.Printf("Warning: failed to save progress: %v\n", err)
	}
}
