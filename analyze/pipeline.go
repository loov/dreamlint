package analyze

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"strings"
	"text/template"

	"github.com/loov/dreamlint/cache"
	"github.com/loov/dreamlint/config"
	"github.com/loov/dreamlint/extract"
	"github.com/loov/dreamlint/llm"
	"github.com/loov/dreamlint/report"
)

// ProgressCallback is called during analysis to report progress
type ProgressCallback func(event ProgressEvent)

// ProgressEvent represents a progress update during analysis
type ProgressEvent struct {
	Phase      string // "summary" or the analysis pass name
	IssueFound *IssueEvent
}

// IssueEvent is emitted when an issue is found
type IssueEvent struct {
	Category string
	Severity string
}

// Pipeline runs the analysis passes on all units
type Pipeline struct {
	config        *config.Config
	cache         *cache.Cache
	llmClient     llm.Client
	prompts       map[string]*template.Template
	summaries     map[string]*SummaryResponse
	externalFuncs map[string]*extract.ExternalFunc
	promptsFS     fs.FS
	onProgress    ProgressCallback
}

// NewPipeline creates a new analysis pipeline
func NewPipeline(cfg *config.Config, c *cache.Cache, client llm.Client, externalFuncs map[string]*extract.ExternalFunc) *Pipeline {
	return &Pipeline{
		config:        cfg,
		cache:         c,
		llmClient:     client,
		prompts:       make(map[string]*template.Template),
		summaries:     make(map[string]*SummaryResponse),
		externalFuncs: externalFuncs,
	}
}

// SetPromptsFS sets a filesystem to load prompts from.
// When set, builtin: prompts will be loaded from this filesystem instead.
func (p *Pipeline) SetPromptsFS(fsys fs.FS) {
	p.promptsFS = fsys
}

// OnProgress sets a callback for progress events during analysis.
func (p *Pipeline) OnProgress(cb ProgressCallback) {
	p.onProgress = cb
}

func (p *Pipeline) reportProgress(event ProgressEvent) {
	if p.onProgress != nil {
		p.onProgress(event)
	}
}

// LoadPrompts loads all prompt templates from config
func (p *Pipeline) LoadPrompts() error {
	for _, pass := range p.config.Analyse {
		if !pass.Enabled {
			continue
		}
		var tmpl *template.Template
		var err error

		// If promptsFS is set and prompt is builtin, load from that filesystem
		if p.promptsFS != nil && strings.HasPrefix(pass.Prompt, "builtin:") {
			name := strings.TrimPrefix(pass.Prompt, "builtin:")
			tmpl, err = LoadPromptFromFS(p.promptsFS, name)
		} else {
			tmpl, err = LoadPrompt(pass.Prompt)
		}
		if err != nil {
			return fmt.Errorf("load prompt %s: %w", pass.Name, err)
		}
		p.prompts[pass.Name] = tmpl
	}
	return nil
}

// Analyze runs all analysis passes on a single unit
func (p *Pipeline) Analyze(ctx context.Context, unit *extract.AnalysisUnit, calleeSummaries map[string]*SummaryResponse) (*report.UnitReport, error) {
	// Build prompt context
	promptCtx := p.BuildPromptContext(unit, calleeSummaries)

	// Check cache for summary
	cacheKey := p.cacheKey(unit, calleeSummaries)
	var summary *SummaryResponse

	if p.config.Cache.Enabled {
		if data, ok := p.cache.Get(cacheKey); ok {
			if err := json.Unmarshal(data, &summary); err == nil {
				p.summaries[unit.ID] = summary
			}
		}
	}

	// Run summary pass if not cached
	if summary == nil {
		p.reportProgress(ProgressEvent{Phase: "summary"})
		var err error
		summary, err = p.runSummaryPass(ctx, promptCtx)
		if err != nil {
			return nil, fmt.Errorf("summary pass for %s: %w", unit.ID, err)
		}
		p.summaries[unit.ID] = summary

		// Cache the summary
		if p.config.Cache.Enabled {
			if data, err := json.Marshal(summary); err == nil {
				p.cache.Set(cacheKey, data)
			}
		}
	}

	// Build unit report
	unitReport := &report.UnitReport{
		Summary: report.FunctionSummary{
			Purpose:    summary.Purpose,
			Behavior:   summary.Behavior,
			Invariants: summary.Invariants,
			Security:   summary.Security,
		},
	}

	// Add function info
	for _, fn := range unit.Functions {
		unitReport.Functions = append(unitReport.Functions, report.FunctionInfo{
			Package:   fn.Package,
			Name:      fn.Name,
			Receiver:  fn.Receiver,
			Signature: fn.Signature,
			Position:  fn.Position,
		})
	}

	// Add summary to prompt context for analysis passes
	promptCtx.Summary = &SummaryContext{
		Purpose:    summary.Purpose,
		Behavior:   summary.Behavior,
		Invariants: summary.Invariants,
		Security:   summary.Security,
	}

	// Build function position lookup for converting relative line numbers
	funcPositions := make(map[string]extract.FunctionInfo)
	for _, fn := range unit.Functions {
		funcPositions[fn.Name] = *fn
	}

	// Run analysis passes
	for _, pass := range p.config.Analyse {
		if !pass.Enabled || pass.Name == "summary" {
			continue
		}

		p.reportProgress(ProgressEvent{Phase: pass.Name})
		issues, err := p.runAnalysisPass(ctx, pass, promptCtx)
		if err != nil {
			return nil, fmt.Errorf("%s pass for %s: %w", pass.Name, unit.ID, err)
		}

		for _, issue := range issues {
			p.reportProgress(ProgressEvent{
				Phase: pass.Name,
				IssueFound: &IssueEvent{
					Category: pass.Name,
					Severity: issue.Severity,
				},
			})
			// Find the function's position and body to locate the code snippet
			var fn *extract.FunctionInfo
			if f, ok := funcPositions[issue.Function]; ok {
				fn = &f
			} else if len(unit.Functions) > 0 {
				fn = unit.Functions[0]
			}

			pos := fn.Position
			// Find line by matching code snippet, using LLM's line as hint
			if issue.Code != "" && fn != nil {
				// Convert absolute line hint to relative line within function body
				hintLine := 0
				if issue.Line > 0 {
					hintLine = issue.Line - pos.Line + 1
				}
				if line := findLineInBody(fn.Body, issue.Code, hintLine); line > 0 {
					pos.Line = pos.Line + line - 1
				}
			} else if issue.Line > 0 {
				// Fallback to LLM line if no code snippet provided
				pos.Line = pos.Line + issue.Line - 1
			}

			unitReport.Issues = append(unitReport.Issues, report.Issue{
				Position:   pos,
				Severity:   report.Severity(issue.Severity),
				Category:   pass.Name,
				Message:    issue.Message,
				Suggestion: issue.Suggestion,
			})
		}
	}

	return unitReport, nil
}

func (p *Pipeline) BuildPromptContext(unit *extract.AnalysisUnit, calleeSummaries map[string]*SummaryResponse) PromptContext {
	ctx := PromptContext{}

	if len(unit.Functions) == 1 {
		fn := unit.Functions[0]
		ctx.Name = fn.Name
		ctx.Package = fn.Package
		ctx.Receiver = fn.Receiver
		ctx.Signature = fn.Signature
		ctx.Body = fn.Body
		ctx.Godoc = fn.Godoc
	} else {
		// SCC with multiple functions
		for _, fn := range unit.Functions {
			ctx.Functions = append(ctx.Functions, FunctionContext{
				Name:      fn.Name,
				Receiver:  fn.Receiver,
				Signature: fn.Signature,
				Body:      fn.Body,
				Godoc:     fn.Godoc,
			})
		}
	}

	// Add callee summaries for internal callees
	for _, calleeID := range unit.Callees {
		if summary, ok := calleeSummaries[calleeID]; ok {
			ctx.Callees = append(ctx.Callees, CalleeSummary{
				Name:       calleeID,
				Purpose:    summary.Purpose,
				Behavior:   summary.Behavior,
				Invariants: summary.Invariants,
				Security:   summary.Security,
			})
		}
	}

	// Add external function info
	for _, calleeID := range unit.Callees {
		if ext, ok := p.externalFuncs[calleeID]; ok {
			ctx.ExternalFuncs = append(ctx.ExternalFuncs, ExternalFuncContext{
				Package:   ext.Package,
				Name:      ext.Name,
				Signature: ext.Signature,
				Godoc:     ext.Godoc,
			})
		}
	}

	return ctx
}

func (p *Pipeline) runSummaryPass(ctx context.Context, promptCtx PromptContext) (*SummaryResponse, error) {
	tmpl, ok := p.prompts["summary"]
	if !ok {
		return nil, fmt.Errorf("summary prompt not loaded")
	}

	prompt, err := ExecutePrompt(tmpl, promptCtx)
	if err != nil {
		return nil, err
	}

	// Find LLM config for summary pass
	llmCfg := p.config.LLM
	for _, pass := range p.config.Analyse {
		if pass.Name == "summary" && pass.LLM != nil {
			llmCfg = *pass.LLM
			break
		}
	}

	resp, err := p.llmClient.Complete(ctx, llm.Request{
		Messages: []llm.Message{{Role: "user", Content: prompt}},
		Config: llm.ModelConfig{
			Model:       llmCfg.Model,
			MaxTokens:   llmCfg.MaxTokens,
			Temperature: llmCfg.Temperature,
			JSONSchema:  SummarySchema,
		},
	})
	if err != nil {
		return nil, err
	}

	return ParseSummaryResponse(resp.Content)
}

func (p *Pipeline) runAnalysisPass(ctx context.Context, pass config.AnalysisPass, promptCtx PromptContext) ([]IssueResponse, error) {
	tmpl, ok := p.prompts[pass.Name]
	if !ok {
		return nil, fmt.Errorf("prompt %s not loaded", pass.Name)
	}

	prompt, err := ExecutePrompt(tmpl, promptCtx)
	if err != nil {
		return nil, err
	}

	// Use pass-specific LLM config or default
	llmCfg := p.config.LLM
	if pass.LLM != nil {
		llmCfg = *pass.LLM
	}

	resp, err := p.llmClient.Complete(ctx, llm.Request{
		Messages: []llm.Message{{Role: "user", Content: prompt}},
		Config: llm.ModelConfig{
			Model:       llmCfg.Model,
			MaxTokens:   llmCfg.MaxTokens,
			Temperature: llmCfg.Temperature,
			JSONSchema:  IssuesSchema,
		},
	})
	if err != nil {
		return nil, err
	}

	return ParseIssuesResponse(resp.Content)
}

func (p *Pipeline) cacheKey(unit *extract.AnalysisUnit, calleeSummaries map[string]*SummaryResponse) string {
	parts := []string{}
	for _, fn := range unit.Functions {
		parts = append(parts, fn.Body)
	}
	for _, calleeID := range unit.Callees {
		if summary, ok := calleeSummaries[calleeID]; ok {
			data, _ := json.Marshal(summary)
			parts = append(parts, string(data))
		}
	}
	return cache.ContentHash(parts...)
}

// GetSummary returns the summary for a unit
func (p *Pipeline) GetSummary(unitID string) *SummaryResponse {
	return p.summaries[unitID]
}

// findLineInBody finds the 1-based line number where the code snippet appears in the body.
// Returns 0 if not found.
// findLineInBody searches for code in body, preferring matches closest to hintLine.
// hintLine is the line number within the body (1-indexed), or 0 if no hint.
func findLineInBody(body, code string, hintLine int) int {
	code = strings.TrimSpace(code)
	if code == "" {
		return 0
	}

	lines := strings.Split(body, "\n")

	// Collect all matching line numbers
	var matches []int
	for i, line := range lines {
		if strings.Contains(line, code) || strings.Contains(strings.TrimSpace(line), code) {
			matches = append(matches, i+1)
		}
	}

	// Try partial match if no exact matches
	if len(matches) == 0 {
		trimmedCode := strings.TrimSpace(code)
		if len(trimmedCode) > 10 {
			for i, line := range lines {
				if strings.Contains(strings.TrimSpace(line), trimmedCode) {
					matches = append(matches, i+1)
				}
			}
		}
	}

	if len(matches) == 0 {
		// If no matches found, return hint line.
		return hintLine
	}

	// If no hint or single match, return first match
	if hintLine <= 0 || len(matches) == 1 {
		return matches[0]
	}

	// Find match closest to hint
	best := matches[0]
	bestDist := abs(best - hintLine)
	for _, m := range matches[1:] {
		if dist := abs(m - hintLine); dist < bestDist {
			best = m
			bestDist = dist
		}
	}
	return best
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
