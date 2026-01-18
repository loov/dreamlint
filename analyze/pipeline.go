// analyze/pipeline.go
package analyze

import (
	"context"
	"encoding/json"
	"fmt"
	"text/template"

	"github.com/loov/reviewmod/cache"
	"github.com/loov/reviewmod/config"
	"github.com/loov/reviewmod/extract"
	"github.com/loov/reviewmod/llm"
	"github.com/loov/reviewmod/report"
)

// Pipeline runs the analysis passes on all units
type Pipeline struct {
	config        *config.Config
	cache         *cache.Cache
	llmClient     llm.Client
	prompts       map[string]*template.Template
	summaries     map[string]*SummaryResponse
	externalFuncs map[string]*extract.ExternalFunc
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

// LoadPrompts loads all prompt templates from config
func (p *Pipeline) LoadPrompts() error {
	for _, pass := range p.config.Analyses {
		if !pass.Enabled {
			continue
		}
		tmpl, err := LoadPrompt(pass.Prompt)
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
	promptCtx := p.buildPromptContext(unit, calleeSummaries)

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
	for _, pass := range p.config.Analyses {
		if !pass.Enabled || pass.Name == "summary" {
			continue
		}

		issues, err := p.runAnalysisPass(ctx, pass, promptCtx)
		if err != nil {
			return nil, fmt.Errorf("%s pass for %s: %w", pass.Name, unit.ID, err)
		}

		for _, issue := range issues {
			// Find the function's position to convert relative line to absolute
			var pos = unit.Functions[0].Position // fallback to first function
			if fn, ok := funcPositions[issue.Function]; ok {
				pos = fn.Position
			}
			pos.Line = pos.Line + issue.Line - 1

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

func (p *Pipeline) buildPromptContext(unit *extract.AnalysisUnit, calleeSummaries map[string]*SummaryResponse) PromptContext {
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
	for _, pass := range p.config.Analyses {
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
