package analyze

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
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
	config          *config.Config
	cache           *cache.Cache
	llmClient       llm.Client
	prompts         map[string]*template.Template
	typeSummaryTmpl *template.Template
	summaries       map[string]*SummaryResponse
	typeSummaries   map[string]*SummaryResponse
	externalFuncs   map[string]*extract.ExternalFunc
	types           map[string]*extract.TypeInfo
	funcs           map[string]*extract.FunctionInfo
	promptsFS       fs.FS
	onProgress      ProgressCallback
	language        string
}

// SetLanguage sets the source language used when rendering prompts.
// When unset, prompts use a neutral "source" placeholder.
func (p *Pipeline) SetLanguage(lang string) {
	p.language = lang
}

// NewPipeline creates a new analysis pipeline.
//
// types and funcs are optional. When types is populated, AnalyzeTypes
// can be called before the unit loop to pre-compute per-type summaries
// that the method prompts will consume.
func NewPipeline(
	cfg *config.Config,
	c *cache.Cache,
	client llm.Client,
	externalFuncs map[string]*extract.ExternalFunc,
	types map[string]*extract.TypeInfo,
	funcs map[string]*extract.FunctionInfo,
) *Pipeline {
	return &Pipeline{
		config:        cfg,
		cache:         c,
		llmClient:     client,
		prompts:       make(map[string]*template.Template),
		summaries:     make(map[string]*SummaryResponse),
		typeSummaries: make(map[string]*SummaryResponse),
		externalFuncs: externalFuncs,
		types:         types,
		funcs:         funcs,
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

// LoadPrompts loads all prompt templates from config, plus the
// built-in type_summary template used by AnalyzeTypes.
func (p *Pipeline) LoadPrompts() error {
	for _, pass := range p.config.Analyse {
		if !pass.Enabled {
			continue
		}
		var tmpl *template.Template
		var err error

		switch {
		case pass.InlinePrompt != "":
			tmpl, err = LoadInlinePrompt(pass.Name, pass.InlinePrompt)
		case p.promptsFS != nil && strings.HasPrefix(pass.Prompt, "builtin:"):
			name := strings.TrimPrefix(pass.Prompt, "builtin:")
			tmpl, err = LoadPromptFromFS(p.promptsFS, name)
		case pass.Prompt != "":
			prompt := pass.Prompt
			if pass.PromptDir != "" && !strings.HasPrefix(prompt, "builtin:") && !filepath.IsAbs(prompt) {
				prompt = filepath.Join(pass.PromptDir, prompt)
			}
			tmpl, err = LoadPrompt(prompt)
		default:
			return fmt.Errorf("pass %s: prompt or inline_prompt must be specified", pass.Name)
		}
		if err != nil {
			return fmt.Errorf("load prompt %s: %w", pass.Name, err)
		}
		p.prompts[pass.Name] = tmpl
	}

	// Built-in type summary prompt. Failure here is non-fatal — if the
	// prompt can't be loaded, AnalyzeTypes will just produce empty
	// summaries and method prompts will fall back to structural context
	// only.
	if p.promptsFS != nil {
		if tmpl, err := LoadPromptFromFS(p.promptsFS, "type_summary"); err == nil {
			p.typeSummaryTmpl = tmpl
		}
	}
	if p.typeSummaryTmpl == nil {
		if tmpl, err := LoadPrompt("builtin:type_summary"); err == nil {
			p.typeSummaryTmpl = tmpl
		}
	}
	return nil
}

// AnalyzeTypes runs a summary pass over every extracted type in the
// project. Results are stored on the pipeline and later rendered into
// method prompts as receiver-type context. Types without methods still
// get summarized — their summaries are harmless and keep the cache
// warm for future runs.
//
// This method is a no-op when no type_summary template was loaded or
// the types map is empty.
func (p *Pipeline) AnalyzeTypes(ctx context.Context) error {
	if p.typeSummaryTmpl == nil || len(p.types) == 0 {
		return nil
	}

	ids := make([]string, 0, len(p.types))
	for id := range p.types {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	llmCfg := p.config.LLM
	for _, pass := range p.config.Analyse {
		if pass.Name == "summary" && pass.LLM != nil {
			llmCfg = *pass.LLM
			break
		}
	}

	for _, id := range ids {
		ti := p.types[id]
		cacheKey := p.typeCacheKey(ti)

		var summary *SummaryResponse
		if p.config.Cache.Enabled && p.cache != nil {
			if data, ok := p.cache.Get(cacheKey); ok {
				if err := json.Unmarshal(data, &summary); err == nil {
					p.typeSummaries[id] = summary
					continue
				}
			}
		}

		p.reportProgress(ProgressEvent{Phase: "type-summary"})

		tctx := p.buildTypePromptContext(ti)
		prompt, err := ExecuteTypePrompt(p.typeSummaryTmpl, tctx)
		if err != nil {
			return fmt.Errorf("type summary prompt for %s: %w", id, err)
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
			return fmt.Errorf("type summary llm for %s: %w", id, err)
		}

		summary, err = ParseSummaryResponse(resp.Content)
		if err != nil {
			return fmt.Errorf("parse type summary for %s: %w", id, err)
		}
		p.typeSummaries[id] = summary

		if p.config.Cache.Enabled && p.cache != nil {
			if data, err := json.Marshal(summary); err == nil {
				p.cache.Set(cacheKey, data)
			}
		}
	}
	return nil
}

// GetTypeSummary returns the summary for a type, or nil if none.
func (p *Pipeline) GetTypeSummary(typeID string) *SummaryResponse {
	return p.typeSummaries[typeID]
}

func (p *Pipeline) buildTypePromptContext(ti *extract.TypeInfo) TypePromptContext {
	lang := p.language
	if lang == "" {
		lang = "source"
	}
	methods := make([]ReceiverMethodContext, 0, len(ti.Methods))
	for _, mID := range ti.Methods {
		fn, ok := p.funcs[mID]
		if !ok {
			continue
		}
		sig, doc := cleanMethodDisplay(fn.Signature, fn.Doc)
		methods = append(methods, ReceiverMethodContext{
			Name:      fn.Name,
			Signature: sig,
			Doc:       doc,
		})
	}
	sig, doc := cleanTypeDisplay(ti.Signature, ti.Doc)
	return TypePromptContext{
		Language:  lang,
		Kind:      ti.Kind,
		Name:      ti.Name,
		Package:   ti.Package,
		Signature: sig,
		Body:      ti.Body,
		Doc:       doc,
		Methods:   methods,
	}
}

func (p *Pipeline) typeCacheKey(ti *extract.TypeInfo) string {
	parts := []string{ti.Body, ti.Doc}
	for _, mID := range ti.Methods {
		if fn, ok := p.funcs[mID]; ok {
			parts = append(parts, fn.Signature)
		}
	}
	return cache.ContentHash(parts...)
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
	lang := p.language
	if lang == "" {
		lang = "source"
	}
	ctx := PromptContext{Language: lang}

	if len(unit.Functions) == 1 {
		fn := unit.Functions[0]
		ctx.Name = fn.Name
		ctx.Package = fn.Package
		ctx.Receiver = fn.Receiver
		ctx.Signature = fn.Signature
		ctx.Body = fn.Body
		ctx.Doc = fn.Doc
	} else {
		// SCC with multiple functions
		for _, fn := range unit.Functions {
			ctx.Functions = append(ctx.Functions, FunctionContext{
				Name:      fn.Name,
				Receiver:  fn.Receiver,
				Signature: fn.Signature,
				Body:      fn.Body,
				Doc:       fn.Doc,
			})
		}
	}

	// Receiver type context: when every function in the unit shares the
	// same receiver type (typical: a single method, or a rare SCC of
	// mutually-recursive methods on the same type), render the type's
	// declaration, LLM summary, and sibling method signatures.
	if rt := p.buildReceiverTypeContext(unit); rt != nil {
		ctx.ReceiverType = rt
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
	for _, calleeID := range unit.External {
		if ext, ok := p.externalFuncs[calleeID]; ok {
			ctx.ExternalFuncs = append(ctx.ExternalFuncs, ExternalFuncContext{
				Package:   ext.Package,
				Name:      ext.Name,
				Signature: ext.Signature,
				Doc:       ext.Doc,
			})
		}
	}

	return ctx
}

func (p *Pipeline) buildReceiverTypeContext(unit *extract.AnalysisUnit) *ReceiverTypeContext {
	if len(unit.Functions) == 0 || len(p.types) == 0 {
		return nil
	}
	typeID := unit.Functions[0].ReceiverType
	if typeID == "" {
		return nil
	}
	for _, fn := range unit.Functions[1:] {
		if fn.ReceiverType != typeID {
			return nil
		}
	}
	ti, ok := p.types[typeID]
	if !ok {
		return nil
	}

	sig, doc := cleanTypeDisplay(ti.Signature, ti.Doc)
	rt := &ReceiverTypeContext{
		Name:      ti.Name,
		Kind:      ti.Kind,
		Signature: sig,
		Body:      ti.Body,
		Doc:       doc,
	}
	if s := p.typeSummaries[typeID]; s != nil {
		rt.Purpose = s.Purpose
		rt.Behavior = s.Behavior
		rt.Invariants = s.Invariants
		rt.Security = s.Security
	}

	currentIDs := make(map[string]bool, len(unit.Functions))
	for _, fn := range unit.Functions {
		currentIDs[fn.ID()] = true
	}
	for _, mID := range ti.Methods {
		if currentIDs[mID] {
			continue
		}
		fn, ok := p.funcs[mID]
		if !ok {
			continue
		}
		sig, doc := cleanMethodDisplay(fn.Signature, fn.Doc)
		rt.Methods = append(rt.Methods, ReceiverMethodContext{
			Name:      fn.Name,
			Signature: sig,
			Doc:       doc,
		})
	}
	return rt
}

// cleanMethodDisplay prepares a SCIP-flavored (Signature, Doc) pair
// for rendering as a single bullet point. Falls back to the first
// code-fence line of Doc when Signature is empty (scip-typescript
// leaves SignatureDocumentation empty and puts the signature inside a
// ```ts fence in Documentation), strips that block from Doc to avoid
// duplicate text, and collapses the remaining Doc to one line.
func cleanMethodDisplay(signature, doc string) (string, string) {
	sig, body := splitSignatureFromDoc(signature, doc)
	return sig, collapseToLine(body)
}

// cleanTypeDisplay is like cleanMethodDisplay but preserves multi-line
// Doc — types often have several sentences of prose worth showing in
// full.
func cleanTypeDisplay(signature, doc string) (string, string) {
	return splitSignatureFromDoc(signature, doc)
}

// splitSignatureFromDoc falls back to the first ``` fence block when
// Signature is empty. Returns trimmed (signature, doc-without-fence).
func splitSignatureFromDoc(signature, doc string) (string, string) {
	sig := strings.TrimSpace(signature)
	body := strings.TrimSpace(doc)
	if sig == "" {
		if s, rest, ok := extractFirstFencedLine(body); ok {
			sig = s
			body = rest
		}
	}
	return sig, body
}

// extractFirstFencedLine pulls the first content line out of the first
// triple-backtick fence in s. Returns (line, remainder, true) on
// success; the remainder is s with the fence removed.
func extractFirstFencedLine(s string) (string, string, bool) {
	lines := strings.Split(s, "\n")
	start := -1
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "```") {
			start = i
			break
		}
	}
	if start < 0 {
		return "", s, false
	}
	end := -1
	for i := start + 1; i < len(lines); i++ {
		if strings.HasPrefix(strings.TrimSpace(lines[i]), "```") {
			end = i
			break
		}
	}
	if end < 0 || end == start+1 {
		return "", s, false
	}
	firstLine := strings.TrimSpace(lines[start+1])
	if firstLine == "" {
		return "", s, false
	}
	remainder := append([]string(nil), lines[:start]...)
	remainder = append(remainder, lines[end+1:]...)
	return firstLine, strings.TrimSpace(strings.Join(remainder, "\n")), true
}

// collapseToLine joins whitespace-separated runs of doc text into a
// single line so it fits in a bullet point.
func collapseToLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
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
		if fn.ReceiverType != "" {
			if s := p.typeSummaries[fn.ReceiverType]; s != nil {
				data, _ := json.Marshal(s)
				parts = append(parts, string(data))
			}
		}
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
