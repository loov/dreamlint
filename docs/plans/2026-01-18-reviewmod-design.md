# reviewmod - Staged Go Code Review Harness

A model harness that performs Go code review in stages, working around LLM context limits by analyzing functions in dependency order.

## Problem

Using LLMs to review large codebases hits context limits, causing the model to miss details. This tool breaks the problem into manageable pieces by analyzing functions bottom-up, building summaries that propagate context without exceeding token limits.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Input scope | Entire Go module(s), all functions | No filtering needed |
| Callgraph | CHA analysis | Good interface handling, reasonable speed |
| External deps | Shallow (signature, godoc, invariants, pitfalls) | Provides context without exploding scope |
| Ordering | Tarjan SCC + topological sort | Analyze callees before callers; handle mutual recursion |
| Config format | Cue | Schema validation, defaults, composability |
| Context depth | Immediate callees only | Summaries encapsulate deeper behavior |
| Caching | Disk-based, hash-keyed | Fast re-runs on large codebases |
| Severity levels | Fixed: critical, high, medium, low, info | Consistent filtering across passes |
| Output | JSON canonical + markdown rendered | Programmatic + human readable |
| LLM interface | OpenAI-compatible primary, abstract interface | Broad compatibility, future flexibility |
| Per-pass LLM | Configurable model per analysis pass | Cost optimization, specialized models |

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Extract   │───▶│    Sort     │───▶│   Analyze   │───▶│   Collect   │───▶│   Report    │
│  Callgraph  │    │ Topological │    │  Functions  │    │   Issues    │    │  Generate   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Stage 1: Extract Callgraph

- Load target module(s) using `golang.org/x/tools/go/packages`
- Build callgraph using `golang.org/x/tools/go/callgraph/cha`
- Partition into internal (full source) and external (shallow info)

### Stage 2: Topological Sort

- Compute SCCs using Tarjan's algorithm
- SCCs with multiple functions (mutual recursion) become single analysis units
- Sort in reverse topological order (callees before callers)

### Stage 3: Analyze Functions

For each unit:
1. Check cache (content hash + callee summary hashes)
2. Build context (source, callee summaries, external func info)
3. Run summary pass first (generates Summary for callers to use)
4. Run each enabled analysis pass
5. Collect issues

### Stage 4: Collect Issues

- Aggregate issues from all passes
- Deduplicate by position + message similarity
- Attach severity and category

### Stage 5: Report Generate

- Build JSON report (canonical)
- Render markdown from JSON

## Data Structures

### AnalysisUnit

```go
type AnalysisUnit struct {
    ID        string            // unique identifier (package path + names)
    Functions []*FunctionInfo   // single function, or multiple if SCC
    Callees   []string          // IDs of units this calls
}

type FunctionInfo struct {
    Package   string
    Name      string
    Receiver  string            // empty for functions
    Signature string
    Body      string            // source code
    Godoc     string
    Position  token.Position    // file:line for reporting
}
```

### ExternalFunc

```go
type ExternalFunc struct {
    Package    string
    Name       string
    Signature  string
    Godoc      string
    Invariants []string  // preconditions, postconditions, guarantees
    Pitfalls   []string  // common mistakes when using this function
}
```

### Summary

```go
type Summary struct {
    UnitID      string
    Purpose     string   // what the function does
    Behavior    string   // how it behaves (side effects, return conditions)
    Invariants  []string // preconditions, postconditions, guarantees
    Security    []string // security-relevant properties
    ContentHash string   // for cache invalidation
}
```

### Issue

```go
type Issue struct {
    UnitID     string
    Position   token.Position
    Severity   Severity         // critical, high, medium, low, info
    Category   string           // which analysis pass found it
    Message    string
    Snippet    string           // relevant code
    Suggestion string           // fix suggestion if available
}
```

### Report

```go
type Report struct {
    Metadata   ReportMetadata           `json:"metadata"`
    Units      map[string]UnitReport    `json:"units"`
    Summary    ReportSummary            `json:"summary"`
}

type ReportMetadata struct {
    GeneratedAt   time.Time `json:"generated_at"`
    Modules       []string  `json:"modules"`
    Config        string    `json:"config_file"`
    TotalUnits    int       `json:"total_units"`
    CacheHits     int       `json:"cache_hits"`
}

type UnitReport struct {
    Functions []FunctionInfo `json:"functions"`
    Summary   Summary        `json:"summary"`
    Issues    []Issue        `json:"issues"`
}

type ReportSummary struct {
    TotalIssues   int            `json:"total_issues"`
    BySeverity    map[string]int `json:"by_severity"`
    ByCategory    map[string]int `json:"by_category"`
    CriticalUnits []string       `json:"critical_units"`
}
```

## Configuration (Cue)

### Schema

```cue
package config

#LLMConfig: {
    provider:    "openai" | "anthropic"
    base_url:    string
    model:       string
    api_key?:    string
    max_tokens:  int | *4096
    temperature: float | *0.1
}

#AnalysisPass: {
    name:    string
    prompt:  string
    enabled: bool | *true
    llm?:    #LLMConfig  // override default LLM for this pass
    include_security_properties?: bool
}

#Config: {
    llm:      #LLMConfig
    cache: {
        dir:     string | *".reviewmod/cache"
        enabled: bool | *true
    }
    output: {
        json:     string | *"reviewmod-report.json"
        markdown: string | *"reviewmod-report.md"
    }
    analyses: [...#AnalysisPass]
}
```

### Example Config

```cue
package config

llm: {
    provider:  "openai"
    base_url:  "http://localhost:8080/v1"
    model:     "llama3-70b"
}

cache: dir: ".reviewmod/cache"

analyses: [
    {name: "summary", prompt: "prompts/summary.txt"},
    {name: "security", prompt: "prompts/security.txt", 
     include_security_properties: true,
     llm: {provider: "openai", base_url: "https://api.openai.com/v1", model: "gpt-4o"}},
    {name: "errors", prompt: "prompts/error-handling.txt"},
    {name: "cleanliness", prompt: "prompts/cleanliness.txt",
     llm: {provider: "openai", base_url: "http://localhost:8080/v1", model: "llama3-8b"}},
]
```

## LLM Interface

```go
package llm

type Client interface {
    Complete(ctx context.Context, req Request) (Response, error)
}

type Request struct {
    System   string
    Messages []Message
    Config   ModelConfig
}

type Message struct {
    Role    string // "user" or "assistant"
    Content string
}

type ModelConfig struct {
    Model       string
    MaxTokens   int
    Temperature float64
}

type Response struct {
    Content string
    Usage   Usage
}

type Usage struct {
    PromptTokens     int
    CompletionTokens int
}
```

## Cache Invalidation

```go
func cacheKey(unit AnalysisUnit, calleeSummaries map[string]Summary) string {
    h := sha256.New()
    for _, fn := range unit.Functions {
        h.Write([]byte(fn.Body))
    }
    for _, calleeID := range unit.Callees {
        h.Write([]byte(calleeSummaries[calleeID].ContentHash))
    }
    return hex.EncodeToString(h.Sum(nil))
}
```

A unit's cache invalidates if its code changes OR any callee's summary changes.

## Package Structure

```
reviewmod/
├── go.mod
├── main.go                     # CLI entry point
├── config/
│   ├── schema.cue              # Cue schema definitions
│   └── config.go               # Load & validate config
├── extract/
│   ├── extract.go              # Load packages, build callgraph
│   ├── external.go             # Extract ExternalFunc from dependencies
│   └── scc.go                  # Tarjan's algorithm, topological sort
├── analyze/
│   ├── pipeline.go             # Main analysis loop
│   ├── context.go              # Build PromptContext for each unit
│   ├── prompt.go               # Load & execute prompt templates
│   └── parse.go                # Parse LLM JSON responses
├── llm/
│   ├── client.go               # LLM interface
│   └── openai.go               # OpenAI-compatible implementation
├── cache/
│   └── cache.go                # Disk cache
├── report/
│   ├── report.go               # Build Report struct
│   ├── json.go                 # JSON output
│   └── markdown.go             # Markdown rendering
├── prompts/                    # Default prompt templates
│   ├── summary.txt
│   ├── security.txt
│   ├── errors.txt
│   └── cleanliness.txt
└── reviewmod.cue               # Example config
```

## CLI Usage

```bash
# Review current module
reviewmod ./...

# Review specific packages
reviewmod ./pkg/auth ./pkg/api

# Custom config
reviewmod -config myconfig.cue ./...

# JSON only
reviewmod -format json ./...
```

## LLM Response Formats

### Summary Pass

```json
{
  "purpose": "Fetches user by ID from database",
  "behavior": "Returns nil, ErrNotFound if user doesn't exist. Panics if db is nil.",
  "invariants": ["db must not be nil", "returns valid User or error, never both nil"],
  "security": ["input id is used in SQL query - must be parameterized"]
}
```

### Analysis Pass

```json
{
  "issues": [
    {
      "line": 42,
      "severity": "high",
      "message": "SQL query built with string concatenation",
      "suggestion": "Use parameterized query with db.Query(sql, args...)"
    }
  ]
}
```
