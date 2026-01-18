# reviewmod Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a staged Go code review harness that analyzes functions in dependency order using LLMs.

**Architecture:** Extract callgraph via CHA, topologically sort SCCs, analyze each unit with configurable LLM passes, cache summaries, output JSON+markdown reports.

**Tech Stack:** Go 1.25, golang.org/x/tools (packages, callgraph), cuelang.org/go/cue, OpenAI-compatible API.

---

## Task 1: Project Setup & Core Types

**Files:**
- Modify: `go.mod`
- Create: `types.go`

**Step 1: Add dependencies to go.mod**

```bash
go get golang.org/x/tools@latest
go get cuelang.org/go/cue@latest
```

**Step 2: Run go mod tidy**

```bash
go mod tidy
```

**Step 3: Create types.go with core data structures**

```go
// types.go
package main

import "go/token"

// Severity levels for issues
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
	SeverityInfo     Severity = "info"
)

// FunctionInfo holds information about a single function
type FunctionInfo struct {
	Package   string         `json:"package"`
	Name      string         `json:"name"`
	Receiver  string         `json:"receiver,omitempty"`
	Signature string         `json:"signature"`
	Body      string         `json:"body"`
	Godoc     string         `json:"godoc,omitempty"`
	Position  token.Position `json:"position"`
}

// AnalysisUnit is the atomic unit of analysis (single func or SCC)
type AnalysisUnit struct {
	ID        string          `json:"id"`
	Functions []*FunctionInfo `json:"functions"`
	Callees   []string        `json:"callees"`
}

// ExternalFunc holds shallow info about external dependencies
type ExternalFunc struct {
	Package    string   `json:"package"`
	Name       string   `json:"name"`
	Signature  string   `json:"signature"`
	Godoc      string   `json:"godoc,omitempty"`
	Invariants []string `json:"invariants,omitempty"`
	Pitfalls   []string `json:"pitfalls,omitempty"`
}

// Summary describes a function's behavior for callers
type Summary struct {
	UnitID      string   `json:"unit_id"`
	Purpose     string   `json:"purpose"`
	Behavior    string   `json:"behavior"`
	Invariants  []string `json:"invariants,omitempty"`
	Security    []string `json:"security,omitempty"`
	ContentHash string   `json:"content_hash"`
}

// Issue represents a problem found during analysis
type Issue struct {
	UnitID     string         `json:"unit_id"`
	Position   token.Position `json:"position"`
	Severity   Severity       `json:"severity"`
	Category   string         `json:"category"`
	Message    string         `json:"message"`
	Snippet    string         `json:"snippet,omitempty"`
	Suggestion string         `json:"suggestion,omitempty"`
}
```

**Step 4: Verify it compiles**

```bash
go build ./...
```
Expected: No errors

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add core type definitions"
```

---

## Task 2: LLM Client Interface & OpenAI Implementation

**Files:**
- Create: `llm/client.go`
- Create: `llm/openai.go`
- Create: `llm/openai_test.go`

**Step 1: Create llm/client.go with interface**

```go
// llm/client.go
package llm

import "context"

// Client is the interface for LLM backends
type Client interface {
	Complete(ctx context.Context, req Request) (Response, error)
}

// Request holds the input for an LLM completion
type Request struct {
	System   string
	Messages []Message
	Config   ModelConfig
}

// Message is a single message in the conversation
type Message struct {
	Role    string // "user" or "assistant"
	Content string
}

// ModelConfig holds model-specific settings
type ModelConfig struct {
	Model       string
	MaxTokens   int
	Temperature float64
}

// Response holds the LLM output
type Response struct {
	Content string
	Usage   Usage
}

// Usage tracks token consumption
type Usage struct {
	PromptTokens     int
	CompletionTokens int
}
```

**Step 2: Verify it compiles**

```bash
go build ./...
```

**Step 3: Create llm/openai.go**

```go
// llm/openai.go
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OpenAIClient implements Client for OpenAI-compatible APIs
type OpenAIClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewOpenAIClient creates a new OpenAI-compatible client
func NewOpenAIClient(baseURL, apiKey string) *OpenAIClient {
	return &OpenAIClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

type openAIRequest struct {
	Model       string          `json:"model"`
	Messages    []openAIMessage `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature float64         `json:"temperature,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Complete sends a completion request to the OpenAI-compatible API
func (c *OpenAIClient) Complete(ctx context.Context, req Request) (Response, error) {
	messages := make([]openAIMessage, 0, len(req.Messages)+1)

	if req.System != "" {
		messages = append(messages, openAIMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	for _, m := range req.Messages {
		messages = append(messages, openAIMessage{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	oaiReq := openAIRequest{
		Model:       req.Config.Model,
		Messages:    messages,
		MaxTokens:   req.Config.MaxTokens,
		Temperature: req.Config.Temperature,
	}

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return Response{}, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Response{}, fmt.Errorf("create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return Response{}, fmt.Errorf("read response: %w", err)
	}

	var oaiResp openAIResponse
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		return Response{}, fmt.Errorf("unmarshal response: %w", err)
	}

	if oaiResp.Error != nil {
		return Response{}, fmt.Errorf("api error: %s", oaiResp.Error.Message)
	}

	if len(oaiResp.Choices) == 0 {
		return Response{}, fmt.Errorf("no choices in response")
	}

	return Response{
		Content: oaiResp.Choices[0].Message.Content,
		Usage: Usage{
			PromptTokens:     oaiResp.Usage.PromptTokens,
			CompletionTokens: oaiResp.Usage.CompletionTokens,
		},
	}, nil
}
```

**Step 4: Verify it compiles**

```bash
go build ./...
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LLM client interface and OpenAI implementation"
```

---

## Task 3: Tarjan's SCC Algorithm

**Files:**
- Create: `extract/scc.go`
- Create: `extract/scc_test.go`

**Step 1: Write the failing test**

```go
// extract/scc_test.go
package extract

import (
	"reflect"
	"testing"
)

func TestTarjanSCC_Simple(t *testing.T) {
	// A -> B -> C (no cycles)
	graph := map[string][]string{
		"A": {"B"},
		"B": {"C"},
		"C": {},
	}

	sccs := TarjanSCC(graph)

	// Each node is its own SCC, order: C, B, A (reverse topo)
	expected := [][]string{{"C"}, {"B"}, {"A"}}
	if !reflect.DeepEqual(sccs, expected) {
		t.Errorf("got %v, want %v", sccs, expected)
	}
}

func TestTarjanSCC_Cycle(t *testing.T) {
	// A -> B -> C -> A (single SCC)
	graph := map[string][]string{
		"A": {"B"},
		"B": {"C"},
		"C": {"A"},
	}

	sccs := TarjanSCC(graph)

	if len(sccs) != 1 {
		t.Fatalf("got %d SCCs, want 1", len(sccs))
	}
	if len(sccs[0]) != 3 {
		t.Errorf("got SCC size %d, want 3", len(sccs[0]))
	}
}

func TestTarjanSCC_Mixed(t *testing.T) {
	// D -> A -> B -> C -> B (B-C cycle), A also -> C
	graph := map[string][]string{
		"D": {"A"},
		"A": {"B", "C"},
		"B": {"C"},
		"C": {"B"},
	}

	sccs := TarjanSCC(graph)

	// Should have: {B,C} as one SCC, then A, then D
	if len(sccs) != 3 {
		t.Fatalf("got %d SCCs, want 3", len(sccs))
	}

	// First SCC should be the cycle {B, C}
	if len(sccs[0]) != 2 {
		t.Errorf("first SCC size %d, want 2", len(sccs[0]))
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./extract/... -v
```
Expected: FAIL (TarjanSCC not defined)

**Step 3: Implement TarjanSCC**

```go
// extract/scc.go
package extract

import "sort"

// TarjanSCC computes strongly connected components using Tarjan's algorithm.
// Returns SCCs in reverse topological order (leaves first).
func TarjanSCC(graph map[string][]string) [][]string {
	var (
		index    = 0
		stack    = []string{}
		onStack  = map[string]bool{}
		indices  = map[string]int{}
		lowlinks = map[string]int{}
		sccs     = [][]string{}
	)

	var strongconnect func(v string)
	strongconnect = func(v string) {
		indices[v] = index
		lowlinks[v] = index
		index++
		stack = append(stack, v)
		onStack[v] = true

		for _, w := range graph[v] {
			if _, ok := indices[w]; !ok {
				strongconnect(w)
				if lowlinks[w] < lowlinks[v] {
					lowlinks[v] = lowlinks[w]
				}
			} else if onStack[w] {
				if indices[w] < lowlinks[v] {
					lowlinks[v] = indices[w]
				}
			}
		}

		if lowlinks[v] == indices[v] {
			scc := []string{}
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[w] = false
				scc = append(scc, w)
				if w == v {
					break
				}
			}
			sort.Strings(scc) // deterministic order within SCC
			sccs = append(sccs, scc)
		}
	}

	// Get all nodes and sort for deterministic order
	nodes := make([]string, 0, len(graph))
	for v := range graph {
		nodes = append(nodes, v)
	}
	sort.Strings(nodes)

	for _, v := range nodes {
		if _, ok := indices[v]; !ok {
			strongconnect(v)
		}
	}

	return sccs
}
```

**Step 4: Run tests**

```bash
go test ./extract/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tarjan's SCC algorithm"
```

---

## Task 4: Package Loading & Function Extraction

**Files:**
- Create: `extract/extract.go`
- Create: `extract/extract_test.go`

**Step 1: Write the failing test**

```go
// extract/extract_test.go
package extract

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExtractFunctions(t *testing.T) {
	// Create a temp directory with a simple Go file
	dir := t.TempDir()

	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

// Hello returns a greeting.
func Hello(name string) string {
	return "Hello, " + name
}

func helper() {}
`
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	funcs, err := ExtractFunctions(dir, "./...")
	if err != nil {
		t.Fatalf("ExtractFunctions: %v", err)
	}

	if len(funcs) != 2 {
		t.Fatalf("got %d functions, want 2", len(funcs))
	}

	// Check Hello function
	var hello *FunctionInfo
	for _, f := range funcs {
		if f.Name == "Hello" {
			hello = f
			break
		}
	}

	if hello == nil {
		t.Fatal("Hello function not found")
	}

	if hello.Signature == "" {
		t.Error("Hello signature is empty")
	}

	if hello.Godoc == "" {
		t.Error("Hello godoc is empty")
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./extract/... -v
```
Expected: FAIL (ExtractFunctions not defined)

**Step 3: Create extract/extract.go**

```go
// extract/extract.go
package extract

import (
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"strings"

	"golang.org/x/tools/go/packages"
)

// FunctionInfo holds information about a single function
type FunctionInfo struct {
	Package   string
	Name      string
	Receiver  string
	Signature string
	Body      string
	Godoc     string
	Position  token.Position
}

// ExtractFunctions loads packages and extracts all function information
func ExtractFunctions(dir string, patterns ...string) ([]*FunctionInfo, error) {
	cfg := &packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedSyntax |
			packages.NeedTypes |
			packages.NeedTypesInfo,
		Dir: dir,
	}

	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, fmt.Errorf("load packages: %w", err)
	}

	var funcs []*FunctionInfo

	for _, pkg := range pkgs {
		if len(pkg.Errors) > 0 {
			return nil, fmt.Errorf("package %s has errors: %v", pkg.PkgPath, pkg.Errors)
		}

		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok {
					continue
				}

				info := &FunctionInfo{
					Package:  pkg.PkgPath,
					Name:     fn.Name.Name,
					Position: pkg.Fset.Position(fn.Pos()),
				}

				// Extract receiver
				if fn.Recv != nil && len(fn.Recv.List) > 0 {
					var buf strings.Builder
					printer.Fprint(&buf, pkg.Fset, fn.Recv.List[0].Type)
					info.Receiver = buf.String()
				}

				// Extract signature
				info.Signature = formatSignature(pkg.Fset, fn)

				// Extract body
				if fn.Body != nil {
					var buf strings.Builder
					printer.Fprint(&buf, pkg.Fset, fn.Body)
					info.Body = buf.String()
				}

				// Extract godoc
				if fn.Doc != nil {
					info.Godoc = fn.Doc.Text()
				}

				funcs = append(funcs, info)
			}
		}
	}

	return funcs, nil
}

func formatSignature(fset *token.FileSet, fn *ast.FuncDecl) string {
	var buf strings.Builder
	buf.WriteString("func ")

	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		buf.WriteString("(")
		printer.Fprint(&buf, fset, fn.Recv.List[0].Type)
		buf.WriteString(") ")
	}

	buf.WriteString(fn.Name.Name)
	printer.Fprint(&buf, fset, fn.Type)

	return buf.String()
}
```

**Step 4: Run tests**

```bash
go test ./extract/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add package loading and function extraction"
```

---

## Task 5: Callgraph Building with CHA

**Files:**
- Modify: `extract/extract.go`
- Create: `extract/callgraph.go`
- Create: `extract/callgraph_test.go`

**Step 1: Write the failing test**

```go
// extract/callgraph_test.go
package extract

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildCallgraph(t *testing.T) {
	dir := t.TempDir()

	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

func A() { B() }
func B() { C() }
func C() {}
`
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	graph, err := BuildCallgraph(dir, "./...")
	if err != nil {
		t.Fatalf("BuildCallgraph: %v", err)
	}

	// A calls B
	if !contains(graph["testpkg.A"], "testpkg.B") {
		t.Errorf("A should call B, got %v", graph["testpkg.A"])
	}

	// B calls C
	if !contains(graph["testpkg.B"], "testpkg.C") {
		t.Errorf("B should call C, got %v", graph["testpkg.B"])
	}

	// C calls nothing
	if len(graph["testpkg.C"]) != 0 {
		t.Errorf("C should call nothing, got %v", graph["testpkg.C"])
	}
}

func contains(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./extract/... -v -run TestBuildCallgraph
```
Expected: FAIL (BuildCallgraph not defined)

**Step 3: Create extract/callgraph.go**

```go
// extract/callgraph.go
package extract

import (
	"fmt"

	"golang.org/x/tools/go/callgraph/cha"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

// BuildCallgraph builds a callgraph using CHA analysis
// Returns a map from function ID to list of callee IDs
func BuildCallgraph(dir string, patterns ...string) (map[string][]string, error) {
	cfg := &packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedSyntax |
			packages.NeedTypes |
			packages.NeedTypesInfo |
			packages.NeedImports |
			packages.NeedDeps,
		Dir: dir,
	}

	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, fmt.Errorf("load packages: %w", err)
	}

	for _, pkg := range pkgs {
		if len(pkg.Errors) > 0 {
			return nil, fmt.Errorf("package %s has errors: %v", pkg.PkgPath, pkg.Errors)
		}
	}

	// Build SSA
	prog, _ := ssautil.AllPackages(pkgs, ssa.SanityCheckFunctions)
	prog.Build()

	// Build callgraph using CHA
	cg := cha.CallGraph(prog)

	// Convert to our format
	graph := make(map[string][]string)

	for fn, node := range cg.Nodes {
		if fn == nil {
			continue
		}

		callerID := funcID(fn)
		if callerID == "" {
			continue
		}

		// Initialize entry even if no callees
		if _, ok := graph[callerID]; !ok {
			graph[callerID] = []string{}
		}

		for _, edge := range node.Out {
			if edge.Callee.Func == nil {
				continue
			}

			calleeID := funcID(edge.Callee.Func)
			if calleeID == "" {
				continue
			}

			// Avoid duplicates
			if !contains(graph[callerID], calleeID) {
				graph[callerID] = append(graph[callerID], calleeID)
			}
		}
	}

	return graph, nil
}

func funcID(fn *ssa.Function) string {
	if fn.Pkg == nil {
		return ""
	}

	pkg := fn.Pkg.Pkg.Path()
	name := fn.Name()

	// Handle methods
	if recv := fn.Signature.Recv(); recv != nil {
		return fmt.Sprintf("%s.(%s).%s", pkg, recv.Type().String(), name)
	}

	return fmt.Sprintf("%s.%s", pkg, name)
}
```

**Step 4: Run tests**

```bash
go test ./extract/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add CHA callgraph building"
```

---

## Task 6: Build Analysis Units from Callgraph

**Files:**
- Create: `extract/units.go`
- Create: `extract/units_test.go`

**Step 1: Write the failing test**

```go
// extract/units_test.go
package extract

import (
	"testing"
)

func TestBuildAnalysisUnits(t *testing.T) {
	funcs := []*FunctionInfo{
		{Package: "pkg", Name: "A"},
		{Package: "pkg", Name: "B"},
		{Package: "pkg", Name: "C"},
	}

	graph := map[string][]string{
		"pkg.A": {"pkg.B"},
		"pkg.B": {"pkg.C"},
		"pkg.C": {},
	}

	units := BuildAnalysisUnits(funcs, graph)

	// Should have 3 units in topological order: C, B, A
	if len(units) != 3 {
		t.Fatalf("got %d units, want 3", len(units))
	}

	// First unit should be C (no dependencies)
	if units[0].ID != "pkg.C" {
		t.Errorf("first unit should be C, got %s", units[0].ID)
	}

	// Last unit should be A
	if units[2].ID != "pkg.A" {
		t.Errorf("last unit should be A, got %s", units[2].ID)
	}

	// A should have B as callee
	if len(units[2].Callees) != 1 || units[2].Callees[0] != "pkg.B" {
		t.Errorf("A callees should be [pkg.B], got %v", units[2].Callees)
	}
}

func TestBuildAnalysisUnits_SCC(t *testing.T) {
	funcs := []*FunctionInfo{
		{Package: "pkg", Name: "A"},
		{Package: "pkg", Name: "B"},
		{Package: "pkg", Name: "C"},
	}

	// B and C form a cycle
	graph := map[string][]string{
		"pkg.A": {"pkg.B"},
		"pkg.B": {"pkg.C"},
		"pkg.C": {"pkg.B"},
	}

	units := BuildAnalysisUnits(funcs, graph)

	// Should have 2 units: {B,C} SCC, then A
	if len(units) != 2 {
		t.Fatalf("got %d units, want 2", len(units))
	}

	// First unit should be the SCC with B and C
	if len(units[0].Functions) != 2 {
		t.Errorf("first unit should have 2 functions, got %d", len(units[0].Functions))
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./extract/... -v -run TestBuildAnalysisUnits
```
Expected: FAIL (BuildAnalysisUnits not defined)

**Step 3: Create extract/units.go**

```go
// extract/units.go
package extract

import (
	"sort"
	"strings"
)

// AnalysisUnit is the atomic unit of analysis
type AnalysisUnit struct {
	ID        string
	Functions []*FunctionInfo
	Callees   []string
}

// BuildAnalysisUnits creates analysis units from functions and callgraph
// Units are returned in topological order (callees before callers)
func BuildAnalysisUnits(funcs []*FunctionInfo, graph map[string][]string) []*AnalysisUnit {
	// Build function lookup
	funcMap := make(map[string]*FunctionInfo)
	for _, f := range funcs {
		id := f.Package + "." + f.Name
		if f.Receiver != "" {
			id = f.Package + ".(" + f.Receiver + ")." + f.Name
		}
		funcMap[id] = f
	}

	// Filter graph to only include internal functions
	internalGraph := make(map[string][]string)
	for caller, callees := range graph {
		if _, ok := funcMap[caller]; !ok {
			continue
		}
		internalCallees := []string{}
		for _, callee := range callees {
			if _, ok := funcMap[callee]; ok {
				internalCallees = append(internalCallees, callee)
			}
		}
		internalGraph[caller] = internalCallees
	}

	// Add nodes with no outgoing edges
	for id := range funcMap {
		if _, ok := internalGraph[id]; !ok {
			internalGraph[id] = []string{}
		}
	}

	// Compute SCCs
	sccs := TarjanSCC(internalGraph)

	// Build units from SCCs
	units := make([]*AnalysisUnit, 0, len(sccs))
	sccMap := make(map[string]int) // function ID -> SCC index

	for i, scc := range sccs {
		for _, id := range scc {
			sccMap[id] = i
		}
	}

	for i, scc := range sccs {
		unit := &AnalysisUnit{
			Functions: make([]*FunctionInfo, 0, len(scc)),
			Callees:   []string{},
		}

		// Build ID from sorted function names
		sort.Strings(scc)
		unit.ID = strings.Join(scc, "+")

		// Collect functions
		for _, id := range scc {
			if f, ok := funcMap[id]; ok {
				unit.Functions = append(unit.Functions, f)
			}
		}

		// Collect external callees (callees in different SCCs)
		seenCallees := make(map[string]bool)
		for _, id := range scc {
			for _, callee := range internalGraph[id] {
				calleeUnit := sccMap[callee]
				if calleeUnit != i && !seenCallees[callee] {
					unit.Callees = append(unit.Callees, sccs[calleeUnit][0])
					seenCallees[callee] = true
				}
			}
		}

		// For single-function units, use simpler ID
		if len(unit.Functions) == 1 {
			f := unit.Functions[0]
			unit.ID = f.Package + "." + f.Name
			if f.Receiver != "" {
				unit.ID = f.Package + ".(" + f.Receiver + ")." + f.Name
			}
		}

		units = append(units, unit)
	}

	return units
}
```

**Step 4: Run tests**

```bash
go test ./extract/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add analysis unit building from callgraph"
```

---

## Task 7: Cue Configuration Loading

**Files:**
- Create: `config/schema.cue`
- Create: `config/config.go`
- Create: `config/config_test.go`

**Step 1: Create config/schema.cue**

```cue
// config/schema.cue
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
	llm?:    #LLMConfig
	include_security_properties?: bool
}

#Config: {
	llm: #LLMConfig
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

**Step 2: Write the failing test**

```go
// config/config_test.go
package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()

	cueFile := `
llm: {
	provider: "openai"
	base_url: "http://localhost:8080/v1"
	model: "llama3"
}

analyses: [
	{name: "summary", prompt: "prompts/summary.txt"},
	{name: "security", prompt: "prompts/security.txt"},
]
`
	if err := os.WriteFile(filepath.Join(dir, "reviewmod.cue"), []byte(cueFile), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(filepath.Join(dir, "reviewmod.cue"))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if cfg.LLM.Provider != "openai" {
		t.Errorf("provider = %s, want openai", cfg.LLM.Provider)
	}

	if cfg.LLM.Model != "llama3" {
		t.Errorf("model = %s, want llama3", cfg.LLM.Model)
	}

	// Check defaults
	if cfg.Cache.Dir != ".reviewmod/cache" {
		t.Errorf("cache.dir = %s, want .reviewmod/cache", cfg.Cache.Dir)
	}

	if len(cfg.Analyses) != 2 {
		t.Errorf("analyses count = %d, want 2", len(cfg.Analyses))
	}
}
```

**Step 3: Run test to verify it fails**

```bash
go test ./config/... -v
```
Expected: FAIL (LoadConfig not defined)

**Step 4: Create config/config.go**

```go
// config/config.go
package config

import (
	_ "embed"
	"fmt"
	"os"

	"cuelang.org/go/cue"
	"cuelang.org/go/cue/cuecontext"
)

//go:embed schema.cue
var schemaCue string

// Config is the main configuration structure
type Config struct {
	LLM      LLMConfig      `json:"llm"`
	Cache    CacheConfig    `json:"cache"`
	Output   OutputConfig   `json:"output"`
	Analyses []AnalysisPass `json:"analyses"`
}

// LLMConfig holds LLM connection settings
type LLMConfig struct {
	Provider    string  `json:"provider"`
	BaseURL     string  `json:"base_url"`
	Model       string  `json:"model"`
	APIKey      string  `json:"api_key,omitempty"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

// CacheConfig holds cache settings
type CacheConfig struct {
	Dir     string `json:"dir"`
	Enabled bool   `json:"enabled"`
}

// OutputConfig holds output settings
type OutputConfig struct {
	JSON     string `json:"json"`
	Markdown string `json:"markdown"`
}

// AnalysisPass defines a single analysis pass
type AnalysisPass struct {
	Name                      string     `json:"name"`
	Prompt                    string     `json:"prompt"`
	Enabled                   bool       `json:"enabled"`
	LLM                       *LLMConfig `json:"llm,omitempty"`
	IncludeSecurityProperties bool       `json:"include_security_properties,omitempty"`
}

// LoadConfig loads and validates a Cue configuration file
func LoadConfig(path string) (*Config, error) {
	ctx := cuecontext.New()

	// Load schema
	schemaVal := ctx.CompileString(schemaCue)
	if schemaVal.Err() != nil {
		return nil, fmt.Errorf("compile schema: %w", schemaVal.Err())
	}

	// Load user config
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	userVal := ctx.CompileBytes(data)
	if userVal.Err() != nil {
		return nil, fmt.Errorf("compile config: %w", userVal.Err())
	}

	// Unify with schema to get defaults and validation
	schema := schemaVal.LookupPath(cue.ParsePath("#Config"))
	unified := schema.Unify(userVal)
	if unified.Err() != nil {
		return nil, fmt.Errorf("validate config: %w", unified.Err())
	}

	// Decode into Go struct
	var cfg Config
	if err := unified.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("decode config: %w", err)
	}

	return &cfg, nil
}
```

**Step 5: Run tests**

```bash
go test ./config/... -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add Cue configuration loading"
```

---

## Task 8: Disk Cache Implementation

**Files:**
- Create: `cache/cache.go`
- Create: `cache/cache_test.go`

**Step 1: Write the failing test**

```go
// cache/cache_test.go
package cache

import (
	"testing"
)

func TestCache_GetSet(t *testing.T) {
	dir := t.TempDir()
	c := New(dir)

	key := "test-key-123"
	data := []byte(`{"purpose": "test function"}`)

	// Should miss initially
	if _, ok := c.Get(key); ok {
		t.Error("expected cache miss")
	}

	// Set and get
	if err := c.Set(key, data); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, ok := c.Get(key)
	if !ok {
		t.Fatal("expected cache hit")
	}

	if string(got) != string(data) {
		t.Errorf("got %s, want %s", got, data)
	}
}

func TestCache_Persistence(t *testing.T) {
	dir := t.TempDir()

	key := "persist-key"
	data := []byte("persistent data")

	// Write with first cache instance
	c1 := New(dir)
	if err := c1.Set(key, data); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// Read with second cache instance
	c2 := New(dir)
	got, ok := c2.Get(key)
	if !ok {
		t.Fatal("expected cache hit after reload")
	}

	if string(got) != string(data) {
		t.Errorf("got %s, want %s", got, data)
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./cache/... -v
```
Expected: FAIL (New not defined)

**Step 3: Create cache/cache.go**

```go
// cache/cache.go
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
)

// Cache provides disk-based caching for summaries
type Cache struct {
	dir string
}

// New creates a new cache with the given directory
func New(dir string) *Cache {
	return &Cache{dir: dir}
}

// Get retrieves data from the cache by key
func (c *Cache) Get(key string) ([]byte, bool) {
	path := c.path(key)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	return data, true
}

// Set stores data in the cache
func (c *Cache) Set(key string, data []byte) error {
	if err := os.MkdirAll(c.dir, 0755); err != nil {
		return err
	}

	path := c.path(key)
	return os.WriteFile(path, data, 0644)
}

// Delete removes an entry from the cache
func (c *Cache) Delete(key string) error {
	path := c.path(key)
	err := os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// path returns the file path for a cache key
func (c *Cache) path(key string) string {
	// Hash the key to avoid filesystem issues with special characters
	h := sha256.Sum256([]byte(key))
	name := hex.EncodeToString(h[:])
	return filepath.Join(c.dir, name)
}

// ContentHash computes a hash of the given content
func ContentHash(content ...string) string {
	h := sha256.New()
	for _, c := range content {
		h.Write([]byte(c))
	}
	return hex.EncodeToString(h.Sum(nil))
}
```

**Step 4: Run tests**

```bash
go test ./cache/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add disk cache implementation"
```

---

## Task 9: Report Data Structures & JSON Output

**Files:**
- Create: `report/report.go`
- Create: `report/json.go`
- Create: `report/json_test.go`

**Step 1: Create report/report.go**

```go
// report/report.go
package report

import (
	"go/token"
	"time"
)

// Severity levels
type Severity string

const (
	SeverityCritical Severity = "critical"
	SeverityHigh     Severity = "high"
	SeverityMedium   Severity = "medium"
	SeverityLow      Severity = "low"
	SeverityInfo     Severity = "info"
)

// Report is the complete analysis report
type Report struct {
	Metadata Metadata              `json:"metadata"`
	Units    map[string]UnitReport `json:"units"`
	Summary  Summary               `json:"summary"`
}

// Metadata holds report metadata
type Metadata struct {
	GeneratedAt time.Time `json:"generated_at"`
	Modules     []string  `json:"modules"`
	ConfigFile  string    `json:"config_file"`
	TotalUnits  int       `json:"total_units"`
	CacheHits   int       `json:"cache_hits"`
}

// UnitReport holds analysis results for a single unit
type UnitReport struct {
	Functions []FunctionInfo  `json:"functions"`
	Summary   FunctionSummary `json:"summary"`
	Issues    []Issue         `json:"issues"`
}

// FunctionInfo holds function metadata
type FunctionInfo struct {
	Package   string         `json:"package"`
	Name      string         `json:"name"`
	Receiver  string         `json:"receiver,omitempty"`
	Signature string         `json:"signature"`
	Position  token.Position `json:"position"`
}

// FunctionSummary describes function behavior
type FunctionSummary struct {
	Purpose    string   `json:"purpose"`
	Behavior   string   `json:"behavior"`
	Invariants []string `json:"invariants,omitempty"`
	Security   []string `json:"security,omitempty"`
}

// Issue represents a found problem
type Issue struct {
	Position   token.Position `json:"position"`
	Severity   Severity       `json:"severity"`
	Category   string         `json:"category"`
	Message    string         `json:"message"`
	Snippet    string         `json:"snippet,omitempty"`
	Suggestion string         `json:"suggestion,omitempty"`
}

// Summary aggregates issue counts
type Summary struct {
	TotalIssues   int            `json:"total_issues"`
	BySeverity    map[string]int `json:"by_severity"`
	ByCategory    map[string]int `json:"by_category"`
	CriticalUnits []string       `json:"critical_units"`
}

// NewReport creates a new empty report
func NewReport() *Report {
	return &Report{
		Metadata: Metadata{
			GeneratedAt: time.Now(),
		},
		Units: make(map[string]UnitReport),
		Summary: Summary{
			BySeverity: make(map[string]int),
			ByCategory: make(map[string]int),
		},
	}
}

// AddIssue adds an issue to a unit and updates summary
func (r *Report) AddIssue(unitID string, issue Issue) {
	unit := r.Units[unitID]
	unit.Issues = append(unit.Issues, issue)
	r.Units[unitID] = unit

	r.Summary.TotalIssues++
	r.Summary.BySeverity[string(issue.Severity)]++
	r.Summary.ByCategory[issue.Category]++

	if issue.Severity == SeverityCritical {
		found := false
		for _, u := range r.Summary.CriticalUnits {
			if u == unitID {
				found = true
				break
			}
		}
		if !found {
			r.Summary.CriticalUnits = append(r.Summary.CriticalUnits, unitID)
		}
	}
}
```

**Step 2: Write the failing test**

```go
// report/json_test.go
package report

import (
	"encoding/json"
	"go/token"
	"testing"
)

func TestWriteJSON(t *testing.T) {
	r := NewReport()
	r.Metadata.Modules = []string{"testpkg"}
	r.Metadata.TotalUnits = 1

	r.Units["testpkg.Hello"] = UnitReport{
		Functions: []FunctionInfo{{
			Package:   "testpkg",
			Name:      "Hello",
			Signature: "func Hello(name string) string",
			Position:  token.Position{Filename: "main.go", Line: 10},
		}},
		Summary: FunctionSummary{
			Purpose:  "Returns a greeting",
			Behavior: "Concatenates 'Hello, ' with name",
		},
	}

	r.AddIssue("testpkg.Hello", Issue{
		Position: token.Position{Filename: "main.go", Line: 12},
		Severity: SeverityMedium,
		Category: "cleanliness",
		Message:  "Consider using fmt.Sprintf",
	})

	data, err := WriteJSON(r)
	if err != nil {
		t.Fatalf("WriteJSON: %v", err)
	}

	// Verify it's valid JSON
	var parsed Report
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if parsed.Summary.TotalIssues != 1 {
		t.Errorf("total issues = %d, want 1", parsed.Summary.TotalIssues)
	}
}
```

**Step 3: Run test to verify it fails**

```bash
go test ./report/... -v
```
Expected: FAIL (WriteJSON not defined)

**Step 4: Create report/json.go**

```go
// report/json.go
package report

import (
	"encoding/json"
	"os"
)

// WriteJSON serializes the report to JSON
func WriteJSON(r *Report) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// WriteJSONFile writes the report to a JSON file
func WriteJSONFile(r *Report, path string) error {
	data, err := WriteJSON(r)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
```

**Step 5: Run tests**

```bash
go test ./report/... -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: add report data structures and JSON output"
```

---

## Task 10: Markdown Report Rendering

**Files:**
- Create: `report/markdown.go`
- Create: `report/markdown_test.go`

**Step 1: Write the failing test**

```go
// report/markdown_test.go
package report

import (
	"go/token"
	"strings"
	"testing"
)

func TestWriteMarkdown(t *testing.T) {
	r := NewReport()
	r.Metadata.Modules = []string{"testpkg"}

	r.Units["testpkg.Hello"] = UnitReport{
		Functions: []FunctionInfo{{
			Package:   "testpkg",
			Name:      "Hello",
			Signature: "func Hello(name string) string",
			Position:  token.Position{Filename: "main.go", Line: 10},
		}},
		Summary: FunctionSummary{
			Purpose:  "Returns a greeting",
			Behavior: "Concatenates strings",
		},
	}

	r.AddIssue("testpkg.Hello", Issue{
		Position:   token.Position{Filename: "main.go", Line: 12},
		Severity:   SeverityCritical,
		Category:   "security",
		Message:    "SQL injection vulnerability",
		Suggestion: "Use parameterized queries",
	})

	md := WriteMarkdown(r)

	// Check key sections exist
	if !strings.Contains(md, "# Code Review Report") {
		t.Error("missing title")
	}

	if !strings.Contains(md, "Critical") {
		t.Error("missing severity")
	}

	if !strings.Contains(md, "SQL injection") {
		t.Error("missing issue message")
	}

	if !strings.Contains(md, "Returns a greeting") {
		t.Error("missing function purpose")
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./report/... -v -run TestWriteMarkdown
```
Expected: FAIL (WriteMarkdown not defined)

**Step 3: Create report/markdown.go**

```go
// report/markdown.go
package report

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

// WriteMarkdown renders the report as markdown
func WriteMarkdown(r *Report) string {
	var b strings.Builder

	// Title
	b.WriteString("# Code Review Report\n\n")
	b.WriteString(fmt.Sprintf("Generated: %s | Modules: %s\n\n",
		r.Metadata.GeneratedAt.Format("2006-01-02 15:04"),
		strings.Join(r.Metadata.Modules, ", ")))

	// Summary table
	b.WriteString("## Summary\n\n")
	b.WriteString("| Severity | Count |\n")
	b.WriteString("|----------|-------|\n")

	for _, sev := range []string{"critical", "high", "medium", "low", "info"} {
		if count, ok := r.Summary.BySeverity[sev]; ok && count > 0 {
			b.WriteString(fmt.Sprintf("| %s | %d |\n", strings.Title(sev), count))
		}
	}
	b.WriteString("\n")

	// Critical issues first
	if len(r.Summary.CriticalUnits) > 0 {
		b.WriteString("## Critical Issues\n\n")
		for _, unitID := range r.Summary.CriticalUnits {
			unit := r.Units[unitID]
			writeUnitIssues(&b, unitID, unit, SeverityCritical)
		}
	}

	// High issues
	highUnits := findUnitsWithSeverity(r, SeverityHigh)
	if len(highUnits) > 0 {
		b.WriteString("## High Priority Issues\n\n")
		for _, unitID := range highUnits {
			unit := r.Units[unitID]
			writeUnitIssues(&b, unitID, unit, SeverityHigh)
		}
	}

	// All functions
	b.WriteString("## All Functions\n\n")

	// Sort unit IDs for deterministic output
	unitIDs := make([]string, 0, len(r.Units))
	for id := range r.Units {
		unitIDs = append(unitIDs, id)
	}
	sort.Strings(unitIDs)

	for _, unitID := range unitIDs {
		unit := r.Units[unitID]
		writeUnitSummary(&b, unitID, unit)
	}

	return b.String()
}

func writeUnitIssues(b *strings.Builder, unitID string, unit UnitReport, severity Severity) {
	pos := ""
	if len(unit.Functions) > 0 {
		pos = fmt.Sprintf(" (%s:%d)", unit.Functions[0].Position.Filename, unit.Functions[0].Position.Line)
	}

	b.WriteString(fmt.Sprintf("### %s%s\n\n", unitID, pos))

	for _, issue := range unit.Issues {
		if issue.Severity != severity {
			continue
		}
		b.WriteString(fmt.Sprintf("**[%s] [%s]** %s\n",
			strings.ToUpper(string(issue.Severity)),
			issue.Category,
			issue.Message))

		if issue.Suggestion != "" {
			b.WriteString(fmt.Sprintf("> Suggestion: %s\n", issue.Suggestion))
		}
		b.WriteString("\n")
	}
	b.WriteString("---\n\n")
}

func writeUnitSummary(b *strings.Builder, unitID string, unit UnitReport) {
	b.WriteString(fmt.Sprintf("### %s\n", unitID))

	if unit.Summary.Purpose != "" {
		b.WriteString(fmt.Sprintf("**Purpose:** %s\n", unit.Summary.Purpose))
	}
	if unit.Summary.Behavior != "" {
		b.WriteString(fmt.Sprintf("**Behavior:** %s\n", unit.Summary.Behavior))
	}
	b.WriteString("\n")

	if len(unit.Issues) > 0 {
		b.WriteString("Issues:\n")
		for _, issue := range unit.Issues {
			b.WriteString(fmt.Sprintf("- [%s] %s\n", issue.Severity, issue.Message))
		}
		b.WriteString("\n")
	}
}

func findUnitsWithSeverity(r *Report, severity Severity) []string {
	var units []string
	for unitID, unit := range r.Units {
		for _, issue := range unit.Issues {
			if issue.Severity == severity {
				units = append(units, unitID)
				break
			}
		}
	}
	sort.Strings(units)
	return units
}

// WriteMarkdownFile writes the markdown report to a file
func WriteMarkdownFile(r *Report, path string) error {
	md := WriteMarkdown(r)
	return os.WriteFile(path, []byte(md), 0644)
}
```

**Step 4: Run tests**

```bash
go test ./report/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add markdown report rendering"
```

---

## Task 11: Prompt Template Loading

**Files:**
- Create: `analyze/prompt.go`
- Create: `analyze/prompt_test.go`

**Step 1: Write the failing test**

```go
// analyze/prompt_test.go
package analyze

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadPrompt(t *testing.T) {
	dir := t.TempDir()

	prompt := `You are analyzing the function: {{.Name}}

Signature: {{.Signature}}

Code:
{{.Body}}

Callees:
{{range .Callees}}- {{.Name}}: {{.Purpose}}
{{end}}
`
	if err := os.WriteFile(filepath.Join(dir, "test.txt"), []byte(prompt), 0644); err != nil {
		t.Fatal(err)
	}

	tmpl, err := LoadPrompt(filepath.Join(dir, "test.txt"))
	if err != nil {
		t.Fatalf("LoadPrompt: %v", err)
	}

	ctx := PromptContext{
		Name:      "Hello",
		Signature: "func Hello(name string) string",
		Body:      `return "Hello, " + name`,
		Callees: []CalleeSummary{
			{Name: "concat", Purpose: "concatenates strings"},
		},
	}

	result, err := ExecutePrompt(tmpl, ctx)
	if err != nil {
		t.Fatalf("ExecutePrompt: %v", err)
	}

	if !contains(result, "Hello") {
		t.Errorf("result should contain function name")
	}

	if !contains(result, "concatenates strings") {
		t.Errorf("result should contain callee purpose")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsAt(s, substr))
}

func containsAt(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./analyze/... -v
```
Expected: FAIL (LoadPrompt not defined)

**Step 3: Create analyze/prompt.go**

```go
// analyze/prompt.go
package analyze

import (
	"bytes"
	"os"
	"text/template"
)

// PromptContext holds data available to prompt templates
type PromptContext struct {
	// Function info
	Name      string
	Package   string
	Receiver  string
	Signature string
	Body      string
	Godoc     string

	// For SCCs with multiple functions
	Functions []FunctionContext

	// Callee summaries
	Callees []CalleeSummary

	// External function info
	ExternalFuncs []ExternalFuncContext

	// For non-summary passes
	Summary *SummaryContext
}

// FunctionContext holds info about a single function in an SCC
type FunctionContext struct {
	Name      string
	Receiver  string
	Signature string
	Body      string
	Godoc     string
}

// CalleeSummary holds a callee's summary for context
type CalleeSummary struct {
	Name       string
	Purpose    string
	Behavior   string
	Invariants []string
	Security   []string
}

// ExternalFuncContext holds external function info
type ExternalFuncContext struct {
	Package    string
	Name       string
	Signature  string
	Godoc      string
	Invariants []string
	Pitfalls   []string
}

// SummaryContext holds this unit's summary
type SummaryContext struct {
	Purpose    string
	Behavior   string
	Invariants []string
	Security   []string
}

// LoadPrompt loads a prompt template from a file
func LoadPrompt(path string) (*template.Template, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	tmpl, err := template.New(path).Parse(string(data))
	if err != nil {
		return nil, err
	}

	return tmpl, nil
}

// ExecutePrompt executes a prompt template with the given context
func ExecutePrompt(tmpl *template.Template, ctx PromptContext) (string, error) {
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, ctx); err != nil {
		return "", err
	}
	return buf.String(), nil
}
```

**Step 4: Run tests**

```bash
go test ./analyze/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add prompt template loading and execution"
```

---

## Task 12: LLM Response Parsing

**Files:**
- Create: `analyze/parse.go`
- Create: `analyze/parse_test.go`

**Step 1: Write the failing test**

```go
// analyze/parse_test.go
package analyze

import (
	"testing"
)

func TestParseSummaryResponse(t *testing.T) {
	response := `{
  "purpose": "Fetches user by ID from database",
  "behavior": "Returns nil, ErrNotFound if user doesn't exist",
  "invariants": ["db must not be nil"],
  "security": ["input id is used in SQL query"]
}`

	summary, err := ParseSummaryResponse(response)
	if err != nil {
		t.Fatalf("ParseSummaryResponse: %v", err)
	}

	if summary.Purpose != "Fetches user by ID from database" {
		t.Errorf("purpose = %q", summary.Purpose)
	}

	if len(summary.Invariants) != 1 {
		t.Errorf("invariants count = %d, want 1", len(summary.Invariants))
	}
}

func TestParseIssuesResponse(t *testing.T) {
	response := `{
  "issues": [
    {
      "line": 42,
      "severity": "high",
      "message": "SQL query built with string concatenation",
      "suggestion": "Use parameterized query"
    }
  ]
}`

	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}

	if len(issues) != 1 {
		t.Fatalf("issues count = %d, want 1", len(issues))
	}

	if issues[0].Severity != "high" {
		t.Errorf("severity = %q, want high", issues[0].Severity)
	}

	if issues[0].Line != 42 {
		t.Errorf("line = %d, want 42", issues[0].Line)
	}
}

func TestParseIssuesResponse_WithMarkdown(t *testing.T) {
	// LLMs sometimes wrap JSON in markdown code blocks
	response := "```json\n" + `{
  "issues": [
    {"line": 10, "severity": "medium", "message": "test"}
  ]
}` + "\n```"

	issues, err := ParseIssuesResponse(response)
	if err != nil {
		t.Fatalf("ParseIssuesResponse: %v", err)
	}

	if len(issues) != 1 {
		t.Fatalf("issues count = %d, want 1", len(issues))
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./analyze/... -v -run TestParse
```
Expected: FAIL (ParseSummaryResponse not defined)

**Step 3: Create analyze/parse.go**

```go
// analyze/parse.go
package analyze

import (
	"encoding/json"
	"regexp"
	"strings"
)

// SummaryResponse is the expected JSON structure for summary pass
type SummaryResponse struct {
	Purpose    string   `json:"purpose"`
	Behavior   string   `json:"behavior"`
	Invariants []string `json:"invariants"`
	Security   []string `json:"security"`
}

// IssueResponse is a single issue from the LLM
type IssueResponse struct {
	Line       int    `json:"line"`
	Severity   string `json:"severity"`
	Message    string `json:"message"`
	Suggestion string `json:"suggestion,omitempty"`
}

// IssuesResponse is the expected JSON structure for analysis passes
type IssuesResponse struct {
	Issues []IssueResponse `json:"issues"`
}

// ParseSummaryResponse parses the LLM response for a summary pass
func ParseSummaryResponse(response string) (*SummaryResponse, error) {
	cleaned := cleanJSON(response)

	var summary SummaryResponse
	if err := json.Unmarshal([]byte(cleaned), &summary); err != nil {
		return nil, err
	}

	return &summary, nil
}

// ParseIssuesResponse parses the LLM response for an analysis pass
func ParseIssuesResponse(response string) ([]IssueResponse, error) {
	cleaned := cleanJSON(response)

	var issues IssuesResponse
	if err := json.Unmarshal([]byte(cleaned), &issues); err != nil {
		return nil, err
	}

	return issues.Issues, nil
}

// cleanJSON extracts JSON from markdown code blocks if present
func cleanJSON(s string) string {
	s = strings.TrimSpace(s)

	// Remove markdown code blocks
	codeBlockRegex := regexp.MustCompile("(?s)```(?:json)?\\s*(.+?)\\s*```")
	if matches := codeBlockRegex.FindStringSubmatch(s); len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	return s
}
```

**Step 4: Run tests**

```bash
go test ./analyze/... -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LLM response parsing"
```

---

## Task 13: Analysis Pipeline Core

**Files:**
- Create: `analyze/pipeline.go`

**Step 1: Create analyze/pipeline.go**

```go
// analyze/pipeline.go
package analyze

import (
	"context"
	"encoding/json"
	"fmt"
	"go/token"
	"text/template"

	"github.com/loov/reviewmod/cache"
	"github.com/loov/reviewmod/config"
	"github.com/loov/reviewmod/extract"
	"github.com/loov/reviewmod/llm"
	"github.com/loov/reviewmod/report"
)

// Pipeline runs the analysis passes on all units
type Pipeline struct {
	config    *config.Config
	cache     *cache.Cache
	llmClient llm.Client
	prompts   map[string]*template.Template
	summaries map[string]*SummaryResponse
}

// NewPipeline creates a new analysis pipeline
func NewPipeline(cfg *config.Config, c *cache.Cache, client llm.Client) *Pipeline {
	return &Pipeline{
		config:    cfg,
		cache:     c,
		llmClient: client,
		prompts:   make(map[string]*template.Template),
		summaries: make(map[string]*SummaryResponse),
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
			return nil, fmt.Errorf("summary pass: %w", err)
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

	// Run analysis passes
	for _, pass := range p.config.Analyses {
		if !pass.Enabled || pass.Name == "summary" {
			continue
		}

		issues, err := p.runAnalysisPass(ctx, pass, promptCtx)
		if err != nil {
			return nil, fmt.Errorf("%s pass: %w", pass.Name, err)
		}

		for _, issue := range issues {
			unitReport.Issues = append(unitReport.Issues, report.Issue{
				Position:   token.Position{Line: issue.Line},
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

	// Add callee summaries
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
```

**Step 2: Verify it compiles**

```bash
go build ./...
```
Expected: No errors

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add analysis pipeline core"
```

---

## Task 14: CLI Entry Point

**Files:**
- Create: `main.go`

**Step 1: Create main.go**

```go
// main.go
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
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

	// Extract functions
	fmt.Println("Extracting functions...")
	funcs, err := extract.ExtractFunctions(".", patterns...)
	if err != nil {
		return fmt.Errorf("extract functions: %w", err)
	}
	fmt.Printf("Found %d functions\n", len(funcs))

	// Build callgraph
	fmt.Println("Building callgraph...")
	graph, err := extract.BuildCallgraph(".", patterns...)
	if err != nil {
		return fmt.Errorf("build callgraph: %w", err)
	}

	// Build analysis units
	fmt.Println("Building analysis units...")
	units := extract.BuildAnalysisUnits(funcs, graph)
	fmt.Printf("Created %d analysis units\n", len(units))

	// Create pipeline
	pipeline := analyze.NewPipeline(cfg, c, client)
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
```

**Step 2: Verify it compiles**

```bash
go build ./...
```
Expected: No errors

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add CLI entry point"
```

---

## Task 15: Default Prompts

**Files:**
- Create: `prompts/summary.txt`
- Create: `prompts/security.txt`
- Create: `prompts/errors.txt`
- Create: `prompts/cleanliness.txt`

**Step 1: Create prompts directory and summary.txt**

```bash
mkdir -p prompts
```

```text
You are analyzing a Go function to create a summary of its behavior.

{{if .Functions}}
## Functions (mutually recursive group)
{{range .Functions}}
### {{.Name}}
Signature: {{.Signature}}
{{if .Godoc}}Documentation: {{.Godoc}}{{end}}

```go
{{.Body}}
```
{{end}}
{{else}}
## Function: {{.Name}}
Package: {{.Package}}
{{if .Receiver}}Receiver: {{.Receiver}}{{end}}
Signature: {{.Signature}}
{{if .Godoc}}Documentation: {{.Godoc}}{{end}}

```go
{{.Body}}
```
{{end}}

{{if .Callees}}
## Called Functions
{{range .Callees}}
### {{.Name}}
Purpose: {{.Purpose}}
Behavior: {{.Behavior}}
{{if .Invariants}}Invariants: {{range .Invariants}}- {{.}}
{{end}}{{end}}
{{end}}
{{end}}

Analyze this function and respond with JSON in this exact format:
{
  "purpose": "One sentence describing what this function does",
  "behavior": "How it behaves - side effects, return conditions, error cases",
  "invariants": ["preconditions", "postconditions", "guarantees"],
  "security": ["security-relevant properties for callers to know"]
}

Be precise and concise. Focus on what callers need to know.
```

**Step 2: Create prompts/security.txt**

```text
You are a security auditor reviewing Go code for vulnerabilities.

## Function: {{.Name}}
Package: {{.Package}}
Signature: {{.Signature}}

```go
{{.Body}}
```

{{if .Summary}}
## Function Summary
Purpose: {{.Summary.Purpose}}
Behavior: {{.Summary.Behavior}}
{{if .Summary.Security}}Security notes: {{range .Summary.Security}}- {{.}}
{{end}}{{end}}
{{end}}

{{if .Callees}}
## Called Functions
{{range .Callees}}
### {{.Name}}
{{if .Security}}Security: {{range .Security}}- {{.}}
{{end}}{{end}}
{{end}}
{{end}}

Look for:
- SQL injection
- Command injection
- Path traversal
- XSS vulnerabilities
- Improper input validation
- Sensitive data exposure
- Authentication/authorization issues
- Cryptographic weaknesses
- Race conditions
- Resource leaks

Respond with JSON:
{
  "issues": [
    {
      "line": <line number>,
      "severity": "critical|high|medium|low|info",
      "message": "Description of the security issue",
      "suggestion": "How to fix it"
    }
  ]
}

If no issues found, return {"issues": []}
```

**Step 3: Create prompts/errors.txt**

```text
You are reviewing Go code for error handling issues.

## Function: {{.Name}}
Package: {{.Package}}
Signature: {{.Signature}}

```go
{{.Body}}
```

{{if .Summary}}
## Function Summary
Purpose: {{.Summary.Purpose}}
Behavior: {{.Summary.Behavior}}
{{end}}

{{if .Callees}}
## Called Functions
{{range .Callees}}
### {{.Name}}
Behavior: {{.Behavior}}
{{end}}
{{end}}

Look for:
- Ignored errors (err not checked)
- Errors that should be wrapped for context
- Panic where error return is better
- Missing nil checks before dereferencing
- Resource cleanup missing in error paths (defer)
- Error messages that expose internal details

Respond with JSON:
{
  "issues": [
    {
      "line": <line number>,
      "severity": "critical|high|medium|low|info",
      "message": "Description of the error handling issue",
      "suggestion": "How to fix it"
    }
  ]
}

If no issues found, return {"issues": []}
```

**Step 4: Create prompts/cleanliness.txt**

```text
You are reviewing Go code for cleanliness and best practices.

## Function: {{.Name}}
Package: {{.Package}}
Signature: {{.Signature}}

```go
{{.Body}}
```

{{if .Summary}}
## Function Summary
Purpose: {{.Summary.Purpose}}
{{end}}

Look for:
- Functions that are too long (>50 lines)
- Too many parameters (>5)
- Deep nesting (>3 levels)
- Magic numbers without constants
- Inconsistent naming
- Dead code or unreachable branches
- Redundant type conversions
- Inefficient string concatenation in loops
- Using fmt.Sprintf where simpler methods work

Respond with JSON:
{
  "issues": [
    {
      "line": <line number>,
      "severity": "medium|low|info",
      "message": "Description of the cleanliness issue",
      "suggestion": "How to improve it"
    }
  ]
}

If no issues found, return {"issues": []}
```

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add default prompt templates"
```

---

## Task 16: Example Configuration File

**Files:**
- Create: `reviewmod.cue`

**Step 1: Create reviewmod.cue**

```cue
// reviewmod.cue - Example configuration

llm: {
	provider:  "openai"
	base_url:  "http://localhost:8080/v1"
	model:     "llama3"
	max_tokens: 4096
	temperature: 0.1
}

cache: {
	dir: ".reviewmod/cache"
	enabled: true
}

output: {
	json:     "reviewmod-report.json"
	markdown: "reviewmod-report.md"
}

analyses: [
	{name: "summary", prompt: "prompts/summary.txt"},
	{name: "security", prompt: "prompts/security.txt", include_security_properties: true},
	{name: "errors", prompt: "prompts/errors.txt"},
	{name: "cleanliness", prompt: "prompts/cleanliness.txt"},
]
```

**Step 2: Commit**

```bash
git add -A
git commit -m "feat: add example configuration file"
```

---

## Task 17: Integration Test

**Files:**
- Create: `integration_test.go`

**Step 1: Create integration_test.go**

```go
//go:build integration

package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/loov/reviewmod/analyze"
	"github.com/loov/reviewmod/cache"
	"github.com/loov/reviewmod/config"
	"github.com/loov/reviewmod/extract"
	"github.com/loov/reviewmod/llm"
)

// MockLLMClient returns predefined responses for testing
type MockLLMClient struct{}

func (m *MockLLMClient) Complete(ctx context.Context, req llm.Request) (llm.Response, error) {
	// Return mock summary
	return llm.Response{
		Content: `{"purpose": "test function", "behavior": "does testing", "invariants": [], "security": []}`,
	}, nil
}

func TestIntegration_FullPipeline(t *testing.T) {
	dir := t.TempDir()

	// Create test Go module
	goMod := `module testpkg

go 1.25
`
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatal(err)
	}

	goFile := `package testpkg

// Add adds two numbers.
func Add(a, b int) int {
	return a + b
}

// Multiply multiplies by calling Add repeatedly.
func Multiply(a, b int) int {
	result := 0
	for i := 0; i < b; i++ {
		result = Add(result, a)
	}
	return result
}
`
	if err := os.WriteFile(filepath.Join(dir, "math.go"), []byte(goFile), 0644); err != nil {
		t.Fatal(err)
	}

	// Extract functions
	funcs, err := extract.ExtractFunctions(dir, "./...")
	if err != nil {
		t.Fatalf("ExtractFunctions: %v", err)
	}

	if len(funcs) != 2 {
		t.Fatalf("got %d functions, want 2", len(funcs))
	}

	// Build callgraph
	graph, err := extract.BuildCallgraph(dir, "./...")
	if err != nil {
		t.Fatalf("BuildCallgraph: %v", err)
	}

	// Multiply should call Add
	if !contains(graph["testpkg.Multiply"], "testpkg.Add") {
		t.Errorf("Multiply should call Add")
	}

	// Build analysis units
	units := extract.BuildAnalysisUnits(funcs, graph)

	if len(units) != 2 {
		t.Fatalf("got %d units, want 2", len(units))
	}

	// First unit should be Add (no dependencies)
	if units[0].Functions[0].Name != "Add" {
		t.Errorf("first unit should be Add, got %s", units[0].Functions[0].Name)
	}

	// Second unit should be Multiply
	if units[1].Functions[0].Name != "Multiply" {
		t.Errorf("second unit should be Multiply, got %s", units[1].Functions[0].Name)
	}
}

func contains(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}
```

**Step 2: Run integration test**

```bash
go test -tags=integration -v ./...
```
Expected: PASS

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add integration test"
```

---

## Summary

This plan implements reviewmod in 17 tasks:

1. **Tasks 1-2**: Project setup, core types, LLM interface
2. **Tasks 3-6**: Callgraph extraction (Tarjan SCC, package loading, CHA, unit building)
3. **Task 7**: Cue configuration loading
4. **Task 8**: Disk cache
5. **Tasks 9-10**: Report generation (JSON, markdown)
6. **Tasks 11-13**: Analysis pipeline (prompts, parsing, core)
7. **Task 14**: CLI entry point
8. **Tasks 15-16**: Default prompts and example config
9. **Task 17**: Integration test

Each task is self-contained with tests and commits frequently.
