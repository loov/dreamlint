// analyze/prompt.go
package analyze

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

// LoadPrompt loads a prompt template.
// Paths starting with "builtin:" load from embedded prompts (e.g., "builtin:summary").
// Other paths are loaded from the filesystem.
func LoadPrompt(path string) (*template.Template, error) {
	if path == "" {
		return nil, errors.New("prompt path is empty")
	}

	var baseData, data []byte
	var err error
	var name string

	if strings.HasPrefix(path, "builtin:") {
		// Load from embedded prompts
		name = strings.TrimPrefix(path, "builtin:")
		baseData, err = embeddedPrompts.ReadFile("prompts/_base.txt")
		if err != nil {
			return nil, fmt.Errorf("failed to read embedded base template: %w", err)
		}
		data, err = embeddedPrompts.ReadFile("prompts/" + name + ".txt")
		if err != nil {
			return nil, fmt.Errorf("failed to read embedded prompt %s: %w", name, err)
		}
	} else {
		// Load from filesystem
		name = path
		baseData, err = os.ReadFile(filepath.Join(filepath.Dir(path), "_base.txt"))
		if err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("failed to read base template: %w", err)
		}
		data, err = os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("failed to read prompt file: %w", err)
		}
	}

	tmpl := template.New(name)

	// Parse base template if available
	if len(baseData) > 0 {
		if _, err := tmpl.Parse(string(baseData)); err != nil {
			return nil, fmt.Errorf("failed to parse base template: %w", err)
		}
	}

	// Parse the main template content into a named template
	mainTmpl, err := tmpl.New(name).Parse(string(data))
	if err != nil {
		return nil, fmt.Errorf("failed to parse prompt template: %w", err)
	}

	return mainTmpl, nil
}

// ExecutePrompt executes a prompt template with the given context
func ExecutePrompt(tmpl *template.Template, ctx PromptContext) (string, error) {
	if tmpl == nil {
		return "", errors.New("template is nil")
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, ctx); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}
	return buf.String(), nil
}
