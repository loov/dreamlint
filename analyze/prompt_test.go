// analyze/prompt_test.go
package analyze

import (
	"os"
	"path/filepath"
	"strings"
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

	if !strings.Contains(result, "Hello") {
		t.Errorf("result should contain function name")
	}

	if !strings.Contains(result, "concatenates strings") {
		t.Errorf("result should contain callee purpose")
	}
}
