// analyze/prompt_test.go
package analyze

import (
	"strings"
	"testing"
)

func TestLoadPrompt(t *testing.T) {
	// Test loading embedded prompts
	prompts := []string{
		"summary",
		"security",
		"errors",
		"cleanliness",
		"concurrency",
		"performance",
		"api-design",
		"testing",
		"logging",
		"resources",
		"validation",
		"dependencies",
		"complexity",
	}

	for _, name := range prompts {
		t.Run(name, func(t *testing.T) {
			tmpl, err := LoadPrompt("builtin:" + name)
			if err != nil {
				t.Fatalf("LoadPrompt(builtin:%s): %v", name, err)
			}
			if tmpl == nil {
				t.Fatal("template is nil")
			}
		})
	}
}

func TestExecutePrompt(t *testing.T) {
	tmpl, err := LoadPrompt("builtin:summary")
	if err != nil {
		t.Fatalf("LoadPrompt: %v", err)
	}

	ctx := PromptContext{
		Name:      "Hello",
		Package:   "main",
		Signature: "func Hello(name string) string",
		Body:      `func Hello(name string) string { return "Hello, " + name }`,
	}

	result, err := ExecutePrompt(tmpl, ctx)
	if err != nil {
		t.Fatalf("ExecutePrompt: %v", err)
	}

	if !strings.Contains(result, "Hello") {
		t.Errorf("result should contain function name")
	}

	if !strings.Contains(result, "main") {
		t.Errorf("result should contain package name")
	}
}
