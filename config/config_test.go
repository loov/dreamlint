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
