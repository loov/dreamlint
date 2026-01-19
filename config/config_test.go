// config/config_test.go
package config

import (
	"testing"
)

func TestLoadConfig(t *testing.T) {
	cfg, err := LoadConfig([]string{"./testdata/simple.cue"}, nil)
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
	if cfg.Cache.Dir != ".dreamlint/cache" {
		t.Errorf("cache.dir = %s, want .dreamlint/cache", cfg.Cache.Dir)
	}
	if len(cfg.Analyse) != 2 {
		t.Errorf("analyses count = %d, want 2", len(cfg.Analyse))
	}
}

func TestLoadConfigMultipleFiles(t *testing.T) {
	cfg, err := LoadConfig([]string{
		"./testdata/base.cue",
		"./testdata/override.cue",
	}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	// Model should be overridden
	if cfg.LLM.Model != "gpt-4" {
		t.Errorf("model = %s, want gpt-4", cfg.LLM.Model)
	}

	// Base URL should be preserved
	if cfg.LLM.BaseURL != "http://localhost:8080/v1" {
		t.Errorf("base_url = %s, want http://localhost:8080/v1", cfg.LLM.BaseURL)
	}
}

func TestLoadConfigInline(t *testing.T) {
	cfg, err := LoadConfig(
		[]string{"./testdata/base.cue"},
		[]string{`llm: { model: "claude-3" }`},
	)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	// Model should be overridden by inline config
	if cfg.LLM.Model != "claude-3" {
		t.Errorf("model = %s, want claude-3", cfg.LLM.Model)
	}
}
