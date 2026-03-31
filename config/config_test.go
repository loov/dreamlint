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

func TestLoadConfig_Auto(t *testing.T) {
	cfg, err := LoadConfig([]string{"./testdata/auto.cue"}, nil)
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

func TestLoadConfig_Auto_Specify(t *testing.T) {
	cfg, err := LoadConfig([]string{"./testdata/auto.cue", "./testdata/auto.cue"},
		[]string{
			"analyse: [pass.security]",
		})
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
	if len(cfg.Analyse) != 1 {
		t.Errorf("analyses count = %d, want 1", len(cfg.Analyse))
	}
}

func TestLoadConfig_Auto_SpecifyInline(t *testing.T) {
	cfg, err := LoadConfig([]string{"./testdata/auto.cue"},
		[]string{
			"analyse: [pass.security]",
		})
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
	if len(cfg.Analyse) != 1 {
		t.Errorf("analyses count = %d, want 1", len(cfg.Analyse))
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

func TestLoadConfig_EnvVar_Anthropic(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
	cfg, err := LoadConfig([]string{"./testdata/anthropic.cue"}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "test-anthropic-key" {
		t.Errorf("api_key = %q, want %q", cfg.LLM.APIKey, "test-anthropic-key")
	}
}

func TestLoadConfig_EnvVar_Gemini(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-gemini-key")
	cfg, err := LoadConfig([]string{"./testdata/gemini.cue"}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "test-gemini-key" {
		t.Errorf("api_key = %q, want %q", cfg.LLM.APIKey, "test-gemini-key")
	}
}

func TestLoadConfig_EnvVar_OpenAI(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-openai-key")
	cfg, err := LoadConfig([]string{"./testdata/openai.cue"}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "test-openai-key" {
		t.Errorf("api_key = %q, want %q", cfg.LLM.APIKey, "test-openai-key")
	}
}

func TestLoadConfig_EnvVar_Generic(t *testing.T) {
	t.Setenv("DREAMLINT_API_KEY", "test-generic-key")
	cfg, err := LoadConfig([]string{"./testdata/base.cue"}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "test-generic-key" {
		t.Errorf("api_key = %q, want %q", cfg.LLM.APIKey, "test-generic-key")
	}
}

func TestLoadConfig_EnvVar_GenericBeatsProviderSpecific(t *testing.T) {
	t.Setenv("DREAMLINT_API_KEY", "generic-key")
	t.Setenv("ANTHROPIC_API_KEY", "anthropic-key")
	cfg, err := LoadConfig([]string{"./testdata/anthropic.cue"}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "generic-key" {
		t.Errorf("api_key = %q, want DREAMLINT_API_KEY to win", cfg.LLM.APIKey)
	}
}

func TestLoadConfig_EnvVar_ConfigBeatsEnv(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-key")
	cfg, err := LoadConfig(
		[]string{"./testdata/anthropic.cue"},
		[]string{`llm: { api_key: "config-key" }`},
	)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "config-key" {
		t.Errorf("api_key = %q, want config value to beat env var", cfg.LLM.APIKey)
	}
}

func TestLoadConfig_EnvVar_WrongProviderNotUsed(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	// base_url is anthropic — GEMINI_API_KEY should not be used
	cfg, err := LoadConfig([]string{"./testdata/anthropic.cue"}, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LLM.APIKey != "" {
		t.Errorf("api_key = %q, want empty: wrong-provider env var must not be used", cfg.LLM.APIKey)
	}
}
