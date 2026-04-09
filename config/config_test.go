package config

import (
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	cfg, err := LoadConfig([]string{"./testdata/simple.cue"}, nil, nil)
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
	cfg, err := LoadConfig([]string{"./testdata/auto.cue"}, nil, nil)
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
		}, nil)
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
		}, nil)
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
	}, nil, nil)
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
		nil,
	)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	// Model should be overridden by inline config
	if cfg.LLM.Model != "claude-3" {
		t.Errorf("model = %s, want claude-3", cfg.LLM.Model)
	}
}

func TestLoadConfig_PromptDir(t *testing.T) {
	cfg, err := LoadConfig([]string{"./testdata/project_a/config.cue"}, nil, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	absDir, _ := filepath.Abs("./testdata/project_a")
	if len(cfg.Analyse) != 1 {
		t.Fatalf("analyses count = %d, want 1", len(cfg.Analyse))
	}
	if cfg.Analyse[0].PromptDir != absDir {
		t.Errorf("prompt_dir = %s, want %s", cfg.Analyse[0].PromptDir, absDir)
	}
}

func TestLoadConfig_PromptDir_MultipleFiles(t *testing.T) {
	cfg, err := LoadConfig([]string{
		"./testdata/project_a/config.cue",
		"./testdata/project_b/config.cue",
	}, nil, nil)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	absDirA, _ := filepath.Abs("./testdata/project_a")
	absDirB, _ := filepath.Abs("./testdata/project_b")

	if len(cfg.Analyse) != 2 {
		t.Fatalf("analyses count = %d, want 2", len(cfg.Analyse))
	}

	dirs := map[string]string{}
	for _, pass := range cfg.Analyse {
		dirs[pass.Name] = pass.PromptDir
	}

	if dirs["summary"] != absDirA {
		t.Errorf("summary prompt_dir = %s, want %s", dirs["summary"], absDirA)
	}
	if dirs["security"] != absDirB {
		t.Errorf("security prompt_dir = %s, want %s", dirs["security"], absDirB)
	}
}

func TestLoadConfigEnv(t *testing.T) {
	cfg, err := LoadConfig(
		[]string{"./testdata/base.cue"},
		[]string{`llm: { api_key: env.MY_API_KEY }`},
		map[string]string{"MY_API_KEY": "test-secret-key"},
	)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if cfg.LLM.APIKey != "test-secret-key" {
		t.Errorf("api_key = %s, want test-secret-key", cfg.LLM.APIKey)
	}
}
