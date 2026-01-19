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
	LLM     LLMConfig      `json:"llm"`
	Cache   CacheConfig    `json:"cache"`
	Output  OutputConfig   `json:"output"`
	Analyse []AnalysisPass `json:"analyse"`
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
	SARIF    string `json:"sarif"`
}

// AnalysisPass defines a single analysis pass
type AnalysisPass struct {
	Name    string     `json:"name"`
	Prompt  string     `json:"prompt"`
	Enabled bool       `json:"enabled"`
	LLM     *LLMConfig `json:"llm,omitempty"`
}

// LoadConfig loads and validates Cue configuration from multiple files and inline strings.
func LoadConfig(paths []string, inlineConfigs []string) (*Config, error) {
	ctx := cuecontext.New()

	// Load schema
	schemaVal := ctx.CompileString(schemaCue)
	if schemaVal.Err() != nil {
		return nil, fmt.Errorf("compile schema: %w", schemaVal.Err())
	}

	schema := schemaVal.LookupPath(cue.ParsePath("#Config"))

	unified := schema

	// Load each config file
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read config %s: %w", path, err)
		}

		val := ctx.CompileBytes(data)
		if val.Err() != nil {
			return nil, fmt.Errorf("compile config %s: %w", path, val.Err())
		}

		unified = unified.Unify(val)
		if unified.Err() != nil {
			return nil, fmt.Errorf("unify config %s: %w", path, unified.Err())
		}
	}

	// Load each inline config
	for _, inline := range inlineConfigs {
		val := ctx.CompileString(inline)
		if val.Err() != nil {
			return nil, fmt.Errorf("compile inline config %q: %w", inline, val.Err())
		}

		unified = unified.Unify(val)
		if unified.Err() != nil {
			return nil, fmt.Errorf("unify inline config %q: %w", inline, unified.Err())
		}
	}

	// Decode each source to a partial config and merge
	var cfg Config
	if err := unified.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("decode config: %w", err)
	}
	return &cfg, nil
}
