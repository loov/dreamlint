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
	SARIF    string `json:"sarif"`
}

// AnalysisPass defines a single analysis pass
type AnalysisPass struct {
	Name                      string     `json:"name"`
	Prompt                    string     `json:"prompt"`
	Enabled                   bool       `json:"enabled"`
	LLM                       *LLMConfig `json:"llm,omitempty"`
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
