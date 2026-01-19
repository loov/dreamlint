package config

import (
	_ "embed"
	"fmt"
	"os"

	"cuelang.org/go/cue"
	"cuelang.org/go/cue/cuecontext"
	"cuelang.org/go/cue/load"
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

	// Build overlay with all config files to compile them together
	overlay := make(map[string]load.Source)

	// Add schema to overlay
	overlay["/schema.cue"] = load.FromString(schemaCue)

	// Read and add all config files to overlay
	for i, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read config %s: %w", path, err)
		}
		// Use virtual paths to avoid filesystem dependencies
		virtualPath := fmt.Sprintf("/config_%d.cue", i)
		overlay[virtualPath] = load.FromBytes(data)
	}

	// Add inline configs to overlay
	for i, inline := range inlineConfigs {
		prefixed := "package config\n" + inline
		virtualPath := fmt.Sprintf("/inline_%d.cue", i)
		overlay[virtualPath] = load.FromString(prefixed)
	}

	// Load all files together as a single instance
	cfg_load := &load.Config{
		Dir:     "/",
		Overlay: overlay,
		Package: "config",
	}

	instances := load.Instances([]string{"."}, cfg_load)
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances loaded")
	}

	// Build the instance
	values, err := ctx.BuildInstances(instances)
	if err != nil {
		return nil, fmt.Errorf("build config: %w", err)
	}

	unified := schema.Unify(values[0])
	if unified.Err() != nil {
		return nil, fmt.Errorf("unify config with schema: %w", unified.Err())
	}

	// Decode to config struct
	var cfg Config
	if err := unified.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("decode config: %w", err)
	}
	return &cfg, nil
}
