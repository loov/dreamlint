package config

import (
	_ "embed"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

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
	Name         string     `json:"name"`
	Prompt       string     `json:"prompt,omitempty"`
	InlinePrompt string     `json:"inline_prompt,omitempty"`
	PromptDir    string     `json:"prompt_dir,omitempty"`
	Enabled      bool       `json:"enabled"`
	LLM          *LLMConfig `json:"llm,omitempty"`
}

// envToCUE generates a CUE source file that defines an env struct
// with the provided key-value pairs.
func envToCUE(env map[string]string) string {
	var b strings.Builder
	b.WriteString("package config\n\nenv: {\n")
	for k, v := range env {
		fmt.Fprintf(&b, "\t%s: %s\n", strconv.Quote(k), strconv.Quote(v))
	}
	b.WriteString("}\n")
	return b.String()
}

// LoadConfig loads and validates Cue configuration from multiple files and inline strings.
// The env map is injected as an env struct accessible in CUE configs.
func LoadConfig(paths []string, inlineConfigs []string, env map[string]string) (*Config, error) {
	// Expand globs in paths, restricting matches to .cue files.
	paths, err := expandGlobs(paths)
	if err != nil {
		return nil, err
	}

	ctx := cuecontext.New()

	// Build overlay with all config files to compile them together
	overlay := make(map[string]load.Source)

	// Add schema to overlay
	overlay["/schema.cue"] = load.FromString(schemaCue)

	// Read and add all config files to overlay.
	// Each file gets _config_dir appended so the schema can default
	// prompt_dir for passes defined in the config.
	for i, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read config %s: %w", path, err)
		}
		abs, err := filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("abs path %s: %w", path, err)
		}
		dir := filepath.Dir(abs)
		// Pre-parse the file to discover which passes it defines,
		// then inject prompt_dir for those specific passes.
		content := string(data)
		passNames := parsePassNames(ctx, data)
		for _, name := range passNames {
			content += fmt.Sprintf("\npass: %s: prompt_dir: string | *%s\n", name, strconv.Quote(dir))
		}
		virtualPath := fmt.Sprintf("/config_%d.cue", i)
		overlay[virtualPath] = load.FromString(content)
	}

	// Add inline configs to overlay
	for i, inline := range inlineConfigs {
		prefixed := "package config\n" + inline
		virtualPath := fmt.Sprintf("/inline_%d.cue", i)
		overlay[virtualPath] = load.FromString(prefixed)
	}

	// Add env vars to overlay (always present so schema can reference env)
	if env == nil {
		env = map[string]string{}
	}
	overlay["/env.cue"] = load.FromString(envToCUE(env))

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

	// Extract #Config schema from the built instance (which includes env)
	// and unify it with the config values.
	schema := values[0].LookupPath(cue.ParsePath("#Config"))
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

// expandGlobs expands glob patterns in paths, filtering results to .cue files.
// Non-glob paths are passed through as-is.
func expandGlobs(paths []string) ([]string, error) {
	var expanded []string
	for _, p := range paths {
		if !strings.ContainsAny(p, "*?[") {
			expanded = append(expanded, p)
			continue
		}
		matches, err := filepath.Glob(p)
		if err != nil {
			return nil, fmt.Errorf("invalid glob pattern %q: %w", p, err)
		}
		sort.Strings(matches)
		for _, m := range matches {
			if filepath.Ext(m) == ".cue" {
				expanded = append(expanded, m)
			}
		}
	}
	return expanded, nil
}

// parsePassNames does a quick standalone parse of a config file to extract
// the pass names it defines. Returns nil if the file can't be parsed alone.
func parsePassNames(ctx *cue.Context, data []byte) []string {
	val := ctx.CompileBytes(data)
	if val.Err() != nil {
		return nil
	}
	passVal := val.LookupPath(cue.ParsePath("pass"))
	if !passVal.Exists() {
		return nil
	}
	iter, err := passVal.Fields()
	if err != nil {
		return nil
	}
	var names []string
	for iter.Next() {
		names = append(names, iter.Selector().String())
	}
	return names
}
