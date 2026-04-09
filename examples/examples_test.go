package examples

import (
	"testing"

	"github.com/loov/dreamlint/config"
)

const baseCue = "../config/testdata/base.cue"

func TestCategories(t *testing.T) {
	categories := []struct {
		name  string
		glob  string
		count int
	}{
		{"core", "core/*", 4},
		{"concurrency", "concurrency/*", 1},
		{"architecture", "architecture/*", 4},
		{"web", "web/*", 5},
		{"testing", "testing/*", 1},
	}

	for _, cat := range categories {
		t.Run(cat.name, func(t *testing.T) {
			cfg, err := config.LoadConfig([]string{baseCue, cat.glob}, nil, nil)
			if err != nil {
				t.Fatalf("LoadConfig: %v", err)
			}
			if len(cfg.Analyse) != cat.count {
				names := make([]string, len(cfg.Analyse))
				for i, p := range cfg.Analyse {
					names[i] = p.Name
				}
				t.Errorf("analyses count = %d, want %d; got %v", len(cfg.Analyse), cat.count, names)
			}
		})
	}
}
