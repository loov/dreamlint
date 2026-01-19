package main

import (
	"context"

	"github.com/zeebo/clingy"
)

type cmdRun struct {
	configPaths   []string
	inlineConfigs []string
	format        string
	resume        bool
	promptsDir    string
	patterns      []string
}

func (c *cmdRun) Setup(params clingy.Parameters) {
	c.configPaths = params.Flag("config", "path to config file",
		[]string{"dreamlint.cue"},
		clingy.Repeated,
	).([]string)

	c.inlineConfigs = params.Flag("c", "inline CUE config",
		[]string{},
		clingy.Repeated,
	).([]string)

	c.format = params.Flag("format", "output format: json, markdown, sarif, or all", "all").(string)

	c.resume = params.Flag("resume", "resume from existing partial report", false,
		clingy.Boolean,
	).(bool)

	c.promptsDir = params.Flag("prompts", "directory to load prompts from", "").(string)

	c.patterns = params.Arg("patterns", "packages to analyze",
		clingy.Optional,
		clingy.Repeated,
	).([]string)
}

func (c *cmdRun) Execute(ctx context.Context) error {
	patterns := c.patterns
	if len(patterns) == 0 {
		patterns = []string{"./..."}
	}

	// workaround for clingy bug
	if len(c.configPaths) == 0 {
		c.configPaths = []string{"dreamlint.cue"}
	}

	return run(c.configPaths, c.inlineConfigs, c.format, c.resume, c.promptsDir, patterns)
}
