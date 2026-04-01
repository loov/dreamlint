package main

import (
	"bytes"
	"context"
	"testing"

	"github.com/zeebo/clingy"
)

// stubCmd delegates Setup to an inner cmdRun and returns nil from Execute,
// letting tests verify flag parsing without invoking the real pipeline.
type stubCmd struct {
	inner *cmdRun
}

func (c *stubCmd) Setup(p clingy.Parameters)        { c.inner.Setup(p) }
func (c *stubCmd) Execute(_ context.Context) error  { return nil }

func runCLI(t *testing.T, args ...string) *cmdRun {
	t.Helper()
	inner := new(cmdRun)
	cmd := &stubCmd{inner: inner}
	env := clingy.Environment{
		Name:   "dreamlint",
		Stdout: new(bytes.Buffer),
		Stderr: new(bytes.Buffer),
		Args:   append([]string{"run"}, args...),
	}
	ok, err := env.Run(context.Background(), func(cmds clingy.Commands) {
		cmds.New("run", "analyze packages for issues", cmd)
	})
	if err != nil {
		t.Fatalf("env.Run: %v", err)
	}
	if !ok {
		t.Fatal("env.Run: setup or execution failed")
	}
	return inner
}

func TestCmdRun_ResumeFlag(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want bool
	}{
		{"absent", nil, false},
		{"before patterns", []string{"-resume"}, true},
		{"after patterns", []string{"./...", "-resume"}, true},
		{"explicit false", []string{"-resume=false"}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := runCLI(t, tc.args...).resume
			if got != tc.want {
				t.Errorf("resume = %v, want %v", got, tc.want)
			}
		})
	}
}
