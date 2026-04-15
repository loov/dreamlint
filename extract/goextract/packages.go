package goextract

import (
	"fmt"

	"golang.org/x/tools/go/packages"
)

// Packages holds loaded package data for reuse across extraction and callgraph building.
type Packages struct {
	Pkgs []*packages.Package
}

// LoadPackages loads Go packages once for use by ExtractFunctions and BuildCallgraph.
func LoadPackages(dir string, patterns ...string) (*Packages, error) {
	cfg := &packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedSyntax |
			packages.NeedTypes |
			packages.NeedTypesInfo |
			packages.NeedImports |
			packages.NeedDeps,
		Dir: dir,
	}

	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, fmt.Errorf("load packages: %w", err)
	}

	for _, pkg := range pkgs {
		if len(pkg.Errors) > 0 {
			return nil, fmt.Errorf("package %s has errors: %v", pkg.PkgPath, pkg.Errors)
		}
	}

	return &Packages{Pkgs: pkgs}, nil
}
