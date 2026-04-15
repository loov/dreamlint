package goextract

import (
	"go/ast"
	"go/printer"
	"go/token"
	"os"
	"strings"

	"golang.org/x/tools/go/packages"

	"github.com/loov/dreamlint/extract"
)

// ExtractFunctions extracts all function information from loaded packages.
func ExtractFunctions(p *Packages) []*extract.FunctionInfo {
	return ExtractFunctionsFromPackages(p.Pkgs)
}

// ExtractFunctionsFromPackages extracts function information from a slice of packages.
func ExtractFunctionsFromPackages(pkgs []*packages.Package) []*extract.FunctionInfo {
	var funcs []*extract.FunctionInfo

	// Build a map of filename to file content for source extraction
	fileContents := make(map[string][]byte)

	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			pos := pkg.Fset.Position(file.Pos())
			if _, ok := fileContents[pos.Filename]; ok {
				continue
			}
			// Read file content for source extraction
			content, err := os.ReadFile(pos.Filename)
			if err == nil {
				fileContents[pos.Filename] = content
			}
		}
	}

	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			filePos := pkg.Fset.Position(file.Pos())
			content := fileContents[filePos.Filename]

			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok {
					continue
				}

				// Determine position of the function (or its doc comment if present)
				startPos := fn.Pos()
				if fn.Doc != nil {
					startPos = fn.Doc.Pos()
				}
				endPos := fn.End()

				info := &extract.FunctionInfo{
					Package:  pkg.PkgPath,
					Name:     fn.Name.Name,
					Position: pkg.Fset.Position(startPos),
				}

				// Extract receiver
				if fn.Recv != nil && len(fn.Recv.List) > 0 {
					var buf strings.Builder
					printer.Fprint(&buf, pkg.Fset, fn.Recv.List[0].Type)
					info.Receiver = buf.String()
				}

				// Extract signature
				info.Signature = formatSignature(pkg.Fset, fn)

				// Extract full function from source (preserves comments and formatting)
				if content != nil {
					startOffset := pkg.Fset.Position(startPos).Offset
					endOffset := pkg.Fset.Position(endPos).Offset
					if startOffset >= 0 && endOffset <= len(content) {
						info.Body = string(content[startOffset:endOffset])
					}
				}
				// Fallback to printer if source extraction failed
				if info.Body == "" {
					var buf strings.Builder
					printer.Fprint(&buf, pkg.Fset, fn)
					info.Body = buf.String()
				}

				// Extract godoc
				if fn.Doc != nil {
					info.Godoc = fn.Doc.Text()
				}

				funcs = append(funcs, info)
			}
		}
	}

	return funcs
}

func formatSignature(fset *token.FileSet, fn *ast.FuncDecl) string {
	var buf strings.Builder
	buf.WriteString("func ")

	if fn.Recv != nil && len(fn.Recv.List) > 0 {
		buf.WriteString("(")
		printer.Fprint(&buf, fset, fn.Recv.List[0].Type)
		buf.WriteString(") ")
	}

	buf.WriteString(fn.Name.Name)
	printer.Fprint(&buf, fset, fn.Type.Params)
	if fn.Type.Results != nil {
		results := fn.Type.Results
		if len(results.List) == 1 && len(results.List[0].Names) == 0 {
			// Single unnamed return value: no parens
			buf.WriteString(" ")
			printer.Fprint(&buf, fset, results.List[0].Type)
		} else if len(results.List) > 0 {
			buf.WriteString(" ")
			printer.Fprint(&buf, fset, results)
		}
	}

	return buf.String()
}
