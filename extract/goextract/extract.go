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

// ExtractTypes extracts all user-defined types from loaded packages.
func ExtractTypes(p *Packages) []*extract.TypeInfo {
	return ExtractTypesFromPackages(p.Pkgs)
}

// ExtractTypesFromPackages walks each package's AST and produces a
// TypeInfo for every TypeSpec. Struct / interface declarations keep
// their verbatim source in Body so the LLM sees the field layout.
func ExtractTypesFromPackages(pkgs []*packages.Package) []*extract.TypeInfo {
	fileContents := make(map[string][]byte)
	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			pos := pkg.Fset.Position(file.Pos())
			if _, ok := fileContents[pos.Filename]; ok {
				continue
			}
			if content, err := os.ReadFile(pos.Filename); err == nil {
				fileContents[pos.Filename] = content
			}
		}
	}

	var types []*extract.TypeInfo
	for _, pkg := range pkgs {
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				gen, ok := decl.(*ast.GenDecl)
				if !ok || gen.Tok != token.TYPE {
					continue
				}
				for _, spec := range gen.Specs {
					ts, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}

					startPos := ts.Pos()
					if ts.Doc != nil {
						startPos = ts.Doc.Pos()
					} else if len(gen.Specs) == 1 && gen.Doc != nil {
						startPos = gen.Doc.Pos()
					}
					endPos := ts.End()

					ti := &extract.TypeInfo{
						Package:  pkg.PkgPath,
						Name:     ts.Name.Name,
						Kind:     typeKind(ts.Type),
						Position: pkg.Fset.Position(startPos),
					}

					ti.Signature = formatTypeSignature(pkg.Fset, ts)

					if content := fileContents[pkg.Fset.Position(startPos).Filename]; content != nil {
						start := pkg.Fset.Position(startPos).Offset
						end := pkg.Fset.Position(endPos).Offset
						if start >= 0 && end <= len(content) {
							ti.Body = string(content[start:end])
						}
					}
					if ti.Body == "" {
						var buf strings.Builder
						printer.Fprint(&buf, pkg.Fset, ts)
						ti.Body = buf.String()
					}

					if ts.Doc != nil {
						ti.Godoc = ts.Doc.Text()
					} else if len(gen.Specs) == 1 && gen.Doc != nil {
						ti.Godoc = gen.Doc.Text()
					}

					types = append(types, ti)
				}
			}
		}
	}
	return types
}

// LinkMethodsToTypes populates FunctionInfo.ReceiverType and appends
// method IDs to the matching TypeInfo.Methods slice. Returns a map
// keyed by type ID (Package + "." + Name).
func LinkMethodsToTypes(funcs []*extract.FunctionInfo, types []*extract.TypeInfo) map[string]*extract.TypeInfo {
	byID := make(map[string]*extract.TypeInfo, len(types))
	for _, t := range types {
		byID[t.Package+"."+t.Name] = t
	}
	for _, fn := range funcs {
		if fn.Receiver == "" {
			continue
		}
		name := stripReceiverPointer(fn.Receiver)
		id := fn.Package + "." + name
		ti, ok := byID[id]
		if !ok {
			continue
		}
		fn.ReceiverType = id
		ti.Methods = append(ti.Methods, fn.ID())
	}
	return byID
}

// stripReceiverPointer removes a leading "*" and any trailing generic
// type parameters from a Go receiver expression, leaving the bare type
// name. E.g. "*Foo[T]" -> "Foo".
func stripReceiverPointer(recv string) string {
	recv = strings.TrimPrefix(recv, "*")
	if i := strings.IndexByte(recv, '['); i >= 0 {
		recv = recv[:i]
	}
	return recv
}

// typeKind classifies a type expression. Non-struct/interface
// declarations (aliases, named primitive types) are reported as "type".
func typeKind(expr ast.Expr) string {
	switch expr.(type) {
	case *ast.StructType:
		return "struct"
	case *ast.InterfaceType:
		return "interface"
	}
	return "type"
}

// formatTypeSignature renders a single-line "type Name …" header.
func formatTypeSignature(fset *token.FileSet, ts *ast.TypeSpec) string {
	var buf strings.Builder
	buf.WriteString("type ")
	buf.WriteString(ts.Name.Name)
	if ts.TypeParams != nil {
		printer.Fprint(&buf, fset, ts.TypeParams)
	}
	switch ts.Type.(type) {
	case *ast.StructType:
		buf.WriteString(" struct")
	case *ast.InterfaceType:
		buf.WriteString(" interface")
	default:
		buf.WriteString(" ")
		printer.Fprint(&buf, fset, ts.Type)
	}
	return buf.String()
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
