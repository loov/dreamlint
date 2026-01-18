// extract/external.go
package extract

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/packages"
)

// ExternalFunc holds shallow info about external dependencies.
type ExternalFunc struct {
	Package   string
	Name      string
	Signature string
	Godoc     string
}

// ExtractExternalFuncs finds functions called from the analyzed packages that are
// defined in external packages (dependencies). Returns a map from function ID to ExternalFunc.
func ExtractExternalFuncs(p *Packages, graph map[string][]string) map[string]*ExternalFunc {
	// Build set of internal package paths
	internal := make(map[string]bool)
	for _, pkg := range p.Pkgs {
		internal[pkg.PkgPath] = true
	}

	// Collect all callee IDs that are external
	externalIDs := make(map[string]bool)
	for _, callees := range graph {
		for _, calleeID := range callees {
			pkg := packageFromID(calleeID)
			if pkg != "" && !internal[pkg] {
				externalIDs[calleeID] = true
			}
		}
	}

	if len(externalIDs) == 0 {
		return nil
	}

	// Build map of all imported packages (including transitive deps)
	allPkgs := collectAllPackages(p.Pkgs)

	// Extract info for each external function
	result := make(map[string]*ExternalFunc)
	for id := range externalIDs {
		if ext := extractExternalFunc(id, allPkgs); ext != nil {
			result[id] = ext
		}
	}

	return result
}

// packageFromID extracts the package path from a function ID.
// e.g., "fmt.Println" -> "fmt", "github.com/foo/bar.Baz" -> "github.com/foo/bar"
func packageFromID(id string) string {
	// Handle method IDs like "pkg.(*Type).Method"
	if idx := strings.Index(id, ".("); idx != -1 {
		return id[:idx]
	}
	// Handle function IDs like "pkg.Func"
	if idx := strings.LastIndex(id, "."); idx != -1 {
		return id[:idx]
	}
	return ""
}

// collectAllPackages traverses imports to collect all packages.
func collectAllPackages(pkgs []*packages.Package) map[string]*packages.Package {
	result := make(map[string]*packages.Package)
	var visit func(pkg *packages.Package)
	visit = func(pkg *packages.Package) {
		if _, seen := result[pkg.PkgPath]; seen {
			return
		}
		result[pkg.PkgPath] = pkg
		for _, imp := range pkg.Imports {
			visit(imp)
		}
	}
	for _, pkg := range pkgs {
		visit(pkg)
	}
	return result
}

// extractExternalFunc extracts function info from loaded packages.
func extractExternalFunc(id string, allPkgs map[string]*packages.Package) *ExternalFunc {
	pkgPath := packageFromID(id)
	pkg, ok := allPkgs[pkgPath]
	if !ok || pkg.Types == nil {
		return nil
	}

	// Parse the function/method name from the ID
	name, recv := parseFuncID(id, pkgPath)
	if name == "" {
		return nil
	}

	ext := &ExternalFunc{
		Package: pkgPath,
		Name:    name,
	}

	// Look up the function in the package's type info
	if recv != "" {
		// Method lookup
		ext.Signature, ext.Godoc = lookupMethod(pkg, recv, name)
	} else {
		// Function lookup
		ext.Signature, ext.Godoc = lookupFunc(pkg, name)
	}

	if ext.Signature == "" {
		return nil
	}

	return ext
}

// parseFuncID parses a function ID into name and optional receiver type.
// Returns (name, receiver) where receiver is empty for plain functions.
func parseFuncID(id, pkgPath string) (name, recv string) {
	suffix := strings.TrimPrefix(id, pkgPath+".")

	// Check for method: (*Type).Method or (Type).Method
	if strings.HasPrefix(suffix, "(") {
		if idx := strings.Index(suffix, ")."); idx != -1 {
			recv = suffix[1:idx]
			name = suffix[idx+2:]
			return name, recv
		}
	}

	// Plain function
	return suffix, ""
}

// lookupFunc finds a function's signature and godoc.
func lookupFunc(pkg *packages.Package, name string) (sig, godoc string) {
	obj := pkg.Types.Scope().Lookup(name)
	if obj == nil {
		return "", ""
	}
	fn, ok := obj.(*types.Func)
	if !ok {
		return "", ""
	}

	sig = fn.Type().String()
	godoc = findGodoc(pkg, name, "")
	return sig, godoc
}

// lookupMethod finds a method's signature and godoc.
func lookupMethod(pkg *packages.Package, recv, name string) (sig, godoc string) {
	// Remove pointer prefix for type lookup
	typeName := strings.TrimPrefix(recv, "*")

	obj := pkg.Types.Scope().Lookup(typeName)
	if obj == nil {
		return "", ""
	}

	named, ok := obj.Type().(*types.Named)
	if !ok {
		return "", ""
	}

	// Check methods on the named type
	for i := 0; i < named.NumMethods(); i++ {
		m := named.Method(i)
		if m.Name() == name {
			sig = fmt.Sprintf("func (%s) %s%s", recv, name, strings.TrimPrefix(m.Type().String(), "func"))
			godoc = findGodoc(pkg, name, typeName)
			return sig, godoc
		}
	}

	// Check methods on pointer to named type
	ptr := types.NewPointer(named)
	mset := types.NewMethodSet(ptr)
	for i := 0; i < mset.Len(); i++ {
		sel := mset.At(i)
		if sel.Obj().Name() == name {
			sig = fmt.Sprintf("func (%s) %s%s", recv, name, strings.TrimPrefix(sel.Type().String(), "func"))
			godoc = findGodoc(pkg, name, typeName)
			return sig, godoc
		}
	}

	return "", ""
}

// findGodoc extracts documentation for a function or method.
func findGodoc(pkg *packages.Package, funcName, typeName string) string {
	for _, file := range pkg.Syntax {
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}

			if typeName != "" {
				// Looking for a method
				if fn.Recv == nil || len(fn.Recv.List) == 0 {
					continue
				}
				recvType := recvTypeName(fn.Recv.List[0].Type)
				if recvType != typeName {
					continue
				}
			} else {
				// Looking for a function
				if fn.Recv != nil {
					continue
				}
			}

			if fn.Name.Name == funcName && fn.Doc != nil {
				return fn.Doc.Text()
			}
		}
	}
	return ""
}

// recvTypeName extracts the type name from a receiver expression.
func recvTypeName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name
		}
	}
	return ""
}
