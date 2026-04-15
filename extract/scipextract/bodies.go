package scipextract

import (
	"fmt"
	"os"
	"strings"

	"github.com/scip-code/scip/bindings/go/scip"
)

// sourceCache caches file contents keyed by absolute path.
type sourceCache struct {
	root  string
	files map[string][]byte
}

func newSourceCache(root string) *sourceCache {
	return &sourceCache{root: root, files: map[string][]byte{}}
}

func (c *sourceCache) get(absPath string) []byte {
	if content, ok := c.files[absPath]; ok {
		return content
	}
	content, err := os.ReadFile(absPath)
	if err != nil {
		c.files[absPath] = nil
		return nil
	}
	c.files[absPath] = content
	return content
}

// extractBody returns the source text for a function's body and an
// optional warning message describing any fallback that was taken.
// Resolution order:
//
//  1. The definition occurrence's EnclosingRange (the most accurate form).
//  2. The definition range alone, sliced from Document.Text if the indexer
//     embedded it.
//  3. The definition range alone, read from disk.
//
// Only case 1 reliably gives the full implementation. Cases 2 and 3 emit
// a warning because the LLM will only see the signature line(s).
func extractBody(info *scip.SymbolInformation, doc *scip.Document, absPath string, cache *sourceCache) (string, string) {
	defOcc := findDefinition(info.Symbol, doc)
	if defOcc == nil {
		return "", fmt.Sprintf("no definition occurrence for %s", info.Symbol)
	}

	if len(defOcc.EnclosingRange) > 0 {
		body, ok := sliceRangeFromSource(defOcc.EnclosingRange, doc, absPath, cache)
		if ok {
			return body, ""
		}
	}

	body, ok := sliceRangeFromSource(defOcc.Range, doc, absPath, cache)
	if !ok {
		return "", fmt.Sprintf("failed to read source for %s at %s", info.Symbol, absPath)
	}
	return body, fmt.Sprintf("no enclosing range for %s; body truncated to signature", info.Symbol)
}

// findDefinition returns the first occurrence of symbol in doc with the
// Definition role set.
func findDefinition(symbol string, doc *scip.Document) *scip.Occurrence {
	for _, occ := range doc.Occurrences {
		if occ.Symbol != symbol {
			continue
		}
		if occ.SymbolRoles&int32(scip.SymbolRole_Definition) != 0 {
			return occ
		}
	}
	return nil
}

// sliceRangeFromSource extracts the text covered by scipRange. It prefers
// Document.Text when the indexer embedded it; otherwise it reads from disk
// via the cache. Returns (text, true) on success.
func sliceRangeFromSource(scipRange []int32, doc *scip.Document, absPath string, cache *sourceCache) (string, bool) {
	r, err := scip.NewRange(scipRange)
	if err != nil {
		return "", false
	}

	if doc.Text != "" {
		if s, ok := sliceLines(doc.Text, int(r.Start.Line), int(r.End.Line)); ok {
			return s, true
		}
	}

	content := cache.get(absPath)
	if content == nil {
		return "", false
	}
	return sliceLines(string(content), int(r.Start.Line), int(r.End.Line))
}

// sliceLines returns the substring covering startLine..endLine inclusive,
// using 0-based line indices. endLine may equal startLine for single-line
// ranges. Returns (text, true) on success.
func sliceLines(text string, startLine, endLine int) (string, bool) {
	lines := strings.SplitAfter(text, "\n")
	if startLine < 0 || startLine >= len(lines) {
		return "", false
	}
	if endLine >= len(lines) {
		endLine = len(lines) - 1
	}
	if endLine < startLine {
		endLine = startLine
	}
	var b strings.Builder
	for i := startLine; i <= endLine; i++ {
		b.WriteString(lines[i])
	}
	return strings.TrimRight(b.String(), "\n"), true
}
