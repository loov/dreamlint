// analyze/embed.go
package analyze

import "embed"

//go:embed prompts/*.txt
var embeddedPrompts embed.FS
