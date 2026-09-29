#!/usr/bin/env bash
# W3C conformance for everything the web surfaces and the docs ship: the serve
# page exactly as server.zig assembles it, the WASM shell page, both compiled
# stylesheets, and the brand SVGs. vnu is the pinned vnu-jar dev dependency and
# needs a Java runtime. Zero errors and zero warnings, or this fails.
# Run from scripts/lint-web.sh (zig build lint-web, CI lint-web).
set -euo pipefail
export LC_ALL=C TZ=UTC

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VNU_JAR="$ROOT/node_modules/vnu-jar/build/dist/vnu.jar"
if [[ ! -f "$VNU_JAR" ]]; then
    echo "check-w3c: $VNU_JAR missing; run bun install --frozen-lockfile" >&2
    exit 1
fi
if ! command -v java >/dev/null 2>&1; then
    echo "check-w3c: java not on PATH (vnu needs a Java 11+ runtime)" >&2
    exit 1
fi

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

# Same concatenation as server.zig html_page.
{
    cat src/web/head.html src/web/style.css src/web/body.html src/web/app.js
    printf '\n</script></body></html>\n'
} > "$SCRATCH/serve.html"

vnu() { java -jar "$VNU_JAR" --format text --Werror "$@"; }

vnu "$SCRATCH/serve.html" web/index.html
vnu --css src/web/style.css web/style.css
vnu --svg docs/brand/*.svg
echo "check-w3c: serve page, shell page, stylesheets and brand SVGs conform"
