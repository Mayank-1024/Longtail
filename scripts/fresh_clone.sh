#!/usr/bin/env bash
# C10 fresh-clone test: clone, uv sync, tests, download, replay demo, then the checks of
# docs/submission-checklist.md that a script can make. Run it on a clean machine or container,
# or on a laptop (it clears every LONGTAIL_* variable and any key, and works in a new directory).
#
#   scripts/fresh_clone.sh                      # main of the GitHub repo, with one scene downloaded
#   scripts/fresh_clone.sh --no-download        # skip the 0.44 GB scene and the real-data tests
#   scripts/fresh_clone.sh --data-dir ~/longtail-data   # real-data tests on scenes already on disk
#   scripts/fresh_clone.sh --ref c/branch --dir /tmp/lt-clone --repo <url>
#
# Needs git, uv and network access; gh (optional) for the repo-visibility check. Makes no paid calls:
# no key is passed in, and LONGTAIL_ALLOW_PAID stays unset. Exit status 1 if any step FAILs or is MISSING.

set -u

REPO="https://github.com/Mayank-1024/longtail.git"
REF="main"
DIR=""
DOWNLOAD=1
DATA_DIR=""
SCENE="001"
DEMO_TIMEOUT_S=600

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --dir) DIR="$2"; shift 2 ;;
    --scene) SCENE="$2"; shift 2 ;;
    --no-download) DOWNLOAD=0; shift ;;
    --data-dir) DATA_DIR="$2"; DOWNLOAD=0; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

DIR="${DIR:-$(mktemp -d "${TMPDIR:-/tmp}/longtail-fresh.XXXXXX")}"
if [ -e "$DIR/repo" ]; then
  echo "$DIR/repo already exists; pass a new --dir" >&2
  exit 2
fi
LOG="$DIR/logs"
mkdir -p "$LOG"
RESULTS=()

record() { # status, step, detail
  RESULTS+=("$1|$2|$3")
  printf '%-8s %-28s %s\n' "$1" "$2" "$3"
}

# A clean environment: no LONGTAIL_* settings, no keys, data and caches inside the clone's directory.
clean() {
  env -i HOME="$HOME" PATH="$PATH" TMPDIR="${TMPDIR:-/tmp}" LANG="${LANG:-C.UTF-8}" \
    LONGTAIL_DATA_DIR="${DATA_DIR:-$DIR/data}" LONGTAIL_CACHE_DIR="$DIR/cache" "$@"
}

# step name, log file, command...: PASS on exit 0, FAIL otherwise (last log lines shown)
run() {
  local name="$1" log="$LOG/$2"; shift 2
  if (cd "$DIR/repo" && clean "$@") >"$log" 2>&1; then
    record PASS "$name" "$(tail -n 1 "$log" | cut -c1-90)"
  else
    record FAIL "$name" "see $log"
    tail -n 5 "$log" | sed 's/^/         | /'
  fi
}

with_timeout() { # seconds, command...  (macOS has no timeout(1))
  perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$@"
}

echo "fresh clone of $REPO ($REF) in $DIR"

# 1. Clone (full history, so the secrets check covers every commit)
if git clone -q --branch "$REF" "$REPO" "$DIR/repo" >"$LOG/clone.log" 2>&1; then
  record PASS clone "$(git -C "$DIR/repo" log --oneline -1 | cut -c1-90)"
else
  record FAIL clone "see $LOG/clone.log"
  exit 1
fi

# 2. Install and test, synthetic data only (the README's "Run it")
run "uv sync --frozen" sync.log uv sync --frozen
run "ruff check" ruff.log uv run --no-sync ruff check .
# an empty data dir, so the real-data tests skip here even with --data-dir
run "pytest (synthetic)" pytest_synthetic.log env LONGTAIL_DATA_DIR="$DIR/no-data" uv run --no-sync pytest -q

# 3. Download one scene and run the real-data tests on it
if [ "$DOWNLOAD" = 1 ]; then
  run "download scene $SCENE" download.log \
    uv run --no-sync python scripts/fetch_pandaset.py --parts full --scenes "$SCENE"
  run "pytest (scene $SCENE)" pytest_real.log \
    uv run --no-sync pytest -q tests/test_pandaset_adapter.py tests/test_projection_real.py
elif [ -n "$DATA_DIR" ]; then
  record SKIP "download scene $SCENE" "using $DATA_DIR"
  run "pytest (scenes on disk)" pytest_real.log \
    uv run --no-sync pytest -q tests/test_pandaset_adapter.py tests/test_projection_real.py
else
  record SKIP "download scene $SCENE" "--no-download"
fi

# 4. The replay demo (plan C10: longtail demo --replay): no key, every model call from the cache
if (cd "$DIR/repo" && clean uv run --no-sync longtail demo --help) >"$LOG/demo_help.log" 2>&1; then
  if (cd "$DIR/repo" && clean with_timeout "$DEMO_TIMEOUT_S" uv run --no-sync longtail demo --replay) \
    >"$LOG/demo.log" 2>&1; then
    record PASS "longtail demo --replay" "$(tail -n 1 "$LOG/demo.log" | cut -c1-90)"
  else
    record FAIL "longtail demo --replay" "see $LOG/demo.log"
  fi
else
  record MISSING "longtail demo --replay" "no 'demo' subcommand in this commit"
fi

# 5. Rules checklist: the parts a script can check (docs/submission-checklist.md has the rest)
cd "$DIR/repo" || exit 1

tracked_secrets=$(git ls-files | grep -E '(^|/)\.env($|\.)' | grep -v '\.env\.example$')
[ -z "$tracked_secrets" ] && record PASS "no .env tracked" "" || record FAIL "no .env tracked" "$tracked_secrets"

if command -v gitleaks >/dev/null 2>&1; then
  if gitleaks detect --source . --no-banner --redact >"$LOG/gitleaks.log" 2>&1; then
    record PASS "gitleaks (all history)" ""
  else
    record FAIL "gitleaks (all history)" "see $LOG/gitleaks.log"
  fi
else
  record SKIP "gitleaks (all history)" "gitleaks not installed; CI's secrets job covers pushes"
fi

media=$(git ls-files | grep -iE '\.(jpe?g|png|mp4|mov|pcd|pkl|pkl\.gz|npz|bin)$' | grep -v '^web/public/' | head -5)
data_dirs=$(git ls-files | grep -E '^(data|cache|runs|gt)/' | head -5)
if [ -z "$media$data_dirs" ]; then
  record PASS "no data or media tracked" ""
else
  record FAIL "no data or media tracked" "$(echo $media $data_dirs | cut -c1-90)"
fi

if head -n 3 LICENSE 2>/dev/null | grep -q "Apache License"; then
  record PASS "LICENSE (Apache-2.0)" ""
else
  record FAIL "LICENSE (Apache-2.0)" "missing or not Apache-2.0"
fi

grep -q "CC BY 4.0" README.md && record PASS "README: PandaSet credit" "CC BY 4.0" \
  || record FAIL "README: PandaSet credit" "no CC BY 4.0 attribution"
# A-109: SAM is used under Meta's SAM License, whose terms require acknowledging SAM
grep -qiE "SAM License|meta-sam-parser" README.md && record PASS "README: SAM acknowledgement" "" \
  || record FAIL "README: SAM acknowledgement" "no mention of the SAM License (A-109)"

for tag in eval-freeze eval-final; do
  git rev-parse -q --verify "refs/tags/$tag" >/dev/null && record PASS "tag $tag" "" \
    || record FAIL "tag $tag" "not in the repo (plan section 10)"
done

slug=$(echo "$REPO" | sed -E 's#(https://github.com/|git@github.com:)##; s#\.git$##')
if command -v gh >/dev/null 2>&1 && vis=$(gh repo view "$slug" --json visibility -q .visibility 2>/dev/null); then
  [ "$vis" = "PUBLIC" ] && record PASS "repo is public" "" || record FAIL "repo is public" "visibility is $vis"
else
  record SKIP "repo is public" "gh not available or not logged in"
fi

echo
fails=$(printf '%s\n' "${RESULTS[@]}" | grep -cE '^(FAIL|MISSING)\|')
echo "$(printf '%s\n' "${RESULTS[@]}" | grep -c '^PASS|') passed, $fails failed or missing; logs in $LOG"
echo "Video length, the submission form and the rest are manual: docs/submission-checklist.md"
[ "$fails" -eq 0 ]
