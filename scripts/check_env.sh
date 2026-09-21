#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# check_env.sh — validate the interactive environment against what the services
# expect (#54)
#
# Run this before make_venvs.sh, before any manual training/eval command, and
# whenever something works in a systemd unit but not in your shell.
#
# What it checks:
#   1. effective TMPDIR / HF_HOME and their filesystem types  (network = FAIL)
#   2. whether ~/.cache/huggingface/hub is on the research share
#   3. free space on / and the share
#   4. what atr-train.service actually sets, vs. what the shell carries
#   5. ~/.bashrc TMPDIR override that would shadow the service's TMPDIR
#   6. the API keys: present, non-empty, and the gate's matching the gateway's
#
# Exit 0 = clean.  Exit 1 = problem detected (explainers printed).
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail

FAILURES=0
WARNINGS=0

print_header() {
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "══════════════════════════════════════════════════════════════"
}

pass()  { echo "  ✓  $1"; }
fail()  { echo "  ✗  $1" >&2; FAILURES=$((FAILURES + 1)); }
warn()  { echo "  ⚠  $1"; WARNINGS=$((WARNINGS + 1)); }
info()  { echo "      $1"; }

fingerprint() {  # 8 hex chars — enough to compare two values without revealing either
  # sha256sum is coreutils (the box); shasum is what macOS ships. An empty
  # fingerprint would quietly defeat the point of comparing without printing.
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | cut -c1-8
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "$1" | shasum -a 256 | cut -c1-8
  else
    echo "no-sha256"
  fi
}

# ── 1. Variables and their backing filesystems ───────────────────────────────

print_header "1 — Environment variables and backing filesystems"

# stat -f -c %T : filesystem type (-f) on Linux; works for both local and network fs
fs_type() {
  stat -f -c %T "$1" 2>/dev/null || echo "unknown"
}

# The share on asterAIx reports **smb2**, not "smb", so every check that compared
# against the literal string passed it as local — including the TMPDIR one, which
# exists because a network TMPDIR broke compile (0ca2379). Globs, not equality,
# and one function so a new mount type is fixed in one place.
is_network_fs() {
  case "$1" in
    nfs*|cifs*|smb*|fuse.sshfs|fuseblk|9p|afs|glusterfs|ceph) return 0 ;;
    *) return 1 ;;
  esac
}

# TMPDIR
TMPDIR_VAL="${TMPDIR:-$(mktemp -u)}"
TMPDIR_FS=$(fs_type "$TMPDIR_VAL")
echo "  TMPDIR      = ${TMPDIR_VAL}"
echo "  TMPDIR fs   = ${TMPDIR_FS}"

if is_network_fs "$TMPDIR_FS"; then
  fail "TMPDIR is on a network filesystem (${TMPDIR_FS}) — the trainer refuses this"
  info "  ketos compile dies with 'Directory not empty' on network TMPDIR, and pip"
  info "  upgrades silently fail with EPERM.  Fix:"
  info "    mkdir -p ~/atr-cache/tmp"
  info "    echo 'TMPDIR=~/atr-cache/tmp' >> ~/.bashrc"
  info "    # then open a new shell"
elif [[ "$TMPDIR_FS" == "unknown" ]]; then
  warn "TMPDIR ${TMPDIR_VAL} does not exist yet — mktemp will use it as prefix"
  info "  it will be created on first use; make sure it is on a local fs"
else
  pass "TMPDIR is on local filesystem (${TMPDIR_FS})"
fi

# datasets Arrow generation cache (#60)
# Distinct from the hub cache, and the distinction is what an 11.5-hour failure
# turned on: hub/ stores downloaded files and is fine on the share, while
# datasets/ is written by pyarrow as a long streaming write, which SMB cannot
# hold a handle open for.
if [ -n "${HF_DATASETS_CACHE:-}" ]; then
  DS_CACHE="$HF_DATASETS_CACHE"
elif [ -n "${HF_HOME:-}" ]; then
  DS_CACHE="${HF_HOME}/datasets"
else
  DS_CACHE="${HOME}/.cache/huggingface/datasets"
fi
DS_RESOLVED=$(readlink -f "$DS_CACHE" 2>/dev/null || echo "$DS_CACHE")
DS_FS=$(fs_type "$DS_RESOLVED")
echo "  datasets cache = ${DS_CACHE}"
[ "$DS_RESOLVED" != "$DS_CACHE" ] && echo "  resolves to    = ${DS_RESOLVED}"
echo "  datasets fs    = ${DS_FS}"

if is_network_fs "$DS_FS"; then
  fail "the datasets Arrow cache is on a network filesystem (${DS_FS})"
  info "  pyarrow cannot hold a write handle open there for a generation pass:"
  info "  'ValueError: I/O operation on closed file' after 11.5 h, zero pages."
  info "  The trainer refuses a CACHED job in this state.  Fix:"
  info "    [ -L ~/.cache/huggingface/datasets ] && rm ~/.cache/huggingface/datasets"
  info "    mkdir -p ~/.cache/huggingface/datasets"
  info "  or stream instead:  ATR_TRAIN_CACHE_DATASETS=false"
else
  pass "datasets Arrow cache is on local filesystem (${DS_FS})"
fi

# HF_HOME
HF_HOME_VAL="${HF_HOME:-}"
HF_HOME_RESOLVED=""
if [ -n "$HF_HOME_VAL" ]; then
  HF_HOME_RESOLVED="$HF_HOME_VAL"
  HF_HOME_FS=$(fs_type "$HF_HOME_RESOLVED")
  echo "  HF_HOME     = ${HF_HOME_VAL}"
  echo "  HF_HOME fs  = ${HF_HOME_FS}"
  if is_network_fs "$HF_HOME_FS"; then
    warn "HF_HOME is on a network filesystem (${HF_HOME_FS}) — downloads will be slow"
  fi
else
  echo "  HF_HOME     = (not set — good; will use ~/.cache/huggingface)"
fi

# HF hub cache — the symlink that matters
HF_HUB_LINK="$HOME/.cache/huggingface/hub"
echo "  ~/.cache/huggingface/hub → $(readlink -f "$HF_HUB_LINK" 2>/dev/null || echo '(broken or missing)')"
HF_HUB_FS=$(fs_type "$HF_HUB_LINK" 2>/dev/null)
if is_network_fs "$HF_HUB_FS"; then
  warn "hub cache resolves to a network filesystem (${HF_HUB_FS}) — downloads go there"
  info "  the service deliberately avoids HF_HOME to prevent this; if you set it"
  info "  in ~/.bashrc, your interactive commands will re-download what the service"
  info "  already cached (and fill the share with 16 GB per base model)"
elif [ ! -e "$HF_HUB_LINK" ]; then
  warn "hub cache symlink does not exist yet"
else
  pass "hub cache is on ${HF_HUB_FS}"
fi

# ── 2. Free disk space ───────────────────────────────────────────────────────

print_header "2 — Free disk space"

check_space() {
  local path="$1"
  local label="${2:-${path}}"
  if df -k "$path" >/dev/null 2>&1; then
    local avail_k
    avail_k=$(df -k --output=avail "$path" 2>/dev/null | tail -1)
    local avail_g=$((avail_k / 1024 / 1024))
    echo "  ${label}: ${avail_g} GB available"
    if [ "$avail_k" -lt 524288 ]; then   # < 512 MB
      fail "${label} has < 512 MB free — installs and compiles will fail"
    elif [ "$avail_k" -lt 5242880 ]; then # < 5 GB
      warn "${label} has < 5 GB free — ensure TMPDIR is local (see §1)"
    else
      pass "${label} has ${avail_g} GB free"
    fi
  else
    warn "cannot check ${label} (path not accessible)"
  fi
}

check_space "/"
check_space "/tmp"
check_space "${TMPDIR_VAL}"

SHARE_MOUNT=""
for cand in "/mnt/wbkolleg_dh_1" "/mnt/research" "/mnt/share"; do
  if [ -d "$cand" ]; then
    SHARE_MOUNT="$cand"
    break
  fi
done
if [ -n "$SHARE_MOUNT" ]; then
  check_space "$SHARE_MOUNT"
fi

# ── 3. Systemd unit environment vs. shell ────────────────────────────────────

print_header "3 — atr-train.service Environment vs. current shell"

UNIT_MAINPID=$(systemctl --user show atr-train -p MainPID --value --no-pager 2>/dev/null || echo "")
if [ -n "$UNIT_MAINPID" ] && [ "$UNIT_MAINPID" != "0" ] && [ -r "/proc/${UNIT_MAINPID}/environ" ]; then
  # The running process's own environment: Environment=, EnvironmentFile= and
  # anything set-environment put there, which is the only view that matches what
  # the service actually sees.
  UNIT_VARS=$(tr '\0' '\n' < "/proc/${UNIT_MAINPID}/environ" | grep -E '^(TMPDIR|HF_HOME|CUDA_VISIBLE_DEVICES|PYTHONPATH|ATR_)' | tr '\n' ' ')
else
  UNIT_VARS=$(systemctl --user show atr-train -p Environment --no-pager 2>/dev/null || echo "")
fi
if [ -z "$UNIT_VARS" ]; then
  warn "cannot read atr-train.service environment (systemctl failed — probably not on asterAIx)"
  info "  run this script on the box where the service is installed"
else
  # Secrets are shown as a fingerprint, never a value. Reading the process
  # environment means ATR_*_KEY is in scope here, and this script's output gets
  # pasted into issues — the same rule as section 5.
  echo "  Unit sets:"
  echo "$UNIT_VARS" | tr ' ' '\n' | while IFS= read -r pair; do
    [ -n "$pair" ] || continue
    case "${pair%%=*}" in
      *KEY|*TOKEN|*SECRET|*PASSWORD)
        echo "    ${pair%%=*}=<set, fingerprint $(fingerprint "${pair#*=}")>" ;;
      *) echo "    ${pair}" ;;
    esac
  done

  # Check TMPDIR
  UNIT_TMPDIR=$(echo "$UNIT_VARS" | tr ' ' '\n' | grep '^TMPDIR=' | cut -d= -f2-)
  if [ -n "$UNIT_TMPDIR" ]; then
    echo ""
    info "  Unit TMPDIR  = ${UNIT_TMPDIR}"
    info "  Shell TMPDIR = ${TMPDIR_VAL}"
    if [ "$UNIT_TMPDIR" != "$TMPDIR_VAL" ]; then
      if is_network_fs "$TMPDIR_FS"; then
        fail "Shell TMPDIR differs from unit AND is on a network fs — unit is right, fix your shell"
      else
        warn "Shell TMPDIR differs from unit — if you run commands by hand while"
        warn "  the service is running, they see different temp space"
      fi
    else
      pass "Shell TMPDIR matches unit"
    fi
  fi

  # Check HF_HOME — unit must NOT have it
  if echo "$UNIT_VARS" | tr ' ' '\n' | grep -q '^HF_HOME='; then
    fail "Unit sets HF_HOME — it must NOT, to avoid filling the share with downloads"
  else
    pass "Unit does NOT set HF_HOME (correct)"
  fi
fi

# ── 4. Shell profile drift ───────────────────────────────────────────────────

print_header "4 — Shell profile drift (TMPDIR override in ~/.bashrc)"

BASHRC_TMPDIR=$(grep -v '^#' "$HOME/.bashrc" 2>/dev/null | grep 'TMPDIR=' | grep -v 'export TMPDIR=' | head -1)
if [ -n "$BASHRC_TMPDIR" ]; then
  warn "~/.bashrc sets TMPDIR without 'export': ${BASHRC_TMPDIR}"
  info "  A systemd unit loads .env, not .bashrc — if your TMPDIR is in .bashrc"
  info "  but the unit's TMPDIR is in .env, they differ and the unit is right."
  info "  Move the setting to .env so both the unit and interactive shell agree:"
  info "    echo 'TMPDIR=~/atr-cache/tmp' >> ~/.env"
  info "    sed -i '/TMPDIR=/d' ~/.bashrc"
  info "    source ~/.bashrc"
else
  pass "~/.bashrc does not override TMPDIR"
fi

# Also check for HF_HOME in bashrc
BASHRC_HFHOME=$(grep -v '^#' "$HOME/.bashrc" 2>/dev/null | grep 'HF_HOME=' | head -1)
if [ -n "$BASHRC_HFHOME" ]; then
  fail "~/.bashrc sets HF_HOME: ${BASHRC_HFHOME}"
  info "  This causes the interactive shell to re-download base models to the share"
  info "  instead of using the service's cache.  Remove it:"
  info "    sed -i '/HF_HOME=/d' ~/.bashrc"
  info "    source ~/.bashrc"
else
  pass "~/.bashrc does not set HF_HOME"
fi

# ── 5. API keys ──────────────────────────────────────────────────────────────

print_header "5 — API keys (gateway + promotion gate)"

# Values are NEVER printed. This script's output gets pasted into issues; a key
# that leaks that way has to be rotated everywhere it is configured.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"

env_value() {  # $1 = variable name; empty string when absent
  [ -f "$ENV_FILE" ] || { echo ""; return; }
  grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2-
}

ENV_API_KEY=$(env_value ATR_API_KEY)
ENV_GATE_KEY=$(env_value ATR_TRAIN_GATEWAY_API_KEY)

if [ ! -f "$ENV_FILE" ]; then
  fail "no .env at ${ENV_FILE} — the units load their configuration from it"
elif [ -z "$ENV_API_KEY" ]; then
  fail ".env has no ATR_API_KEY — the gateway will refuse every authenticated route"
else
  pass ".env sets ATR_API_KEY (fingerprint $(fingerprint "$ENV_API_KEY"))"

  # The promotion gate (#36) posts a held-out page to the gateway's /ocr. With no
  # key it authenticates as nobody, every model registers disabled with that as
  # the reason, and nothing is ever promoted into /models. Nothing fails loudly:
  # the job completes, the record says "not promoted", and the cause is a blank
  # variable set weeks earlier.
  if [ -z "$ENV_GATE_KEY" ]; then
    fail ".env has no ATR_TRAIN_GATEWAY_API_KEY — the promotion gate cannot authenticate"
    info "  Trained models will register disabled and never be promoted, silently."
    info "    echo \"ATR_TRAIN_GATEWAY_API_KEY=\$(grep ^ATR_API_KEY= ${ENV_FILE} | cut -d= -f2-)\" >> ${ENV_FILE}"
    info "    systemctl --user restart atr-train"
  elif [ "$ENV_GATE_KEY" != "$ENV_API_KEY" ]; then
    fail "ATR_TRAIN_GATEWAY_API_KEY does not match ATR_API_KEY"
    info "  gate    fingerprint $(fingerprint "$ENV_GATE_KEY")"
    info "  gateway fingerprint $(fingerprint "$ENV_API_KEY")"
    info "  The gate posts to the gateway, so it must present the gateway's key."
  else
    pass "ATR_TRAIN_GATEWAY_API_KEY matches ATR_API_KEY"
  fi
fi

# The shell's copy. This is the one that bit on 2026-08-11: ATR_API_KEY was unset
# interactively, /health still answered (it is public), /models returned
# {"detail":"missing or invalid X-API-Key"} which jq read as null, and a gate key
# exported from the empty variable would have been empty too.
SHELL_API_KEY="${ATR_API_KEY:-}"
if [ -z "$SHELL_API_KEY" ]; then
  warn "ATR_API_KEY is not set in this shell — authenticated curl calls will 401"
  info "  /health is public and answers anyway, so the shell looks fine until"
  info "  /models returns null. Load it from .env before running anything by hand:"
  info "    export ATR_API_KEY=\$(grep ^ATR_API_KEY= ${ENV_FILE} | cut -d= -f2-)"
elif [ -n "$ENV_API_KEY" ] && [ "$SHELL_API_KEY" != "$ENV_API_KEY" ]; then
  warn "shell ATR_API_KEY differs from .env ($(fingerprint "$SHELL_API_KEY") vs $(fingerprint "$ENV_API_KEY"))"
  info "  Commands you run by hand are talking to the gateway as someone else"
  info "  than the services are."
else
  pass "shell ATR_API_KEY matches .env"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

print_header "Summary"
echo ""
if [ "$FAILURES" -gt 0 ]; then
  echo "  ✗  ${FAILURES} failure(s) — MUST fix before training"
elif [ "$WARNINGS" -gt 0 ]; then
  echo "  ⚠  ${WARNINGS} warning(s) — review before production training"
else
  echo "  ✓  Environment is clean"
fi

echo ""
echo "Most common fix:"
echo "    mkdir -p ~/atr-cache/tmp"
echo "    echo 'TMPDIR=~/atr-cache/tmp' >> ~/.env"
echo "    sed -i '/TMPDIR=/d' ~/.bashrc"
echo "    source ~/.bashrc"

if [ "$FAILURES" -gt 0 ]; then
  exit 1
fi
exit 0