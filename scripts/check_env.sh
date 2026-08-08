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

# ── 1. Variables and their backing filesystems ───────────────────────────────

print_header "1 — Environment variables and backing filesystems"

# stat -f -c %T : filesystem type (-f) on Linux; works for both local and network fs
fs_type() {
  stat -f -c %T "$1" 2>/dev/null || echo "unknown"
}

# TMPDIR
TMPDIR_VAL="${TMPDIR:-$(mktemp -u)}"
TMPDIR_FS=$(fs_type "$TMPDIR_VAL")
echo "  TMPDIR      = ${TMPDIR_VAL}"
echo "  TMPDIR fs   = ${TMPDIR_FS}"

if [[ "$TMPDIR_FS" == "nfs" || "$TMPDIR_FS" == "cifs" || "$TMPDIR_FS" == "smb" ]]; then
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

# HF_HOME
HF_HOME_VAL="${HF_HOME:-}"
HF_HOME_RESOLVED=""
if [ -n "$HF_HOME_VAL" ]; then
  HF_HOME_RESOLVED="$HF_HOME_VAL"
  HF_HOME_FS=$(fs_type "$HF_HOME_RESOLVED")
  echo "  HF_HOME     = ${HF_HOME_VAL}"
  echo "  HF_HOME fs  = ${HF_HOME_FS}"
  if [[ "$HF_HOME_FS" == "nfs" || "$HF_HOME_FS" == "cifs" || "$HF_HOME_FS" == "smb" ]]; then
    warn "HF_HOME is on a network filesystem (${HF_HOME_FS}) — downloads will be slow"
  fi
else
  echo "  HF_HOME     = (not set — good; will use ~/.cache/huggingface)"
fi

# HF hub cache — the symlink that matters
HF_HUB_LINK="$HOME/.cache/huggingface/hub"
echo "  ~/.cache/huggingface/hub → $(readlink -f "$HF_HUB_LINK" 2>/dev/null || echo '(broken or missing)')"
HF_HUB_FS=$(fs_type "$HF_HUB_LINK" 2>/dev/null)
if [[ "$HF_HUB_FS" == "nfs" || "$HF_HUB_FS" == "cifs" || "$HF_HUB_FS" == "smb" ]]; then
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

UNIT_VARS=$(systemctl --user show atr-train -p Environment --no-pager 2>/dev/null || echo "")
if [ -z "$UNIT_VARS" ]; then
  warn "cannot read atr-train.service environment (systemctl failed — probably not on asterAIx)"
  info "  run this script on the box where the service is installed"
else
  echo "  Unit sets:"
  echo "$UNIT_VARS" | tr ' ' '\n' | sed 's/^/    /'

  # Check TMPDIR
  UNIT_TMPDIR=$(echo "$UNIT_VARS" | tr ' ' '\n' | grep '^TMPDIR=' | cut -d= -f2-)
  if [ -n "$UNIT_TMPDIR" ]; then
    echo ""
    info "  Unit TMPDIR  = ${UNIT_TMPDIR}"
    info "  Shell TMPDIR = ${TMPDIR_VAL}"
    if [ "$UNIT_TMPDIR" != "$TMPDIR_VAL" ]; then
      if [[ "$TMPDIR_FS" == "nfs" || "$TMPDIR_FS" == "cifs" || "$TMP_FS" == "smb" ]]; then
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