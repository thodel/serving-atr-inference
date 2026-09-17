"""The infrastructure docs are checked against the files they describe (#142).

A description of infrastructure goes stale without anyone noticing. A unit moves
to another port, a value becomes one that both machines must agree on, a diagram
loses its closing fence, and the prose still reads as well as before. Each test
here fails on one such drift, and its message names the fact that no longer
holds. The rendering itself is left to GitHub's preview; these are cheap probes
that need no network.
"""

from __future__ import annotations

import re
from pathlib import Path

from atr_serving.clients import TRAINER_CONNECT_TIMEOUT_S
from atr_serving.config import Settings
from atr_serving.manager import MEASURED_VRAM_BUDGET_MB
from atr_serving.shared_registry import RegistryWatch

REPO = Path(__file__).resolve().parents[1]
INFRA = REPO / "docs" / "INFRASTRUCTURE.md"
UNIT_DIR = REPO / "deploy" / "systemd"
ENV_EXAMPLE = REPO / ".env.example"

IDHEFIX_IP = "130.92.59.240"
ASTERAIX_IP = "130.92.59.242"

#: The docs #142 wrote or rewrote. docs/idhefix-environment.md is left out on
#: purpose: #142 changed only its title and banner, and its body still calls the
#: box asterAIx. Fixing that is #136's job; add the file here once #136 lands.
CHECKED_DOCS = ("docs/INFRASTRUCTURE.md", "README.md", "docs/DEPLOY.md", "docs/SPLIT_PLAN.md")

#: What may follow ```mermaid on the first line. GitHub renders more types; these
#: are the ones a doc in this repo has a use for.
MERMAID_TYPES = ("flowchart", "graph", "sequenceDiagram", "stateDiagram-v2", "stateDiagram",
                 "classDiagram", "erDiagram", "gantt", "pie", "timeline")

#: Directories whose Markdown is not this repository's documentation.
SKIPPED_DIRS = {".git", ".venv", ".venvs", "node_modules", ".pytest_cache", ".ruff_cache",
                ".claude", "__pycache__"}


# ── helpers ──────────────────────────────────────────────────────────────────
def _doc(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _repo_files(*suffixes: str) -> list[Path]:
    return sorted(p for p in REPO.rglob("*")
                  if p.is_file() and p.suffix in suffixes
                  and not SKIPPED_DIRS.intersection(p.relative_to(REPO).parts))


def _cells(row: str) -> list[str]:
    """The cells of a Markdown table row, split on unescaped pipes."""
    inner = row.strip()
    inner = inner[1:] if inner.startswith("|") else inner
    inner = inner[:-1] if inner.endswith("|") and not inner.endswith("\\|") else inner
    return [c.strip() for c in re.split(r"(?<!\\)\|", inner)]


def _is_separator(cells: list[str]) -> bool:
    return all(c and set(c) <= set("-: ") for c in cells)


def _outside_fences(text: str) -> list[str]:
    """The lines of ``text`` that are not inside a fenced block."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return out


def _tables(text: str) -> list[list[list[str]]]:
    """Every table outside fenced blocks: its rows as cells, header row first."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for line in _outside_fences(text):
        if not line.lstrip().startswith("|"):
            current = None
            continue
        cells = _cells(line)
        if current is None:
            current = []
            tables.append(current)
        if not _is_separator(cells):
            current.append(cells)
    return tables


def _table_rows(text: str) -> list[list[str]]:
    """Every data row of every table outside fenced blocks (headers left out)."""
    return [row for table in _tables(text) for row in table[1:]]


def _section(text: str, heading: str) -> str:
    """The body under ``## heading`` up to the next level-2 heading."""
    match = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"docs/INFRASTRUCTURE.md has no '## {heading}' section"
    return match.group(1)


def _exec_start(unit: Path) -> str:
    """A unit's ExecStart, with backslash continuations joined."""
    lines = [line for line in unit.read_text().replace("\\\n", " ").splitlines()
             if line.startswith("ExecStart=")]
    assert len(lines) == 1, f"{unit.name}: expected one ExecStart=, found {len(lines)}"
    return lines[0].removeprefix("ExecStart=")


def shared_entries(env_text: str) -> list[str]:
    """The variables a ``>>> SHARED <<<`` marker applies to, in order.

    A marker is a comment line that begins with it; the entry is the first
    variable assigned (or shown commented out) after it. A sentence that only
    mentions the marker ("Values marked >>> SHARED <<< …") is not one.
    """
    names: list[str] = []
    pending: int | None = None
    for number, line in enumerate(env_text.splitlines(), 1):
        if re.match(r"#\s*>>> SHARED <<<", line):
            assert pending is None, (
                f".env.example:{pending}: a >>> SHARED <<< marker with no variable before "
                f"the next marker at line {number}")
            pending = number
            continue
        assigned = re.match(r"#?\s*([A-Z][A-Z0-9_]*)=", line)
        if assigned and pending is not None:
            names.append(assigned.group(1))
            pending = None
    assert pending is None, f".env.example:{pending}: a >>> SHARED <<< marker with no variable after it"
    return names


_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_BLOCK_START = re.compile(r"(#{1,6} |[-*+] |\d+\. )")


def statements(text: str) -> list[str]:
    """What a reader takes in as one statement.

    A sentence of prose; a table cell, read together with its column's header
    and its row's first cell (``| IP | 130.92.59.240 |`` under an ``idhefix``
    column says "idhefix is 130.92.59.240"); a single line of a fenced block.
    """
    out: list[str] = []
    paragraph: list[str] = []
    header: list[str] | None = None
    fenced = False

    def flush() -> None:
        if paragraph:
            out.extend(_SENTENCE_END.split(" ".join(paragraph)))
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(("```", "~~~")):
            flush()
            fenced = not fenced
            continue
        if fenced:
            out.append(line)
            continue
        if line.startswith("|"):
            flush()
            cells = _cells(line)
            if header is None:
                header = cells
                out.extend(cells)
            elif not _is_separator(cells):
                out.extend(" ".join((cell, header[i] if i < len(header) else "", cells[0]))
                           for i, cell in enumerate(cells))
            continue
        header = None
        if line.startswith(">"):
            line = line.lstrip("> ").strip()
        if not line:
            flush()
            continue
        if _BLOCK_START.match(line):
            flush()
        paragraph.append(line)
    flush()
    return out


def misnamed_hosts(text: str) -> list[str]:
    """Statements that put an IP next to the other machine's name."""
    wrong = {IDHEFIX_IP: "asteraix", ASTERAIX_IP: "idhefix"}
    return [f"{ip} next to {name!r}: {st[:200]}"
            for st in statements(text)
            for ip, name in wrong.items()
            if ip in st and name in st.lower()]


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading (without the -1, -2 of duplicates)."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)   # a link keeps its text
    text = text.replace("`", "").strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def _anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    for line in _outside_fences(text):
        match = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        slug = github_slug(match.group(1))
        count = seen.get(slug, 0)
        anchors.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return anchors


def mermaid_blocks(text: str) -> tuple[list[tuple[int, list[str]]], list[str]]:
    """Every ```mermaid block as (line number, body lines), and what is broken.

    Broken: a fence that is never closed (it swallows the rest of the file, and
    GitHub then renders nothing after it the way the author meant), and a block
    whose first line is not a diagram type Mermaid knows. A missing closing fence
    rarely shows as an open fence at the end of the file: the next block's bare
    closing fence closes this one instead, and that block's opening line lands
    inside the diagram. So a fence line inside a mermaid block counts as broken.
    """
    blocks: list[tuple[int, list[str]]] = []
    problems: list[str] = []
    fence: tuple[int, str, bool] | None = None      # (line, marker, is mermaid)
    body: list[str] = []
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if fence is None:
            opened = re.match(r"(`{3,}|~{3,})\s*(\S*)", stripped)
            if opened:
                fence = (number, opened.group(1), opened.group(2) == "mermaid")
                body = []
            continue
        if stripped.startswith(fence[1]) and not stripped[len(fence[1]):].strip():
            if fence[2]:
                blocks.append((fence[0], body))
            fence = None
            continue
        body.append(line)
    if fence is not None:
        problems.append(f"line {fence[0]}: fence {fence[1]} is never closed")
    for start, lines in blocks:
        first = next((line.strip() for line in lines if line.strip()), "")
        if first.split(" ")[0] not in MERMAID_TYPES:
            problems.append(f"line {start}: mermaid block starts with {first!r}, "
                            f"not one of {MERMAID_TYPES}")
        for offset, line in enumerate(lines, 1):
            if line.strip().startswith(("```", "~~~")):
                problems.append(f"line {start + offset}: a fence inside the mermaid block "
                                f"opened at line {start}; that block is not closed")
    return blocks, problems


def mermaid_house_rules(lines: list[str]) -> list[str]:
    """What GitHub's renderer trips over, as this repo writes diagrams.

    Every flowchart label quoted; one edge per line (no ``&``); no HTML entities
    anywhere; no ``;`` and no brackets in sequence-diagram text, where ``;``
    ends a statement.
    """
    problems = []
    kind = next((line.strip().split(" ")[0] for line in lines if line.strip()), "")
    for line in lines:
        text = line.strip()
        if re.search(r"&[A-Za-z]+;|&#\d+;|#\d+;", text):
            problems.append(f"HTML entity in: {text}")
        if kind in ("flowchart", "graph"):
            bare = re.sub(r'"[^"]*"', '""', text)
            if re.search(r"[\[({]+[^\[\](){}\"]", bare):
                problems.append(f"unquoted label in: {text}")
            if "&" in bare:
                problems.append(f"'&' chains edges in: {text}")
        if kind == "sequenceDiagram":
            if ";" in text:
                problems.append(f"';' in a sequence diagram: {text}")
            if text.startswith(("participant", "actor")) and re.search(r"[()<>]", text):
                problems.append(f"bracket in an unquoted participant label: {text}")
    return problems


# ── the tests ────────────────────────────────────────────────────────────────
def test_every_unit_is_documented():
    """Every unit in deploy/systemd has a row in the services table, with the
    port, bind and venv its ExecStart really uses."""
    units = sorted(UNIT_DIR.glob("*.service"))
    assert units, f"no unit files in {UNIT_DIR}"
    rows = _table_rows(INFRA.read_text(encoding="utf-8"))
    problems = []
    for unit in units:
        command = _exec_start(unit)
        expected = []
        if port := re.search(r"--port[= ](\d+)", command):
            expected.append(("port", port.group(1)))
        if host := re.search(r"--host[= ](\S+)", command):
            expected.append(("bind", host.group(1)))
        if venv := re.search(r"/(\.venvs/[^/]+)/", command):
            expected.append(("venv", venv.group(1)))
        named = [r for r in rows if r and r[0].strip("` ") == unit.stem]
        if not named:
            problems.append(f"{unit.name}: no table row whose first cell is `{unit.stem}`")
            continue
        if not any(all(any(value == cell.strip("` ") or f"`{value}`" in cell for cell in row)
                       for _, value in expected)
                   for row in named):
            problems.append(f"{unit.name}: no `{unit.stem}` row shows "
                            + ", ".join(f"{what} {value}" for what, value in expected))
    assert not problems, "docs/INFRASTRUCTURE.md is out of step with deploy/systemd:\n" + \
        "\n".join(problems)


def test_every_shared_value_is_documented():
    """Every >>> SHARED <<< entry of .env.example is in the shared-values table,
    in the idhefix column, and every variable in that column is marked. The table
    and the markers are one list written twice. .env.example points at this
    section by anchor, and training-atr-models#16 will link its markers here, so
    the heading is pinned too."""
    names = shared_entries(ENV_EXAMPLE.read_text(encoding="utf-8"))
    assert names, ".env.example marks no value >>> SHARED <<<"
    tables = _tables(_section(INFRA.read_text(encoding="utf-8"), "Shared values"))
    assert tables, "docs/INFRASTRUCTURE.md#shared-values has no table"
    # The first table is the shared one; the asteraix-only settings follow it.
    first_cells = [row[0] for row in tables[0][1:]]
    documented = {n for cell in first_cells for n in re.findall(r"`(ATR_[A-Z0-9_]+)`", cell)}
    missing = sorted(set(names) - documented)
    assert not missing, (
        f"marked >>> SHARED <<< in .env.example but not in the idhefix column of "
        f"docs/INFRASTRUCTURE.md#shared-values: {missing}")
    unmarked = sorted(documented - set(names))
    assert not unmarked, (
        f"in the idhefix column of docs/INFRASTRUCTURE.md#shared-values but not marked "
        f">>> SHARED <<< in .env.example: {unmarked}")


def test_the_shared_marker_parser_reads_what_a_person_reads():
    text = (
        "# Values marked >>> SHARED <<< must agree.\n"
        "# >>> SHARED <<< with asteraix: explained here, ATR_OTHER mentioned\n"
        "# ATR_ONE=\n"
        "ATR_NOT_SHARED=1\n"
        "# >>> SHARED <<< again\n"
        "#\n"
        "ATR_TWO=x\n"
    )
    assert shared_entries(text) == ["ATR_ONE", "ATR_TWO"]


def test_the_docs_name_the_hosts_correctly():
    """130.92.59.240 is idhefix and 130.92.59.242 is asteraix (#136). Checked in
    the docs #142 touched, not repo-wide; the sweep is #136's."""
    problems = [f"{rel}: {p}" for rel in CHECKED_DOCS for p in misnamed_hosts(_doc(rel))]
    assert not problems, "\n".join(problems)


def test_the_host_check_sees_a_misnamed_host():
    assert misnamed_hosts("Topology: **asterAIx** (`srv`, `130.92.59.240`) runs this.")
    assert misnamed_hosts("| | asteraix |\n|---|---|\n| IP | 130.92.59.240 |")
    assert misnamed_hosts("The trainer on idhefix listens at\n130.92.59.242:8204.")
    # Its own name, or the other name in another statement, is fine.
    assert not misnamed_hosts("idhefix is 130.92.59.240. The trainer is on asteraix.")
    assert not misnamed_hosts("| | idhefix | asteraix |\n|---|---|---|\n"
                              "| IP | 130.92.59.240 | 130.92.59.242 |")


def test_every_mermaid_block_is_closed():
    """Every fence is closed and every mermaid block names a known diagram type;
    whether it renders is for the PR preview to show."""
    problems = []
    for path in _repo_files(".md"):
        _, broken = mermaid_blocks(path.read_text(encoding="utf-8"))
        problems += [f"{path.relative_to(REPO)}: {p}" for p in broken]
    assert not problems, "\n".join(problems)


def test_the_overview_has_its_diagrams():
    blocks, _ = mermaid_blocks(INFRA.read_text(encoding="utf-8"))
    kinds = [next(line.strip().split(" ")[0] for line in body if line.strip())
             for _, body in blocks]
    assert "flowchart" in kinds, "the system picture is gone from docs/INFRASTRUCTURE.md"
    assert "sequenceDiagram" in kinds, "the model's path is gone from docs/INFRASTRUCTURE.md"


def test_the_mermaid_blocks_keep_to_what_github_renders():
    problems = []
    for path in _repo_files(".md"):
        blocks, _ = mermaid_blocks(path.read_text(encoding="utf-8"))
        for start, body in blocks:
            problems += [f"{path.relative_to(REPO)}:{start}: {p}"
                         for p in mermaid_house_rules(body)]
    assert not problems, "\n".join(problems)


def test_the_mermaid_checks_see_breakage():
    _, broken = mermaid_blocks("text\n```mermaid\nflowchart LR\n  a --> b\n")
    assert broken and "never closed" in broken[0]
    _, broken = mermaid_blocks("```mermaid\nflowchat LR\n```\n")
    assert broken and "flowchat" in broken[0]
    _, broken = mermaid_blocks("```bash\necho\n```\n```mermaid\nsequenceDiagram\n```\n")
    assert not broken
    assert mermaid_house_rules(["flowchart LR", "  a[unquoted] --> b"])
    assert mermaid_house_rules(["flowchart LR", '  a["x"] & b["y"] --> c["z"]'])
    assert mermaid_house_rules(["sequenceDiagram", "  A->>B: one; two"])
    assert mermaid_house_rules(["flowchart LR", '  a["a &amp; b"]'])
    assert not mermaid_house_rules(["flowchart LR", '  a["x (y)"] -- "a label" --> b["z"]',
                                    '  s[("share")]'])


def test_nothing_links_to_the_old_host_doc_name():
    """docs/asteraix-environment.md is docs/idhefix-environment.md now (#142).
    Only the plan that ordered the rename, and the two docs that record it,
    may still say the old name, and none of them as a link."""
    old = "asteraix-environment.md"
    may_mention = {"docs/SPLIT_PLAN.md", "docs/INFRASTRUCTURE.md", "docs/idhefix-environment.md"}
    problems = []
    for path in _repo_files(".md", ".py", ".sh", ".txt", ".yaml", ".yml", ".def", ".toml",
                            ".json", ".example"):
        rel = path.relative_to(REPO).as_posix()
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if old not in text:
            continue
        if rel not in may_mention or re.search(rf"\]\([^)]*{re.escape(old)}", text):
            problems.append(rel)
    assert not problems, f"still pointing at {old}: {problems}"
    assert (REPO / "docs" / "idhefix-environment.md").is_file()
    assert not (REPO / "docs" / old).exists()


def test_the_relative_links_resolve():
    """Every relative link in the checked docs names a file that exists and,
    with a #fragment, a heading that exists — training-atr-models links here
    by anchor."""
    problems = []
    for rel in (*CHECKED_DOCS, "docs/idhefix-environment.md"):
        source = REPO / rel
        text = "\n".join(_outside_fences(source.read_text(encoding="utf-8")))
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+:", target):
                continue                                   # http:, https:, mailto:
            path_part, _, fragment = target.partition("#")
            resolved = (source.parent / path_part).resolve() if path_part else source
            if REPO not in resolved.parents and resolved != REPO:
                continue                                   # GitHub-relative, e.g. ../../issues
            if not resolved.exists():
                problems.append(f"{rel}: {target} does not exist")
                continue
            if fragment and resolved.suffix == ".md" \
                    and fragment not in _anchors(resolved.read_text(encoding="utf-8")):
                problems.append(f"{rel}: {target} names no heading in {resolved.name}")
    assert not problems, "\n".join(problems)


def test_github_slugs_are_computed_like_github():
    assert github_slug("How a trained model reaches /models") == \
        "how-a-trained-model-reaches-models"
    assert github_slug("idhefix: `~/Repo/serving-atr-inference`") == \
        "idhefix-reposerving-atr-inference"
    assert github_slug("8. Training is not on this box") == "8-training-is-not-on-this-box"
    assert github_slug("idhefix — serving") == "idhefix--serving"


def test_the_numbers_quoted_from_the_code_match_it():
    """The doc quotes defaults; a changed default must change the doc too."""
    text = " ".join(INFRA.read_text(encoding="utf-8").split())
    fields = Settings.model_fields
    quoted = {
        "vLLM port base": (r"(\d+) and up \(`ATR_VLLM_PORT_BASE`\)",
                           fields["vllm_port_base"].default),
        "vLLM reserve": (r"`ATR_VLLM_VRAM_RESERVE_MB` \((\d+)\)",
                         fields["vllm_vram_reserve_mb"].default),
        "budget fallback": (r"falls back to (\d+) MiB", MEASURED_VRAM_BUDGET_MB),
        "registry reload": (r"at most every (\d+) s \(`ATR_REGISTRY_RELOAD_INTERVAL_S`\)",
                            fields["registry_reload_interval_s"].default),
        "trainer read timeout": (r"(\d+) s for an answer \(`ATR_TRAIN_TIMEOUT_S`\)",
                                 fields["train_timeout_s"].default),
        "trainer connect timeout": (r"waits (\d+) s to connect", TRAINER_CONNECT_TIMEOUT_S),
        "startup wait for the share": (r"curated models after at most (\d+) s",
                                       RegistryWatch.startup_wait_s),
    }
    problems = []
    for what, (pattern, value) in quoted.items():
        found = re.findall(pattern, text)
        if not found:
            problems.append(f"{what}: the sentence quoting it is gone ({pattern})")
        elif any(float(f) != float(value) for f in found):
            problems.append(f"{what}: the doc says {found}, the code says {value}")
    assert not problems, "\n".join(problems)
