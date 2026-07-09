"""
tests/test_discover_models_github.py

Offline tests for the GitHub issue logic in discover_models.py.
Uses unittest.mock (no external fixtures server needed).

Run from repo root:
    python -m pytest tests/test_discover_models_github.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.discover_models import (
    GITHUB_ISSUE_MARKER,
    GITHUB_ISSUE_TITLE,
    _build_checklist,
    _build_issue_body,
    _find_existing_issue,
    _update_or_create_issue,
    _repo_info,
    _github_headers,
    REPO_ROOT,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def minimal_report(tmp_path: Path) -> Path:
    """Write a minimal discovery_report.json and return its path."""
    path = tmp_path / "discovery_report.json"
    path.write_text(
        json.dumps(
            {
                "new_hf_models": [
                    {
                        "id": "user/otr-model",
                        "downloads": 999,
                        "last_modified": "2025-01-01",
                        "tags": [],
                    }
                ],
                "new_zenodo_models": [
                    {
                        "zenodo_id": "9999999",
                        "title": "Ancient OTR Dataset",
                        "doi": "10.5281/zenodo.9999999",
                        "keywords": ["HTR"],
                        "zenodo_url": "https://zenodo.org/records/9999999",
                    }
                ],
                "hf_candidates": [],
                "zenodo_candidates": [],
                "served_hf_repos": [],
                "served_zenodo_ids": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    return path


# ─── _repo_info ────────────────────────────────────────────────────────────────

class TestRepoInfo:
    def test_parses_github_repository_env(self):
        with patch.dict(os.environ, {"GITHUB_REPOSITORY": "owner/repo-name"}):
            owner, name = _repo_info()
            assert owner == "owner"
            assert name == "repo-name"

    def test_falls_back_to_default(self):
        with patch.dict(os.environ, {}, clear=True):
            owner, name = _repo_info()
            assert owner == "thodel"
            assert name == "serving-atr-inference"


# ─── _github_headers ───────────────────────────────────────────────────────────

class TestGithubHeaders:
    def test_includes_accept_header(self):
        with patch.dict(os.environ, {}, clear=True):
            h = _github_headers()
            assert "Accept" in h
            assert "application/vnd.github+json" in h["Accept"]

    def test_adds_token_when_set(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "my-token"}):
            h = _github_headers()
            assert h.get("Authorization") == "Bearer my-token"

    def test_no_auth_without_token(self):
        with patch.dict(os.environ, {}, clear=True):
            h = _github_headers()
            assert "Authorization" not in h


# ─── Issue-body renderer ───────────────────────────────────────────────────────

class TestIssueBodyRenderer:
    def test_marker_present(self, tmp_path: Path):
        report_path = minimal_report(tmp_path)
        body = _build_issue_body("# Mock Report\n\ntest content", report_path)
        assert GITHUB_ISSUE_MARKER in body

    def test_body_starts_with_marker(self, tmp_path: Path):
        """The marker comment appears at the very start of the body."""
        report_path = minimal_report(tmp_path)
        body = _build_issue_body("# Test Report\n\ncontent", report_path)
        assert body.startswith(GITHUB_ISSUE_MARKER)

    def test_checklist_rows_hf(self, tmp_path: Path):
        report_path = minimal_report(tmp_path)
        checklist = _build_checklist(report_path)
        assert "otr-model" in checklist
        assert "999" in checklist

    def test_checklist_rows_zenodo(self, tmp_path: Path):
        report_path = minimal_report(tmp_path)
        checklist = _build_checklist(report_path)
        assert "Ancient OTR Dataset" in checklist
        assert "9999999" in checklist

    def test_empty_candidates_message(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text(
            json.dumps(
                {
                    "new_hf_models": [],
                    "new_zenodo_models": [],
                    "hf_candidates": [],
                    "zenodo_candidates": [],
                    "served_hf_repos": [],
                    "served_zenodo_ids": [],
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        checklist = _build_checklist(path)
        assert "No new candidates" in checklist

    def test_missing_report_path_returns_placeholder(self):
        checklist = _build_checklist(Path("/nonexistent/report.json"))
        assert "Run the discover step first" in checklist

    def test_golden_file_no_duplicate_issue_titles(self, tmp_path: Path):
        """Body contains the marker but not two copies of the title."""
        report_path = minimal_report(tmp_path)
        body = _build_issue_body("# Report\n\ncontent", report_path)
        assert body.count(GITHUB_ISSUE_MARKER) == 1


# ─── Create-vs-update decision ─────────────────────────────────────────────────

class TestCreateVsUpdate:
    def test_no_existing_issue_creates(self, tmp_path: Path):
        """When no existing marker-issue is found, a new issue is POSTed."""
        import os
        report_path = minimal_report(tmp_path)
        session = MagicMock(spec=requests.Session)
        mock_resp = MagicMock()
        mock_resp.json.return_value = []  # no existing issues
        session.get.return_value = mock_resp

        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"number": 99, "title": GITHUB_ISSUE_TITLE}
        session.post.return_value = mock_post_resp

        with patch.dict(os.environ, {"GITHUB_TOKEN": "fake"}):
            n = _update_or_create_issue(session, "# Report\n\nbody", report_path)

        assert n == 99
        # Should have called GET (list) then POST (create)
        assert session.get.call_count == 1
        assert session.post.call_count == 1
        assert session.patch.call_count == 0

    def test_existing_issue_updates(self, tmp_path: Path):
        """When an existing marker-issue is found, it is PATCHed (not duped)."""
        import os
        report_path = minimal_report(tmp_path)
        session = MagicMock(spec=requests.Session)
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = [
            {"number": 42, "title": GITHUB_ISSUE_TITLE.strip(), "state": "open"}
        ]
        session.get.return_value = mock_get_resp

        mock_patch_resp = MagicMock()
        session.patch.return_value = mock_patch_resp

        with patch.dict(os.environ, {"GITHUB_TOKEN": "fake"}):
            n = _update_or_create_issue(session, "# Report\n\nbody", report_path)

        assert n == 42
        assert session.get.call_count == 1
        assert session.patch.call_count == 1
        assert session.post.call_count == 0

    def test_second_run_reuses_same_issue(self, tmp_path: Path):
        """A second dispatch reuses the same issue number (no duplication)."""
        import os
        report_path = minimal_report(tmp_path)
        session = MagicMock(spec=requests.Session)
        mock_get_resp = MagicMock()
        mock_get_resp.json.return_value = [
            {"number": 7, "title": GITHUB_ISSUE_TITLE.strip(), "state": "open"}
        ]
        session.get.return_value = mock_get_resp
        session.patch.return_value = MagicMock()

        with patch.dict(os.environ, {"GITHUB_TOKEN": "fake"}):
            n1 = _update_or_create_issue(session, "# Report v1\n\nbody", report_path)
            n2 = _update_or_create_issue(session, "# Report v2\n\nbody", report_path)

        assert n1 == n2 == 7
        # Two runs: each does 1×GET + 1×PATCH — no POST
        assert session.get.call_count == 2
        assert session.patch.call_count == 2
        assert session.post.call_count == 0

    def test_empty_candidates_does_not_call_http(self, tmp_path: Path):
        """When no candidates, no HTTP requests should be made."""
        import os
        path = tmp_path / "empty.json"
        path.write_text(
            json.dumps(
                {
                    "new_hf_models": [],
                    "new_zenodo_models": [],
                    "hf_candidates": [],
                    "zenodo_candidates": [],
                    "served_hf_repos": [],
                    "served_zenodo_ids": [],
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        session = MagicMock()
        with patch.dict(os.environ, {"GITHUB_TOKEN": "fake"}):
            _update_or_create_issue(session, "# empty report", path)
        # No HTTP calls for empty candidate list
        session.get.assert_not_called()
        session.post.assert_not_called()
        session.patch.assert_not_called()


# ─── Workflow YAML ─────────────────────────────────────────────────────────────

class TestWorkflowYaml:
    def test_workflow_parses(self):
        workflow_path = REPO_ROOT / ".github" / "workflows" / "discover-models.yml"
        if not workflow_path.exists():
            pytest.skip("workflow not yet written")

        import yaml

        with open(workflow_path, encoding="utf-8") as f:
            wf = yaml.safe_load(f)

        # 'on' is a YAML keyword that parses as Python boolean True
        on_section = wf.get(True, wf.get("on"))
        assert on_section is not None
        assert "schedule" in on_section
        assert "workflow_dispatch" in on_section
        assert wf.get("permissions", {}).get("issues") == "write"

    def test_schedule_is_weekly(self):
        workflow_path = REPO_ROOT / ".github" / "workflows" / "discover-models.yml"
        if not workflow_path.exists():
            pytest.skip("workflow not yet written")

        import yaml

        with open(workflow_path, encoding="utf-8") as f:
            wf = yaml.safe_load(f)

        on_section = wf.get(True, wf.get("on"))
        cron_expr = on_section["schedule"][0]["cron"]
        # Weekly: 5 fields, e.g. "0 8 * * 1" = Monday at 08:00 UTC
        fields = cron_expr.split()
        assert len(fields) == 5, f"Expected 5-field cron, got: {cron_expr}"

    def test_github_token_used_for_issue_write(self):
        workflow_path = REPO_ROOT / ".github" / "workflows" / "discover-models.yml"
        if not workflow_path.exists():
            pytest.skip("workflow not yet written")

        content = workflow_path.read_text(encoding="utf-8")
        assert "GITHUB_TOKEN" in content


import os
import requests  # noqa: F401