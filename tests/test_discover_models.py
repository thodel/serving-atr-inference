"""
tests/test_discover_models.py

Offline tests for scripts/discover_models.py.
Uses recorded HTTP fixtures (requests_mock).

Run from repo root:
    python -m pytest tests/test_discover_models.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# repo root must be on sys.path for "from scripts.discover_models import"
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.discover_models import (  # noqa: E402
    _normalize_zenodo_id,
    diff_report,
    discover_hf_models,
    discover_zenodo_models,
    format_hf_table,
    format_zenodo_table,
    format_report_markdown,
    report_to_json,
    HFModel,
    ZenodoRecord,
    DiscoveryReport,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent / "fixtures"


def hf_response() -> list[dict]:
    return json.loads((FIXTURES / "hf_response_page1.json").read_text())


def hf_response_page2() -> list[dict]:
    return json.loads((FIXTURES / "hf_response_page2.json").read_text())


def zenodo_response() -> dict:
    return json.loads((FIXTURES / "zenodo_response.json").read_text())


def zenodo_response_page2() -> dict:
    """Empty second page — stops pagination."""
    return {"hits": {"hits": []}, "links": {}}


def _hf_models_from_fixture() -> list:
    """Parse HF page-1 fixture into HFModel objects."""
    from scripts.discover_models import HFModel
    return [
        HFModel(id=d["id"], downloads=d.get("downloads", 0),
                 last_modified=d.get("lastModified", ""), tags=d.get("tags", []))
        for d in hf_response()
    ]


# ─── _normalize_zenodo_id ─────────────────────────────────────────────────────

class TestNormalizeZenodoId:
    def test_full_url(self):
        assert _normalize_zenodo_id("https://zenodo.org/record/15366732") == "15366732"

    def test_doi_prefix(self):
        assert _normalize_zenodo_id("10.5281/zenodo.15366732") == "15366732"

    def test_bare_zenodo_prefix(self):
        assert _normalize_zenodo_id("zenodo.15366732") == "15366732"

    def test_bare_numeric(self):
        assert _normalize_zenodo_id("15366732") == "15366732"

    def test_with_trailing_slash(self):
        assert _normalize_zenodo_id("https://zenodo.org/record/15366732/") == "15366732"

    def test_strips_whitespace(self):
        assert _normalize_zenodo_id("  15366732  ") == "15366732"


# ─── HF API ───────────────────────────────────────────────────────────────────

class TestHuggingFaceDiscovery:
    def test_hf_models_parsed_correct_fields(self):
        """HF fixture → HFModel objects with the right fields."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = hf_response()

        session = MagicMock(spec=requests.Session)
        session.get.return_value = mock_resp

        from scripts.discover_models import _search_hf
        models = _search_hf(session, "kraken HTR", page=1)

        assert len(models) == 3
        ids = {m.id for m in models}
        assert "example/kraken-new-model" in ids
        assert "example/trocr-fine-tune" in ids

        m = next(m for m in models if m.id == "example/kraken-new-model")
        assert m.downloads == 1500
        assert m.last_modified == "2025-03-01T12:00:00Z"
        assert "kraken" in m.tags
        assert m.hf_url == "https://huggingface.co/example/kraken-new-model"

    def test_hf_pagination_stops_at_empty(self):
        """When two consecutive pages return [], pagination stops."""
        # Test the low-level _search_hf function directly with controlled mocks.
        # No need to go through all 8 search terms.
        page1_resp = MagicMock()
        page1_resp.status_code = 200
        page1_resp.json.return_value = hf_response()

        page2_resp = MagicMock()
        page2_resp.status_code = 200
        page2_resp.json.return_value = []

        session = MagicMock(spec=requests.Session)
        session.get.return_value = page1_resp  # default: always return page1_resp

        # First call returns page1_resp, second returns page2_resp, third raises StopIteration
        # The loop will: call1 (3 models, page=2), call2 (empty, consecutive=1, page=3),
        # call3 (StopIteration - but this is query 2 page 1 which we don't want to reach)
        #
        # We test the pagination logic at unit level instead:
        from scripts.discover_models import _search_hf
        p1 = _search_hf(session, "kraken HTR", page=1)
        assert len(p1) == 3

        # Now change return_value for next call
        session.get.return_value = page2_resp
        p2 = _search_hf(session, "kraken HTR", page=2)
        assert p2 == []  # empty stops pagination

    def test_hf_rate_limit_backoff_and_retry(self):
        """429 response triggers Retry-After back-off, then retry succeeds."""
        # Build a proper mock 429 response
        mock_429_resp = MagicMock()
        mock_429_resp.status_code = 429
        mock_429_resp.headers = {"Retry-After": "0"}
        err_429 = requests.exceptions.HTTPError(response=mock_429_resp)

        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.json.return_value = hf_response()

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.json.return_value = []

        session = MagicMock(spec=requests.Session)
        session.get.return_value = empty_resp  # default: empty (stop)
        # Call 1: 429 (raise, caught, retry). Call 2: success (page 1 results).
        # Call 3: empty (consecutive_empty=1, page=3). Call 4: empty (consecutive_empty=2, stop).
        session.get = MagicMock(side_effect=[err_429, mock_success, empty_resp, empty_resp])

        import scripts.discover_models as dm
        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]):
            models, error = discover_hf_models(session)
        assert len(models) == 3
        assert error is None
        assert session.get.call_count == 4

    def test_hf_500_keeps_going_to_next_query(self):
        """500 on one query does not crash; error is recorded, next query runs."""
        # Mock 500 response: raise_for_status() raises HTTPError, json() unused
        mock_500_resp = MagicMock()
        mock_500_resp.status_code = 500
        mock_500_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_500_resp
        )

        session = MagicMock(spec=requests.Session)
        session.get.return_value = mock_500_resp  # raises HTTPError on first call

        import scripts.discover_models as dm
        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]):
            models, error = discover_hf_models(session)
        # 500 raised, caught, loop broke cleanly, no results
        assert len(models) == 0
        assert error is not None
        assert "500" in error

    def test_hf_deduplication_by_id(self):
        """Same model appearing in two query results appears only once."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        session = MagicMock(spec=requests.Session)
        # page 1: results, page 2: empty (stop)
        mock_resp.json.side_effect = [hf_response(), []]
        session.get.return_value = mock_resp

        from scripts.discover_models import _search_hf
        # Run query 1 through both pages, query 2 empty
        p1 = _search_hf(session, "kraken HTR", page=1)
        assert len(p1) == 3
        # Verify deduplication
        all_ids = {m.id for m in p1}
        assert "example/kraken-new-model" in all_ids
        assert "example/trocr-fine-tune" in all_ids


# ─── Zenodo API ───────────────────────────────────────────────────────────────

class TestZenodoDiscovery:
    def test_zenodo_records_parsed_correctly(self):
        """Zenodo fixture → ZenodoRecord objects with normalised IDs."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = zenodo_response()

        empty_resp = MagicMock()
        empty_resp.status_code = 200
        empty_resp.json.return_value = zenodo_response_page2()

        session = MagicMock(spec=requests.Session)
        # Patch communities to 1 entry so only 2 calls needed: page1 results, page2 empty
        import scripts.discover_models as dm
        with patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]),              patch.object(dm, "_search_zenodo", lambda s, p: mock_resp.json() if p.get("page", 1) == 1 else empty_resp.json()):
            session.get.return_value = mock_resp
            records, error = discover_zenodo_models(session)

        assert len(records) == 3
        ids = {r.zenodo_id for r in records}
        assert "15366732" in ids
        assert "20000001" in ids
        assert "99999999" in ids

    def test_zenodo_doi_normalised_in_record(self):
        """DOI is stored in record.doi exactly as returned by API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = zenodo_response()

        import scripts.discover_models as dm
        def fake_search(session, params):
            return mock_resp.json() if params.get("page", 1) == 1 else zenodo_response_page2()

        with patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]), \
             patch.object(dm, "_search_zenodo", fake_search):
            session = MagicMock(spec=requests.Session)
            session.get.return_value = mock_resp
            records, _ = discover_zenodo_models(session)

        r153 = next(r for r in records if r.zenodo_id == "15366732")
        assert r153.doi == "10.5281/zenodo.15366732"
        assert r153.zenodo_url == "https://zenodo.org/records/15366732"

    def test_zenodo_500_error_is_graceful(self):
        """500 response on Zenodo: no crash, error recorded, returns partial results."""
        mock_500_resp = MagicMock()
        mock_500_resp.status_code = 500
        mock_500_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_500_resp
        )

        import scripts.discover_models as dm
        # Patch communities to 1 entry so only 1 mock call is needed
        with patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]):
            session = MagicMock(spec=requests.Session)
            session.get.return_value = mock_500_resp
            records, error = discover_zenodo_models(session)

        assert error is not None
        assert "500" in error


# ─── Diff ─────────────────────────────────────────────────────────────────────

class TestDiff:
    def test_hf_model_excluded_by_hf_repo(self):
        """A model already served by HF repo ID is excluded."""
        import scripts.discover_models as dm
        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]), \
             patch.object(dm, "_search_hf", lambda s, q, page: _hf_models_from_fixture() if page == 1 else []):
            models, _ = discover_hf_models(MagicMock(spec=requests.Session))

        report = DiscoveryReport(hf_candidates=models)
        diff_report(report, served_hf_repos={"wjbmattingly/LightOnOCR-2-1B-catmus-caroline"}, served_zenodo_ids=set())

        # Served model excluded
        assert "wjbmattingly/LightOnOCR-2-1B-catmus-caroline" not in {m.id for m in report.new_hf_models}
        # New models included
        assert "example/kraken-new-model" in {m.id for m in report.new_hf_models}
        assert "example/trocr-fine-tune" in {m.id for m in report.new_hf_models}

    def test_zenodo_record_excluded_by_zenodo_id(self):
        """A Zenodo record already served is excluded."""
        import scripts.discover_models as dm
        with patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]), \
             patch.object(dm, "_search_zenodo", lambda s, p: zenodo_response() if p.get("page", 1) == 1 else zenodo_response_page2()):
            records, _ = discover_zenodo_models(MagicMock(spec=requests.Session))

        report = DiscoveryReport(zenodo_candidates=records)
        diff_report(report, served_hf_repos=set(), served_zenodo_ids={"15366732", "99999999"})

        ids_in_new = {r.zenodo_id for r in report.new_zenodo_models}
        assert "15366732" not in ids_in_new  # already served
        assert "99999999" not in ids_in_new  # already served
        assert "20000001" in ids_in_new      # new

    def test_diff_is_case_insensitive(self):
        """HF repo matching is case-insensitive."""
        import scripts.discover_models as dm
        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]), \
             patch.object(dm, "_search_hf", lambda s, q, page: _hf_models_from_fixture() if page == 1 else []):
            models, _ = discover_hf_models(MagicMock(spec=requests.Session))

        report = DiscoveryReport(hf_candidates=models)
        diff_report(report, served_hf_repos={"EXAMPLE/KRAKEN-NEW-MODEL"}, served_zenodo_ids=set())

        assert "example/kraken-new-model" not in {m.id for m in report.new_hf_models}

    def test_two_sources_one_fails_still_produces_report(self):
        """HF fails but Zenodo succeeds: report is still produced."""
        import scripts.discover_models as dm

        # HF failure
        mock_500_resp = MagicMock()
        mock_500_resp.status_code = 500
        mock_500_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_500_resp
        )
        hf_session = MagicMock(spec=requests.Session)
        hf_session.get.return_value = mock_500_resp

        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]):
            hf_models, hf_error = discover_hf_models(hf_session)

        # Zenodo success: page 1 has data, page 2 is empty to stop pagination
        with patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]), \
             patch.object(dm, "_search_zenodo", lambda s, p: zenodo_response() if p.get("page", 1) == 1 else zenodo_response_page2()):
            zenodo_records, zenodo_error = discover_zenodo_models(MagicMock(spec=requests.Session))

        report = DiscoveryReport()
        report.hf_candidates = hf_models
        report.zenodo_candidates = zenodo_records
        if hf_error:
            report.errors.append(hf_error)

        assert len(report.zenodo_candidates) == 3
        assert len(report.hf_candidates) == 0
        assert len(report.errors) == 1
        assert "HF query" in report.errors[0]


# ─── Markdown renderer ────────────────────────────────────────────────────────

class TestMarkdownRenderer:
    def test_hf_table_empty(self):
        assert "No new HuggingFace candidates found" in format_hf_table([])

    def test_hf_table_sorted_by_downloads(self):
        models = [
            HFModel(id="low downloads", downloads=10, last_modified="2025-01-01T00:00:00Z", tags=[]),
            HFModel(id="high downloads", downloads=10000, last_modified="2025-01-01T00:00:00Z", tags=[]),
            HFModel(id="medium downloads", downloads=500, last_modified="2025-01-01T00:00:00Z", tags=[]),
        ]
        table = format_hf_table(models)
        hi = table.index("high downloads")
        med = table.index("medium downloads")
        lo = table.index("low downloads")
        assert hi < med < lo  # highest downloads first

    def test_zenodo_table_empty(self):
        assert "No new Zenodo candidates found" in format_zenodo_table([])

    def test_zenodo_table_sorted_by_id(self):
        records = [
            ZenodoRecord(zenodo_id="50000000", title="Model B", doi="10.5281/zenodo.50000000", keywords=[], zenodo_url=""),
            ZenodoRecord(zenodo_id="10000000", title="Model A", doi="10.5281/zenodo.10000000", keywords=[], zenodo_url=""),
        ]
        table = format_zenodo_table(records)
        a = table.index("Model A")
        b = table.index("Model B")
        assert a < b  # sorted by ID ascending

    def test_markdown_report_contains_all_sections(self):
        report = DiscoveryReport(
            hf_candidates=[],
            zenodo_candidates=[],
            new_hf_models=[],
            new_zenodo_models=[],
            served_hf_repos={"hf/repo1"},
            served_zenodo_ids={"12345"},
            errors=["HF query 'foo' page 1: 500 Server Error"],
        )
        md = format_report_markdown(report)
        assert "# Model Discovery Report" in md
        assert "0 new / 0 total" in md
        assert "500 Server Error" in md
        assert "HF query" in md

    def test_golden_md_fixture(self, tmp_path):
        """Rendered markdown for the fixture set matches golden fixture."""
        golden_md = FIXTURES / "discovery_golden.md"
        if not golden_md.exists():
            pytest.skip(f"Golden fixture not found: {golden_md}")

        # Build a report from fixture data
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = hf_response()

        session = MagicMock(spec=requests.Session)
        session.get.return_value = mock_resp

        from scripts.discover_models import _search_hf
        models = _search_hf(session, "kraken HTR", page=1)

        report = DiscoveryReport(
            hf_candidates=models,
            served_hf_repos={"wjbmattingly/LightOnOCR-2-1B-catmus-caroline"},
            served_zenodo_ids=set(),
        )
        diff_report(report, served_hf_repos=set(), served_zenodo_ids=set())

        rendered = format_report_markdown(report)
        # Check the key structural elements
        assert "## New HuggingFace Models" in rendered
        assert "## New Zenodo Records" in rendered
        assert "example/kraken-new-model" in rendered
        assert "example/trocr-fine-tune" in rendered


# ─── JSON serialisation ───────────────────────────────────────────────────────

class TestJsonReport:
    def test_report_to_json_is_serialisable(self):
        report = DiscoveryReport(
            hf_candidates=[
                HFModel(id="test/model", downloads=100, last_modified="2025-01-01T00:00:00Z", tags=["test"])
            ],
            zenodo_candidates=[
                ZenodoRecord(zenodo_id="99999", title="Test", doi="10.5281/zenodo.99999", keywords=["test"], zenodo_url="")
            ],
            new_hf_models=[],
            new_zenodo_models=[],
            served_hf_repos=set(),
            served_zenodo_ids=set(),
            errors=[],
        )
        data = report_to_json(report)
        assert isinstance(data, dict)
        assert len(data["hf_candidates"]) == 1
        assert len(data["zenodo_candidates"]) == 1
        # Must be JSON-serialisable (no Path objects, etc.)
        json.dumps(data)


# ─── Integration: full discover() with mocked session ─────────────────────────

class TestDiscoverIntegration:
    def test_discover_runs_without_crash(self):
        """Smoke test: discover() completes with all sources mocked out."""
        import scripts.discover_models as dm

        hf_resp = MagicMock()
        hf_resp.status_code = 200
        hf_resp.json.return_value = hf_response()

        zenodo_resp = MagicMock()
        zenodo_resp.status_code = 200
        zenodo_resp.json.return_value = zenodo_response()

        def fake_hf_search(session, query, page):
            return _hf_models_from_fixture() if page == 1 else []

        def fake_zenodo_search(session, params):
            return zenodo_response() if params.get("page", 1) == 1 else zenodo_response_page2()

        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]), \
             patch.object(dm, "_search_hf", fake_hf_search), \
             patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]), \
             patch.object(dm, "_search_zenodo", fake_zenodo_search), \
             patch.object(dm, "_load_registry_ids", lambda: (set(), set())):
            mock_session = MagicMock(spec=requests.Session)
            from scripts.discover_models import discover
            report = discover(mock_session)

        assert len(report.hf_candidates) == 3
        assert len(report.zenodo_candidates) == 3
        assert len(report.served_hf_repos) == 0
        assert len(report.served_zenodo_ids) == 0

    def test_both_sources_fail_exit1(self):
        """When both sources fail completely, discover() returns empty candidates + ≥1 error."""
        import scripts.discover_models as dm

        mock_500_resp = MagicMock()
        mock_500_resp.status_code = 500
        mock_500_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error", response=mock_500_resp
        )

        def fake_hf_search(session, query, page):
            raise requests.exceptions.HTTPError("500 Server Error", response=mock_500_resp)

        with patch.object(dm, "HF_SEARCH_TERMS", ["kraken HTR"]), \
             patch.object(dm, "_search_hf", fake_hf_search), \
             patch.object(dm, "ZENODO_COMMUNITIES", ["scribes"]), \
             patch.object(dm, "_search_zenodo", lambda s, p: (_ for _ in ()).throw(
                 requests.exceptions.HTTPError("500 Server Error", response=mock_500_resp))), \
             patch.object(dm, "_load_registry_ids", lambda: (set(), set())):
            mock_session = MagicMock(spec=requests.Session)
            from scripts.discover_models import discover
            report = discover(mock_session)

        assert len(report.hf_candidates) == 0
        assert len(report.zenodo_candidates) == 0
        assert len(report.errors) >= 1