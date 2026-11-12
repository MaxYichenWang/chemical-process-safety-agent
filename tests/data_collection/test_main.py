"""Collection failures must propagate through the supported CLI."""

from unittest.mock import Mock

import pytest

from src.data_collection import main as collection


@pytest.mark.parametrize("results", [{}, {"inspection": {"file_path": None}}])
def test_empty_enforcement_download_is_a_cli_failure(tmp_path, monkeypatch, results):
    scraper = Mock()
    scraper.run_scraper.return_value = results
    monkeypatch.setattr(collection, "OSHAEnforcementDataScraper", lambda **kwargs: scraper)
    monkeypatch.setattr("sys.argv", ["collection", "--mode", "osha", "--data-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as raised:
        collection.main()
    assert raised.value.code == 1
