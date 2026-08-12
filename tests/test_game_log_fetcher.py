import requests
import pytest

import game_log_fetcher as glf


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'HTTP {self.status_code}')

    def json(self):
        return self._payload


def test_successful_empty_response_is_not_a_failure(monkeypatch):
    monkeypatch.setattr(
        glf._req,
        'get',
        lambda *args, **kwargs: FakeResponse(payload={'stats': []}),
    )

    assert glf._get_splits(1, 2026, {'season': 2026}) == []


def test_404_is_treated_as_no_history(monkeypatch):
    monkeypatch.setattr(
        glf._req,
        'get',
        lambda *args, **kwargs: FakeResponse(status_code=404),
    )

    assert glf._get_splits(1, 2026, {'season': 2026}) == []


@pytest.mark.parametrize('failure', [
    requests.Timeout('timed out'),
    requests.ConnectionError('connection failed'),
])
def test_transport_failure_raises_instead_of_becoming_empty(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(glf._req, 'get', fail)

    with pytest.raises(glf.LogFetchError):
        glf._get_splits(1, 2026, {'season': 2026})


def test_http_500_raises_instead_of_becoming_empty(monkeypatch):
    monkeypatch.setattr(
        glf._req,
        'get',
        lambda *args, **kwargs: FakeResponse(status_code=500),
    )

    with pytest.raises(glf.LogFetchError):
        glf._get_splits(1, 2026, {'season': 2026})


def test_malformed_mlb_row_raises(monkeypatch):
    monkeypatch.setattr(
        glf,
        '_get_splits',
        lambda *args, **kwargs: [{'stat': {'atBats': 'not-an-int'}}],
    )

    with pytest.raises(glf.LogFetchError, match='malformed MLB game log'):
        glf._fetch_mlb_rows(1, 2026)


def test_malformed_milb_row_raises(monkeypatch):
    monkeypatch.setattr(
        glf,
        '_get_splits',
        lambda *args, **kwargs: [{'stat': {'atBats': 'not-an-int'}}],
    )

    with pytest.raises(glf.LogFetchError, match='malformed MiLB game log'):
        glf._fetch_milb_rows(1, 2026)


def test_fetch_player_logs_does_not_fallback_after_current_fetch_failure(monkeypatch):
    calls = []

    def fail_current(player_id, season):
        calls.append((player_id, season))
        raise glf.LogFetchError('upstream failed')

    monkeypatch.setattr(glf, '_fetch_mlb_rows', fail_current)

    with pytest.raises(glf.LogFetchError, match='upstream failed'):
        glf.fetch_player_logs(1)

    assert len(calls) == 1
