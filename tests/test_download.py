"""Downloader behaviour with a fake HTTP layer: no network access is made."""
import hashlib
import http.client
import io
import sys
import urllib.error

import pytest

import download_public_data as downloader

PAYLOAD = b'verified parquet bytes ' * 4096
OLD = b'older local copy'


class Response(io.BytesIO):
    def __init__(self, payload, content_length=None):
        super().__init__(payload)
        self.headers = {} if content_length is None else {'Content-Length': str(content_length)}


class BrokenResponse(Response):
    def read(self, size=-1):
        raise http.client.IncompleteRead(b'partial')


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setitem(downloader.SOURCE_HASHES, 'train', hashlib.sha256(PAYLOAD).hexdigest())


def serve(monkeypatch, factory):
    requested = []

    def urlopen(request, timeout):
        requested.append(request.full_url)
        return factory()

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', urlopen)
    return requested


def leftovers(directory):
    return sorted(path.name for path in directory.iterdir() if path.name != 'train.parquet')


def run_main(monkeypatch, directory, *extra):
    monkeypatch.setattr(sys, 'argv', ['download_public_data.py', '--data-dir', str(directory), '--splits', 'train', *extra])
    return downloader.main()


def test_sha256_file(tmp_path):
    path = tmp_path / 'file.bin'
    path.write_bytes(PAYLOAD)
    assert downloader.sha256_file(path) == hashlib.sha256(PAYLOAD).hexdigest()


def test_verified_file_is_reused_without_network(tmp_path, monkeypatch, pinned):
    (tmp_path / 'train.parquet').write_bytes(PAYLOAD)
    requested = serve(monkeypatch, lambda: pytest.fail('network must not be used'))
    assert downloader.ensure_split('train', tmp_path, check_only=False, timeout=1) == tmp_path / 'train.parquet'
    assert requested == []


def test_check_only_never_downloads(tmp_path, monkeypatch, pinned):
    requested = serve(monkeypatch, lambda: pytest.fail('network must not be used'))
    with pytest.raises(FileNotFoundError):
        downloader.ensure_split('train', tmp_path, check_only=True, timeout=1)
    (tmp_path / 'train.parquet').write_bytes(OLD)
    with pytest.raises(ValueError):
        downloader.ensure_split('train', tmp_path, check_only=True, timeout=1)
    assert (tmp_path / 'train.parquet').read_bytes() == OLD and requested == []


def test_download_is_installed_only_after_hash_verification(tmp_path, monkeypatch, pinned):
    target = tmp_path / 'raw'
    requested = serve(monkeypatch, lambda: Response(PAYLOAD, len(PAYLOAD)))
    assert downloader.ensure_split('train', target, check_only=False, timeout=1) == target / 'train.parquet'
    assert (target / 'train.parquet').read_bytes() == PAYLOAD and leftovers(target) == []
    assert requested == [f'{downloader.BASE_URL}/{downloader.REVISION}/data/train.parquet']


def test_wrong_hash_keeps_the_existing_file(tmp_path, monkeypatch, pinned):
    (tmp_path / 'train.parquet').write_bytes(OLD)
    serve(monkeypatch, lambda: Response(b'tampered ' + PAYLOAD))
    with pytest.raises(ValueError, match='SHA-256'):
        downloader.ensure_split('train', tmp_path, check_only=False, timeout=1)
    assert (tmp_path / 'train.parquet').read_bytes() == OLD and leftovers(tmp_path) == []


def test_truncated_download_keeps_the_existing_file(tmp_path, monkeypatch, pinned):
    (tmp_path / 'train.parquet').write_bytes(OLD)
    serve(monkeypatch, lambda: Response(PAYLOAD[:100], len(PAYLOAD)))
    with pytest.raises(ValueError, match='incomplete'):
        downloader.ensure_split('train', tmp_path, check_only=False, timeout=1)
    assert (tmp_path / 'train.parquet').read_bytes() == OLD and leftovers(tmp_path) == []


def test_main_reports_a_dropped_connection_without_a_traceback(tmp_path, monkeypatch, pinned, capsys):
    (tmp_path / 'train.parquet').write_bytes(OLD)
    serve(monkeypatch, lambda: BrokenResponse(PAYLOAD, len(PAYLOAD)))
    assert run_main(monkeypatch, tmp_path) == 1
    assert 'ERROR train: IncompleteRead' in capsys.readouterr().err
    assert (tmp_path / 'train.parquet').read_bytes() == OLD and leftovers(tmp_path) == []


def test_main_hides_urls_in_http_errors(tmp_path, monkeypatch, pinned, capsys):
    def refuse():
        raise urllib.error.HTTPError('https://example.invalid/secret?token=abc', 404, 'Not Found', {}, None)

    serve(monkeypatch, refuse)
    assert run_main(monkeypatch, tmp_path) == 1
    error = capsys.readouterr().err
    assert 'HTTP 404' in error and 'example.invalid' not in error and 'token' not in error
    assert leftovers(tmp_path) == [] and not (tmp_path / 'train.parquet').exists()


def test_main_check_only_and_success_exit_codes(tmp_path, monkeypatch, pinned, capsys):
    assert run_main(monkeypatch, tmp_path, '--check-only') == 1
    assert 'data file is missing' in capsys.readouterr().err
    serve(monkeypatch, lambda: Response(PAYLOAD, len(PAYLOAD)))
    assert run_main(monkeypatch, tmp_path) == 0
    assert run_main(monkeypatch, tmp_path, '--check-only') == 0
