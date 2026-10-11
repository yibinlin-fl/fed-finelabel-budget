import importlib.util
import io
from pathlib import Path
import pytest


def test_interrupted_download_never_creates_final_archive(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cifar_download", Path(__file__).parents[1] / "scripts/download_cifar100.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class BrokenStream(io.BytesIO):
        def read(self, size):
            if self.tell():
                raise TimeoutError("simulated network interruption")
            return super().read(size)

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: BrokenStream(b"partial"))
    archive = tmp_path / "cifar-100-python.tar.gz"
    with pytest.raises(TimeoutError):
        module.download_archive(archive)
    assert not archive.exists()
    assert archive.with_suffix(".download").read_bytes() == b"partial"


def test_corrupt_download_not_published(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("cifar_download", Path(__file__).parents[1] / "scripts/download_cifar100.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"invalid"))
    archive = tmp_path / "archive.tar.gz"
    with pytest.raises(RuntimeError, match="MD5"):
        module.download_archive(archive)
    assert not archive.exists()
