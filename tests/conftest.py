import pytest

from murakami_lab_modules import utils


@pytest.fixture(autouse=True)
def silence_library_logging(monkeypatch):
    monkeypatch.setattr(utils, 'logging', lambda *args, **kwargs: None)
    monkeypatch.setattr(utils, '_device_alart', False)
