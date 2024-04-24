import importlib
import pkgutil

import pytest


def test_imports():
    try:
        package = importlib.import_module("parea")
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"parea.{module_name}")
    except ImportError:
        pytest.fail("Import failed", pytrace=False)
