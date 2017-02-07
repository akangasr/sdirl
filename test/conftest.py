import pytest
def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true",
            help="run slow tests")
    parser.addoption("--vslow", action="store_true",
            help="run very slow tests")


