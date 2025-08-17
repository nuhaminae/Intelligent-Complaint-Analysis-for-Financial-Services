# tests/conftest.py
import nltk

def pytest_sessionstart(session):
    nltk.download('punkt', quiet=True)
