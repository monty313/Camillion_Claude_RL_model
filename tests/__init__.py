# Marks tests/ as a regular package so `from tests._audit_helpers import cache`
# resolves identically on Python 3.10 (sandbox) and 3.12 (Colab). Without this,
# tests/ was a namespace package and the import failed under 3.12.
