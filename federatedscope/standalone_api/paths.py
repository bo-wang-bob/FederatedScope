"""Portable defaults: resolve local relative paths against the backend checkout."""
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def project_path(value, repo=REPO_ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else Path(repo) / path).resolve()


def env_path(name, default, repo=REPO_ROOT):
    return project_path(os.environ.get(name) or default, repo)
