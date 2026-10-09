"""Add existing, read-only testing resources without replacing deployed data.

The JSON plan maps resource-relative destinations to existing source paths.
Hard links keep the deployment self-contained without duplicating large image
collections on the same filesystem. No source or existing destination is edited.
"""
import argparse
import json
import os
from pathlib import Path
import uuid


def add_resources(resources, plan):
    resources = Path(resources).resolve(strict=True)
    entries = []
    for relative, source in plan.items():
        parts = Path(relative)
        if parts.is_absolute() or '..' in parts.parts or ':' in relative or not parts.parts:
            raise ValueError('invalid resource destination: ' + relative)
        target = resources / parts
        if resources not in target.resolve().parents:
            raise ValueError('resource destination escapes root: ' + relative)
        source = Path(source).resolve(strict=True)
        if source == resources or resources in source.parents or source in resources.parents:
            raise ValueError('source overlaps deployed resources: ' + str(source))
        entries.append((relative, source, target))
    report = []
    for relative, source, target in entries:
        if target.exists() or target.is_symlink():
            report.append(dict(path=relative, status='preserved-existing'))
            continue
        # Stage only in this exact parent; an interrupted run never publishes a
        # partial dataset. Failed staging is retained for explicit inspection.
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.with_name('.' + target.name + '.incoming-' + uuid.uuid4().hex)
        count, size = 0, 0
        if source.is_file():
            os.link(source, stage)
            count, size = 1, source.stat().st_size
        elif source.is_dir():
            stage.mkdir()
            for directory, dirs, names in os.walk(source, followlinks=False):
                directory = Path(directory)
                if any((directory / name).is_symlink() for name in dirs):
                    raise ValueError('nested directory symlink requires an explicit source: ' + str(directory))
                destination = stage / directory.relative_to(source)
                destination.mkdir(parents=True, exist_ok=True)
                for name in names:
                    item = (directory / name).resolve(strict=True)
                    if not item.is_file():
                        raise ValueError('not a regular resource: ' + str(item))
                    os.link(item, destination / name)
                    count += 1
                    size += item.stat().st_size
        else:
            raise ValueError('invalid resource source: ' + str(source))
        if target.exists() or target.is_symlink():
            raise ValueError('destination appeared during staging; left unchanged: ' + str(target))
        stage.rename(target)
        report.append(dict(path=relative, status='added', files=count, bytes=size))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resources', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    print(json.dumps(add_resources(args.resources, plan), indent=2))


if __name__ == '__main__':
    main()
