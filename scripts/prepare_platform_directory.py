"""Copy a self-contained release directory without changing a running installation.

This prepares files, not a Docker image. All CLI paths resolve from the caller's
working directory. Existing destinations, links and changing job snapshots fail.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TERMINAL = {'completed', 'failed', 'stopped', 'interrupted'}
IGNORE = shutil.ignore_patterns('.git', '__pycache__', '*.pyc', 'node_modules',
                               '.env', '.env.*', '*.tsbuildinfo')
HELPER_HASHES = {
    '530107bc5e7679cc3f49cbdc16a3f0030e59cf17e7b26a962b278efb7bf1f513',
    '24c47c2c434195bfbad196f05b953d73a9bbe01a4638ac7a5fa5180c5b24cd2e',
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def files(root):
    """No links: even an internal absolute link would break after relocation."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError('symbolic link is not portable: ' + str(root))
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise ValueError('missing input: ' + str(root))
    for directory, dirs, names in os.walk(root, followlinks=False):
        ignored = IGNORE(directory, dirs + names)
        dirs[:] = [name for name in dirs if name not in ignored]
        for name in dirs + [n for n in names if n not in ignored]:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError('symbolic link is not portable: ' + str(path))
        for name in names:
            if name not in ignored:
                path = Path(directory) / name
                if not path.is_file():
                    raise ValueError('not a regular file: ' + str(path))
                yield path


def copy_checked(source, destination):
    source, destination = Path(source), Path(destination)
    for path in files(source):
        target = destination / path.relative_to(source) if source.is_dir() else destination
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(path)
        shutil.copy2(path, target)
        if sha256(target) != before or sha256(path) != before:
            raise ValueError('input changed while copying: ' + str(path))


def state_snapshot(root):
    root = Path(root)
    snapshot = {}
    for path in files(root / 'jobs'):
        if path.name == 'job.json':
            job = json.loads(path.read_text(encoding='utf-8'))
            if (job['status'] not in TERMINAL or not job.get('cleanup', {}).get('ok', True)):
                raise ValueError('active/unclean experiment; stop it before preparing: ' + job['id'])
        # Process identity / service leases / matplotlib caches are host-local.
        if path.name in {'process.json', 'service.lock'} or 'mpl' in path.relative_to(root).parts:
            continue
        snapshot[path.relative_to(root).as_posix()] = sha256(path)
    if not snapshot:
        raise ValueError('no recorded experiments to copy')
    return snapshot


def portable_privacy_helper(path):
    """Adapt only the two reviewed resource-script versions, never arbitrary code."""
    original_hash = sha256(path)
    if original_hash not in HELPER_HASHES:
        raise ValueError('unreviewed privacy helper; review its paths before packaging')
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    replacements = {
        'REPO_ROOT': 'Path(__file__).resolve().parents[2]',
        'NO_DEFENSE_RUN_DIR': "str(Path(__file__).resolve().parent / 'runs/no_defense')",
        'DEFENSE_RUN_DIR': "str(Path(__file__).resolve().parent / 'runs/defense')",
    }
    edits = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in replacements:
                edits.append((node.lineno - 1, node.end_lineno,
                              name + ' = ' + replacements.pop(name) + '\n'))
    if replacements:
        raise ValueError('privacy helper layout changed')
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    source = ''.join(lines)
    source = source.replace('    cfg.freeze()\n', '    cfg.freeze(save=False, inform=False)\n')
    marker = '    cfg.freeze(save=False, inform=False)\n'
    if source.count(marker) != 1:
        raise ValueError('privacy helper freeze contract changed')
    source = source.replace(marker, "    cfg.data.root = str(Path(__file__).resolve().parent /\n"
                            "                        'datasets/OfficeHomeDataset_10072016')\n" + marker)
    ast.parse(source)
    path.write_text(source, encoding='utf-8')
    return dict(originalSha256=original_hash, packagedSha256=sha256(path),
                changes=['backend import root', 'local run roots', 'local dataset root', 'read-only config'])


def prepare(args):
    output = args.output.absolute()
    sources = [REPO, args.frontend, args.dataset, args.features, args.weights, args.privacy, args.state]
    # Lexical existence also catches a dangling destination link.
    if os.path.lexists(output):
        raise ValueError('output already exists; choose a NEW directory: ' + str(output))
    output = output.resolve()
    for source in sources:
        resolved = source.resolve()
        if output == resolved or output in resolved.parents or resolved in output.parents:
            raise ValueError('output must not overlap any source: ' + str(source))
    snapshot = state_snapshot(args.state)
    if sha256(args.privacy / 'show_fedmia_examples.py') not in HELPER_HASHES:
        raise ValueError('unreviewed privacy helper')
    output.mkdir(parents=True)
    (output / 'PREPARATION_INCOMPLETE').write_text('Do not deploy until preparation succeeds.\n')
    backend, frontend = output / 'backend', output / 'frontend'
    for name in ('federatedscope', 'scripts', 'deploy', 'docs', 'tests',
                 'run.py', 'setup.py', 'README.md', 'LICENSE', '.gitignore'):
        copy_checked(REPO / name, backend / name)
    for name in ('src', 'resource', 'tests', 'package.json', 'package-lock.json',
                 'index.html', 'vite.config.ts', 'vitest.config.ts', 'playwright.config.ts',
                 'README.md', 'LICENSE', '.gitignore', 'dist'):
        copy_checked(args.frontend / name, frontend / name)
    for path in args.frontend.glob('tsconfig*.json'):
        copy_checked(path, frontend / path.name)
    if (args.frontend / 'public').exists():
        copy_checked(args.frontend / 'public', frontend / 'public')
    resources = backend / 'resources'
    copy_checked(args.dataset, resources / 'datasets/MilitaryAircraft3D')
    copy_checked(args.features, resources / 'exp/distributed_feature_cache/military_aircraft_vit_fixedsplit_v2')
    copy_checked(args.weights, resources / 'models/ViT-B-16.pt')
    privacy = resources / 'fedmia_local'
    for name in ('show_fedmia_examples.py', 'datasets/OfficeHomeDataset_10072016',
                 'runs/no_defense/config.yaml', 'runs/no_defense/ggeur_fedmia_features',
                 'runs/defense/config.yaml', 'runs/defense/ggeur_fedmia_features'):
        copy_checked(args.privacy / name, privacy / name)
    helper = portable_privacy_helper(privacy / 'show_fedmia_examples.py')
    state = backend / 'exp/platform'
    for name, digest in snapshot.items():
        copy_checked(args.state / name, state / name)
        if sha256(state / name) != digest:
            raise ValueError('experiment changed during snapshot: ' + name)
    if state_snapshot(args.state) != snapshot:
        raise ValueError('experiment state changed; repeat into a new directory')
    subprocess.run([sys.executable, str(backend / 'scripts/export_platform_cache.py'),
                    '--job', args.cache_job], check=True, cwd=output)
    # Source revisions are supplied by the caller and remain descriptive, not
    # a claim that an uncommitted worktree matches a commit exactly.
    metadata = dict(formatVersion=1, sourceLabel=args.source_label, privacyHelper=helper,
                    notes=['No Docker image included.', 'Historical logs/configs are preserved as evidence; '
                           'runtime resolves datasets and checkpoints from this directory.'])
    (output / 'DIRECTORY.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(backend / 'docs/PACKAGE_DIRECTORY.md', output / 'README.md')
    inventory = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                 for p in files(output) if p.name != 'PREPARATION_INCOMPLETE'}
    (output / 'FILES.sha256.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'PREPARATION_INCOMPLETE').unlink()
    print(json.dumps(dict(directory=str(output), files=len(inventory),
                         bytes=sum(v['bytes'] for v in inventory.values()), jobs=len(list(state.glob('jobs/*/job.json'))))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output', 'frontend', 'dataset', 'features', 'weights', 'privacy', 'state'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--cache-job', required=True)
    parser.add_argument('--source-label', default='working copy (see file hashes)')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
