"""Publish the approved frontend on 4090lziy without restarting the backend.

Only frontend/docs source changes are accepted. Old assets and the previous
index remain available for existing browser sessions and rollback.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path('/root/autodl-tmp/FederatedScope-worktrees/single-host-platform')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def run(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def publish(args):
    require(ROOT.resolve() == ROOT, 'Unexpected deployment root')
    uploads = ROOT / 'exp/platform_release'
    require(uploads.resolve() == uploads, 'Release directory must not be a symlink')
    for path, expected in [(args.archive, args.archive_sha), (args.bundle, args.bundle_sha)]:
        require(path.resolve().parent == uploads, 'Upload must be in platform_release')
        require(digest(path) == expected, 'Upload checksum mismatch')
    require(re.fullmatch(r'codex/[a-z0-9][a-z0-9/-]*', args.branch), 'Unexpected branch name')
    require(run('git', 'rev-parse', 'HEAD') == args.base, 'Server HEAD changed')
    require(not run('git', 'status', '--porcelain'), 'Server has uncommitted changes')
    previous_branch = run('git', 'symbolic-ref', '--short', 'HEAD')
    service = 'federatedscope-platform.service'
    pid_before = run('systemctl', 'show', service, '--property=MainPID', '--value')
    require(pid_before != '0', 'Backend is not running')

    stage = Path(tempfile.mkdtemp(prefix='photo-live-', dir=uploads))
    seen, total = set(), 0
    with tarfile.open(args.archive, 'r:gz') as archive:
        for item in archive.getmembers():
            path = PurePosixPath(item.name)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in item.name,
                    'Unsafe archive path')
            if item.isdir():
                require(str(path) in ('.', 'assets'), 'Unexpected directory')
                continue
            require(item.isfile(), 'Links and special files are not allowed')
            require(str(path) == 'index.html' or (len(path.parts) == 2 and path.parts[0] == 'assets'),
                    'Archive must contain only the static build')
            require(str(path) not in seen, 'Duplicate archive member')
            seen.add(str(path))
            total += item.size
            require(item.size <= 50_000_000 and total <= 200_000_000, 'Unexpected build size')
            destination = stage.joinpath(*path.parts)
            destination.parent.mkdir(exist_ok=True)
            with archive.extractfile(item) as source, destination.open('xb') as target:
                shutil.copyfileobj(source, target)
    require((stage / 'index.html').is_file() and (stage / 'assets').is_dir(), 'Incomplete build')
    index_text = (stage / 'index.html').read_text()
    entry_assets = re.findall(r'(?:src|href)="(/assets/[^"?#]+)"', index_text)
    require(entry_assets, 'No entry assets')
    require(all((stage / asset.lstrip('/')).is_file() for asset in entry_assets), 'Missing entry asset')

    dist = ROOT / 'frontend/dist'
    assets = dist / 'assets'
    index = dist / 'index.html'
    require(dist.resolve() == dist and assets.resolve() == assets and index.resolve() == index,
            'Static paths must not be symlinks')
    require(index.is_file() and assets.is_dir(), 'No existing frontend to back up')
    for asset in (stage / 'assets').iterdir():
        target = assets / asset.name
        require(not target.is_symlink(), 'Existing asset is a symlink')
        require(not target.exists() or digest(target) == digest(asset), 'Hashed asset collision')

    run('git', 'bundle', 'verify', str(args.bundle))
    run('git', 'fetch', str(args.bundle), args.branch + ':refs/heads/' + args.branch)
    require(run('git', 'rev-parse', args.branch) == args.commit, 'Unexpected source commit')
    run('git', 'merge-base', '--is-ancestor', args.base, args.commit)
    changes = run('git', 'diff', '--name-only', args.base, args.commit).splitlines()
    require(changes and all(name.startswith(('frontend/', 'docs/')) for name in changes),
            'Release contains non-frontend changes')

    backup = stage / 'index.previous.html'
    shutil.copy2(index, backup)
    switched = published = False
    try:
        run('git', 'switch', args.branch)
        switched = True
        for asset in (stage / 'assets').iterdir():
            target = assets / asset.name
            if not target.exists():
                with asset.open('rb') as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
        candidate = dist / ('index-' + stage.name + '.tmp')
        shutil.copy2(stage / 'index.html', candidate)
        os.replace(candidate, index)
        published = True
        with urlopen('http://127.0.0.1:8001/', timeout=20) as response:
            served = hashlib.sha256(response.read()).hexdigest()
        require(served == digest(index), 'Backend did not serve the new index')
        with urlopen('http://127.0.0.1:8001/api/health', timeout=20) as response:
            health = json.load(response)
        require(health['data']['status'] == 'ok', 'Health check failed')
        pid_after = run('systemctl', 'show', service, '--property=MainPID', '--value')
        require(pid_before == pid_after, 'Backend PID changed during release')
    except Exception:
        if published:
            candidate = dist / ('rollback-' + stage.name + '.tmp')
            shutil.copy2(backup, candidate)
            os.replace(candidate, index)
        if switched:
            run('git', 'switch', previous_branch)
        raise
    print(json.dumps({'commit': args.commit, 'branch': args.branch, 'backup': str(backup),
                      'indexSha256': served, 'backendPid': pid_after, 'health': health}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--branch', required=True)
    publish(parser.parse_args())
