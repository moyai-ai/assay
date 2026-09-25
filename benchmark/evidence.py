"""Host-collected evidence. Never extract candidate archives or execute their files.

Docker pause keeps processes/files in the original environment intact through the
selection barrier. Only the task's main container is supported by this experiment.
"""
import asyncio
import difflib
import hashlib
import json
import os
import re
import subprocess
import tarfile
import threading
from pathlib import Path, PurePosixPath
from uuid import uuid4

IGNORED = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache', '.cache'}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def save_json(path: Path, value, *, exclusive=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if exclusive else path.with_name(f'.{path.name}-{uuid4().hex}.tmp')
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if not exclusive:
            os.replace(target, path)
    finally:
        if not exclusive:
            target.unlink(missing_ok=True)


class LimitedReader:
    def __init__(self, stream, limit):
        self.stream, self.remaining = stream, limit

    def read(self, size=-1):
        if size < 0 or size > self.remaining:
            raise ValueError('Candidate archive exceeds byte limit')
        data = self.stream.read(size)
        self.remaining -= len(data)
        return data


def read_tree(stream, byte_limit=256 * 1024 * 1024, text_limit=8 * 1024 * 1024) -> dict:
    files, retained, members, logical_bytes = {}, 0, 0, 0
    with tarfile.open(fileobj=LimitedReader(stream, byte_limit), mode='r|') as archive:
        for entry in archive:
            members += 1
            if members > 100_000:
                raise ValueError('Too many candidate archive entries')
            path = PurePosixPath(entry.name)
            if path.is_absolute() or '..' in path.parts or len(entry.name) > 4096:
                raise ValueError('Unsafe candidate archive path')
            if any(part in IGNORED for part in path.parts) or entry.isdir():
                continue
            name = str(path)
            if name in files:
                raise ValueError('Duplicate candidate archive path')
            item = {'mode': entry.mode, 'size': entry.size}
            if entry.isfile():
                # Sparse tar entries can describe far more logical data than
                # the bytes read from the archive. Bound hashing work as well.
                logical_bytes += entry.size
                if entry.size < 0 or logical_bytes > byte_limit:
                    raise ValueError('Candidate file contents exceed byte limit')
                item['kind'] = 'file'
                source = archive.extractfile(entry)  # Read bytes only; no filesystem extraction.
                if source is None:
                    raise ValueError('Missing archive member body')
                sha, chunks, size = hashlib.sha256(), [], 0
                keep = entry.size <= 256 * 1024 and retained + entry.size <= text_limit
                while data := source.read(64 * 1024):
                    sha.update(data)
                    size += len(data)
                    if keep:
                        chunks.append(data)
                if size != entry.size:
                    raise ValueError('Truncated archive member')
                item['sha256'] = sha.hexdigest()
                if keep:
                    raw = b''.join(chunks)
                    try:
                        if b'\0' in raw:
                            raise UnicodeError('binary')
                        item['text'] = raw.decode('utf8')
                        retained += size
                    except UnicodeError:
                        item['omitted'] = 'binary or non-UTF-8 content; hash only'
                else:
                    item['omitted'] = 'file/text evidence limit; hash only'
            elif entry.issym() or entry.islnk():
                item.update(kind='symlink' if entry.issym() else 'hardlink', target=entry.linkname)
            else:
                raise ValueError('Special files are not supported in candidate evidence')
            files[name] = item
    manifest = {p: {k: v for k, v in item.items() if k not in ('text', 'omitted')} for p, item in sorted(files.items())}
    return {'sha256': digest(manifest), 'files': files, 'text_bytes': retained,
            'scope': '/app excluding ' + ', '.join(sorted(IGNORED))}


def candidate_evidence(candidate_id: str, base_sha: str, before: dict, after: dict, commands: Path) -> dict:
    changes = []
    for name in sorted(before['files'].keys() | after['files'].keys()):
        old, new = before['files'].get(name), after['files'].get(name)
        comparable = lambda item: {k: v for k, v in (item or {}).items() if k not in ('text', 'omitted')}
        if comparable(old) == comparable(new):
            continue
        change = {'path': name, 'before': comparable(old), 'after': comparable(new)}
        if (old is None or 'text' in old) and (new is None or 'text' in new):
            change['diff'] = ''.join(difflib.unified_diff(
                (old or {}).get('text', '').splitlines(keepends=True),
                (new or {}).get('text', '').splitlines(keepends=True),
                fromfile=f'before/{name}', tofile=f'after/{name}'))
        else:
            change['content_not_reviewed'] = True
            # Still expose the readable side of a binary/text transition.
            if new and 'text' in new:
                change['new_text'] = new['text']
        changes.append(change)
    # These are host-recorded results of agent-initiated commands, NOT official
    # or independently authored grading tests. Read only the bounded tail; many
    # tool calls must not require loading the entire command log into host RAM.
    tail = ''
    if commands.exists():
        with commands.open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 16_384))
            tail = stream.read(16_384).decode('utf8', errors='replace')
    return {'id': candidate_id, 'baseSha': base_sha,
            'diff': json.dumps({'scope': after['scope'], 'changes': changes}, ensure_ascii=True),
            'testOutput': 'Untrusted agent-initiated shell commands/results, recorded by host; '
                          'not official grading or independently authored tests. Tail may be incomplete:\n' + tail}


async def docker(*args: str, timeout=60) -> str:
    process = await asyncio.create_subprocess_exec('docker', *args, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
        if process.returncode:
            raise RuntimeError(f'Docker {args[0]} failed: {err.decode(errors="replace")[-2000:]}')
        return out.decode().strip()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class DockerWorkspace:
    def __init__(self, container_id: str, image_id: str, byte_limit: int, image_fingerprint: str | None = None):
        self.container_id, self.image_id, self.byte_limit = container_id, image_id, byte_limit
        self.image_fingerprint = image_fingerprint or image_id
        self.paused = False

    @classmethod
    async def resolve(cls, environment, byte_limit):
        project = re.sub(r'[^a-z0-9_-]', '-', environment.session_id.lower()).lstrip('-_')
        ids = (await docker('ps', '-aq', '--filter', f'label=com.docker.compose.project={project}',
                            '--filter', 'label=com.docker.compose.service=main')).splitlines()
        if len(ids) != 1 or not re.fullmatch(r'[0-9a-f]{12,64}', ids[0]):
            raise RuntimeError('Expected exactly one Harbor main Docker container')
        image = await docker('inspect', '--format', '{{.Image}}', ids[0])
        # Compose injects per-project image labels, so image IDs can differ even
        # for identical cached filesystem layers. Compare layers and effective
        # image runtime configuration, excluding names/labels/build timestamps.
        metadata = json.loads(await docker('image', 'inspect', image))[0]
        config = metadata.get('Config') or {}
        fingerprint = digest({'layers': metadata['RootFS'], 'os': metadata['Os'],
                              'architecture': metadata['Architecture'],
                              'runtime': {k: config.get(k) for k in ('Env', 'Cmd', 'Entrypoint', 'WorkingDir', 'User', 'Volumes', 'ExposedPorts')}})
        return cls(ids[0], image, byte_limit, fingerprint)

    async def pause(self):
        if not self.paused:
            # Reconcile state even if the caller is cancelled while pause runs.
            try:
                await docker('pause', self.container_id)
            finally:
                self.paused = (await docker('inspect', '--format', '{{.State.Paused}}', self.container_id)) == 'true'
            if not self.paused:
                raise RuntimeError('Candidate environment did not pause')

    async def unpause(self):
        if self.paused:
            await docker('unpause', self.container_id)
            self.paused = False

    def _capture(self):
        process = subprocess.Popen(['docker', 'cp', f'{self.container_id}:/app/.', '-'],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        timer = threading.Timer(90, process.kill)
        timer.start()
        try:
            tree = read_tree(process.stdout, self.byte_limit)
            # tar ends before Docker necessarily closes its pipe. Drain bounded
            # padding; do not let a child hang while writing the last blocks.
            padding = process.stdout.read(1024 * 1024 + 1)
            if len(padding) > 1024 * 1024 or process.wait(timeout=5):
                raise RuntimeError('Docker evidence copy failed')
            return tree
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()

    async def capture(self):
        if not self.paused:
            raise RuntimeError('Evidence must be collected from a paused workspace')
        # Wait for the bounded copy to finish before allowing teardown on cancel.
        job = asyncio.create_task(asyncio.to_thread(self._capture))
        try:
            return await asyncio.shield(job)
        except asyncio.CancelledError:
            await job
            raise


async def browser_health(environment):
    """No benchmark inputs: prove a real browser can execute a synthetic alert.

    Mirrors the grader's driver discovery/options rather than silently installing
    a different browser or changing official assertions.
    """
    import shlex
    script = '''from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
o=Options()
for a in ["--headless","--no-sandbox","--disable-dev-shm-usage","--disable-gpu"]: o.add_argument(a)
d=webdriver.Chrome(options=o)
try:
 d.get("data:text/html,<script>alert('assay-health')</script>")
 WebDriverWait(d,5).until(EC.alert_is_present())
 assert d.switch_to.alert.text == "assay-health"
finally: d.quit()
print("ASSAY_BROWSER_HEALTH_OK")
'''
    result = await environment.exec('python3 -I -c ' + shlex.quote(script), timeout_sec=90)
    if result.return_code or 'ASSAY_BROWSER_HEALTH_OK' not in (result.stdout or ''):
        raise RuntimeError('Browser health check failed; cannot trust HTML XSS grading. '
                           'Use a working native task image/driver (try --force-build). '
                           + f'exit={result.return_code}; ' + ((result.stderr or '') + (result.stdout or ''))[-1500:])
