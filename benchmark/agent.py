"""Harbor BaseAgent -> host-side Assay SDK worker -> sandbox-only shell.

Neither credentials, the host repository, nor the task's hidden tests are passed
through the protocol. Harbor installs/runs its verifier after run() returns.
"""
import asyncio
import json
import shlex
from pathlib import Path
from uuid import uuid4

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BYTES = 16_384


def command_wrapper(command: str, timeout_sec: int) -> str:
    # Bound output *inside* the container, before Harbor buffers it on the host.
    # coreutils timeout also terminates timed-out foreground process groups.
    return (
        'd=$(mktemp -d) || exit 1; trap \'rm -rf "$d"\' EXIT; '
        f'timeout --kill-after=5s {timeout_sec}s bash -lc {shlex.quote(command)} '
        '>"$d/out" 2>"$d/err"; rc=$?; '
        f'if [ "$(wc -c <"$d/out")" -gt {OUTPUT_BYTES} ]; then echo "[stdout truncated to tail]"; fi; '
        f'tail -c {OUTPUT_BYTES} "$d/out"; '
        f'if [ "$(wc -c <"$d/err")" -gt {OUTPUT_BYTES} ]; then echo "[stderr truncated to tail]" >&2; fi; '
        f'tail -c {OUTPUT_BYTES} "$d/err" >&2; exit "$rc"'
    )


def validate_exec(message: dict) -> tuple[str, str | None, int]:
    command, cwd, timeout = message.get('command'), message.get('cwd'), message.get('timeout_sec')
    if message.get('type') != 'exec' or type(message.get('id')) is not int:
        raise ValueError('Invalid exec protocol message')
    if not isinstance(command, str) or not 1 <= len(command) <= 32768:
        raise ValueError('Invalid command')
    if cwd is not None and (not isinstance(cwd, str) or not cwd.startswith('/') or '\0' in cwd):
        raise ValueError('cwd must be an absolute container path or null')
    if type(timeout) is not int or not 1 <= timeout <= 120:
        raise ValueError('timeout_sec must be in [1, 120]')
    return command, cwd, timeout


async def execute(environment: BaseEnvironment, message: dict) -> dict:
    try:
        command, cwd, timeout = validate_exec(message)
    except ValueError as error:
        # A model-supplied bad argument is a recoverable tool error, not an
        # adapter crash. Reject it without executing anything in the sandbox.
        return {'stdout': '', 'stderr': str(error), 'exit_code': None, 'validation_error': True}
    try:
        result = await environment.exec(
            command_wrapper(command, timeout), cwd=cwd, timeout_sec=timeout + 15,
        )
        return {'stdout': result.stdout or '', 'stderr': result.stderr or '', 'exit_code': result.return_code}
    except (RuntimeError, TimeoutError) as error:
        return {'stdout': '', 'stderr': str(error), 'exit_code': None, 'execution_error': True}


def apply_usage(context: AgentContext, usage: dict) -> None:
    context.n_input_tokens = usage.get('inputTokens')
    context.n_output_tokens = usage.get('outputTokens')
    context.n_cache_tokens = sum(item.get('cached_tokens', 0) for item in usage.get('inputTokensDetails', []))
    # No guessed prices: Harbor cost stays null unless a pricing source is added.


class AssayAgent(BaseAgent):
    def __init__(self, *args, max_turns: int = 60, max_output_tokens: int = 32768,
                 request_timeout_sec: int = 300, reasoning_effort: str = 'none',
                 api: str = 'chat-completions', credential_profile: str = 'generator',
                 capture_snapshot: bool = True, host_logs_dir: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Scaling runs keep worker/evidence logs outside Harbor's container-
        # writable log mounts. The model never receives this host path.
        if host_logs_dir is not None:
            self.logs_dir = Path(host_logs_dir)
        if not 1 <= max_turns <= 200 or not 256 <= max_output_tokens <= 32768:
            raise ValueError('Invalid generation limits')
        if not 15 <= request_timeout_sec <= 900:
            raise ValueError('request_timeout_sec must be in [15, 900]')
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.request_timeout_sec = request_timeout_sec
        if reasoning_effort not in ('provider', 'none', 'low', 'medium', 'high'):
            raise ValueError('Invalid reasoning_effort')
        self.reasoning_effort = reasoning_effort
        if api not in ('chat-completions', 'responses') or credential_profile not in ('generator', 'baseline'):
            raise ValueError('Invalid API or credential profile')
        self.api, self.credential_profile = api, credential_profile
        self.capture_snapshot = capture_snapshot

    @staticmethod
    def name() -> str:
        return 'assay'

    def version(self) -> str:
        return '0.1.0-benchmark-1'

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec('command -v bash timeout tail mktemp tar', timeout_sec=15)
        if result.return_code:
            raise RuntimeError('Task image must provide bash, coreutils, and tar')

    async def _snapshot(self, environment: BaseEnvironment) -> None:
        # All ten selected tasks write their deliverables under /app. This is an
        # inspection artifact, NOT a complete replayable image or a Git patch.
        target = f'/tmp/assay-app-{uuid4().hex}.tar.gz'
        command = (
            f'ulimit -f 32768; timeout --kill-after=5s 25s tar -czf {shlex.quote(target)} '
            '--exclude=__pycache__ --exclude=.git -C /app .'
        )
        try:
            result = await environment.exec(command, timeout_sec=35)
            if result.return_code:
                raise RuntimeError(f'/app snapshot failed (exit {result.return_code}); possibly exceeds 32 MiB file limit')
            await asyncio.wait_for(environment.download_file(target, self.logs_dir / 'app-before-grading.tar.gz'), timeout=30)
        except Exception as error:
            (self.logs_dir / 'snapshot-error.txt').write_text(str(error))
        finally:
            try:
                await environment.exec(f'rm -f {shlex.quote(target)}', timeout_sec=5)
            except Exception:
                pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        context.metadata = {'implementation': 'src/agents/coder.ts', 'selection': 'none; single independent attempt',
                            'tool_strict': False, 'api': self.api, 'stop_reason': 'running'}
        (self.logs_dir / 'instruction.txt').write_text(instruction)
        stderr = (self.logs_dir / 'worker-stderr.log').open('w')
        events = (self.logs_dir / 'commands.jsonl').open('w')
        process = None
        completed = False
        try:
            # Credentials are inherited only by this host process, never sent
            # via environment.exec(env=...) or installed in the task container.
            process = await asyncio.create_subprocess_exec(
                'node', '--import', 'tsx', str(ROOT / 'benchmark/worker.ts'),
                cwd=ROOT, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=stderr, limit=1024 * 1024,
            )
            assert process.stdin and process.stdout
            init = {'type': 'init', 'instruction': instruction, 'logsDir': str(self.logs_dir.resolve()),
                    'model': self.model_name, 'maxTurns': self.max_turns, 'maxOutputTokens': self.max_output_tokens,
                    'requestTimeoutSec': self.request_timeout_sec, 'reasoningEffort': self.reasoning_effort,
                    'api': self.api, 'credentialProfile': self.credential_profile}
            process.stdin.write((json.dumps(init) + '\n').encode())
            await process.stdin.drain()
            seen = set()
            while line := await process.stdout.readline():
                message = json.loads(line)
                if message.get('type') == 'exec':
                    if type(message.get('id')) is not int:
                        raise RuntimeError('Invalid command id')
                    if message['id'] in seen:
                        raise RuntimeError('Duplicate command id')
                    seen.add(message['id'])
                    events.write(json.dumps({'request': message}) + '\n')
                    events.flush()
                    result = await execute(environment, message)
                    events.write(json.dumps({'id': message['id'], 'result': result}) + '\n')
                    events.flush()
                    process.stdin.write((json.dumps({'type': 'result', 'id': message['id'], 'result': result}) + '\n').encode())
                    await process.stdin.drain()
                elif message.get('type') == 'done':
                    apply_usage(context, message['usage'])
                    stop_reason = message.get('stop_reason', 'completed')
                    if stop_reason not in ('completed', 'turn_limit'):
                        raise RuntimeError('Invalid worker stop reason')
                    context.metadata['stop_reason'] = stop_reason
                    completed = True
                    break
                elif message.get('type') == 'error':
                    raise RuntimeError(f"Assay worker: {message.get('message')}")
                else:
                    raise RuntimeError('Unexpected Assay protocol message')
            if not completed:
                raise RuntimeError('Assay worker exited without a completion message; see worker-stderr.log')
            if await asyncio.wait_for(process.wait(), timeout=10):
                raise RuntimeError('Assay worker exited unsuccessfully')
            if self.capture_snapshot:
                await self._snapshot(environment)
        except asyncio.CancelledError:
            context.metadata['stop_reason'] = 'interrupted'
            raise
        except Exception:
            context.metadata['stop_reason'] = 'error'
            raise
        finally:
            if process and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            if process and process.stdin:
                process.stdin.close()
            events.close()
            stderr.close()
