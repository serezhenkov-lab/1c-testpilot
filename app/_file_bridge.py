"""Optional file transport for MCP calls across a redirected RDP drive.

The worker owns the unmodified Testpilot stdio server. Never replay a claimed
request: a missing response is an unknown outcome, not a failed UI operation.
"""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import traceback
import uuid

PROTOCOL = 1
MAX_BYTES = 32 * 1024 * 1024
LEASE_SECONDS = 30
PREFIX = 'tp_bridge_'


class BridgeError(RuntimeError):
    pass


def cleanup_io(action):
    """Keep the original failure when the disconnected share also rejects cleanup."""
    unwinding = sys.exc_info()[0] is not None
    try:
        action()
    except OSError:
        if not unwinding:
            raise


def io_failure(exc):
    if isinstance(exc, BaseExceptionGroup):
        return all(io_failure(child) for child in exc.exceptions)
    return isinstance(exc, OSError)


def report_worker_failure(exc, channel):
    # The redirected drive is unavailable: diagnostics must stay on the terminal.
    log = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         prefix='testpilot-worker-', suffix='.log',
                                         delete=False) as stream:
            traceback.print_exception(exc, file=stream)
            log = stream.name
    except OSError:
        pass
    print(f'Worker stopped: channel={channel}. File/transport I/O failed; '
          'the RDP redirected drive may be disconnected or inaccessible.\n'
          'Reconnect RDP, check the redirected drive, then run the worker startup script again.\n'
          'Commands are NOT replayed. If a request was in progress, its outcome may be unknown; '
          'check tp_bridge_status/result and 1C before repeating any action.', file=sys.stderr, flush=True)
    if log:
        print(f'Local diagnostic log: {log}', file=sys.stderr, flush=True)
    else:
        print('Local diagnostic log could not be written.', file=sys.stderr, flush=True)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{32}', value):
        raise BridgeError('Invalid session/request identifier')
    return value


def channel_path(root, channel):
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', channel)
            or re.fullmatch(r'(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])', channel)):
        raise BridgeError('Invalid channel name')
    return Path(root).resolve() / channel.lower()


def read_json(path):
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise BridgeError('JSON exceeds transport size limit')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise BridgeError('Expected a JSON object')
    return value


def atomic_json(path, value):
    path = Path(path)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(raw) > MAX_BYTES:
        raise BridgeError('JSON exceeds transport size limit')
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Windows/RDP may briefly deny replacing a file while the other side
        # reads it. Retrying publication of identical bytes cannot replay an action.
        deadline = time.monotonic() + 5
        delay = .02
        while True:
            try:
                os.replace(temp, path)
                break
            except OSError as exc:
                # RDP redirected drives can report ERROR_ALREADY_EXISTS (183)
                # during replacement, in addition to temporary sharing denials.
                # Retry the same publication; never unlink the existing target.
                if not isinstance(exc, PermissionError) and getattr(exc, 'winerror', None) != 183:
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, .25)
    finally:
        cleanup_io(lambda: temp.unlink(missing_ok=True))


@contextmanager
def exclusive(path):
    """OS lock released on process exit, including on Windows/redirected drives."""
    stream = Path(path).open('a+b')
    try:
        if stream.seek(0, 2) == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BridgeError('Channel is busy (another proxy/worker holds the lock)') from exc
        try:
            yield
        finally:
            def unlock():
                stream.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            cleanup_io(unlock)
    finally:
        cleanup_io(stream.close)


def envelope(value, channel, session=None, request=None):
    if value.get('protocol') != PROTOCOL or value.get('channel') != channel:
        raise BridgeError('Protocol/channel mismatch')
    identifier(value.get('session'))
    if session is not None and value['session'] != session:
        raise BridgeError('Session mismatch')
    if request is not None and value.get('request') != request:
        raise BridgeError('Request mismatch')
    return value


class Proxy:
    def __init__(self, root, channel, timeout=120, poll=0.25):
        if not math.isfinite(timeout) or timeout <= 0:
            raise BridgeError('Timeout must be positive and finite')
        self.directory = channel_path(root, channel)
        self.channel = self.directory.name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timeout, self.poll = timeout, poll

    def manifest(self, live=False):
        manifest = envelope(read_json(self.directory / 'current.json'), self.channel)
        if live:
            heartbeat_path = self.directory / 'heartbeat.json'
            heartbeat = (envelope(read_json(heartbeat_path), self.channel, manifest['session'])
                         if heartbeat_path.exists() else manifest)
            age = time.time() - float(heartbeat['heartbeat'])
            if not math.isfinite(age) or age < -LEASE_SECONDS or age > LEASE_SECONDS or not heartbeat.get('online'):
                raise BridgeError('Worker offline/stale; check terminal and clock synchronization')
        return manifest

    def pending(self):
        path = self.directory / 'pending.json'
        if not path.exists():
            return None
        value = envelope(read_json(path), self.channel)
        identifier(value.get('request'))
        return value

    def response_path(self, pending):
        return self.directory / identifier(pending['session']) / (identifier(pending['request']) + '.response.json')

    def status(self):
        result = {'channel': self.channel}
        try:
            manifest = self.manifest(live=True)
            result.update(online=True, session=manifest['session'])
        except (OSError, ValueError, KeyError, BridgeError) as exc:
            result.update(online=False, error=str(exc))
        pending = self.pending()
        result['pending'] = pending
        result['response_available'] = bool(pending and self.response_path(pending).is_file())
        last = self.directory / 'last-result.json'
        result['last_request'] = read_json(last).get('request') if last.exists() else None
        return result

    def _receive(self, pending):
        from mcp.types import CallToolResult
        response = envelope(read_json(self.response_path(pending)), self.channel,
                            pending['session'], pending['request'])
        result = CallToolResult.model_validate(response['result'])
        # Persist delivery before releasing the channel. Result files stay for diagnosis.
        atomic_json(self.directory / 'last-result.json', response)
        (self.directory / 'pending.json').unlink()
        return result

    def collect(self, request):
        identifier(request)
        with exclusive(self.directory / 'proxy.lock'):
            pending = self.pending()
            if not pending or pending['request'] != request:
                # MCP caller may have lost delivery after the proxy received the result.
                # Readback is idempotent and must not release another pending action.
                last = self.directory / 'last-result.json'
                if last.is_file():
                    response = envelope(read_json(last), self.channel, request=request)
                    from mcp.types import CallToolResult
                    return CallToolResult.model_validate(response['result'])
                raise BridgeError('No matching pending request')
            if not self.response_path(pending).exists():
                raise BridgeError('Outcome still unknown; do not retry the UI action')
            return self._receive(pending)

    def call(self, name, arguments):
        with exclusive(self.directory / 'proxy.lock'):
            pending = self.pending()
            if pending:
                raise BridgeError(f"Unresolved request {pending['request']}; use tp_bridge_result, never resend the action")
            manifest = self.manifest(live=True)
            if name.startswith(PREFIX) or name not in {t['name'] for t in manifest['tools']}:
                raise BridgeError('Tool not advertised by worker')
            if not isinstance(arguments, dict):
                raise BridgeError('Tool arguments must be an object')
            pending = dict(protocol=PROTOCOL, channel=self.channel, session=manifest['session'],
                           request=uuid.uuid4().hex, tool=name, created=time.time(),
                           expires=time.time() + self.timeout)
            directory = self.directory / pending['session']
            # Persist intent FIRST. Any error after this point conservatively blocks the channel.
            atomic_json(self.directory / 'pending.json', pending)
            atomic_json(directory / (pending['request'] + '.request.json'), {**pending, 'arguments': arguments})
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                if self.response_path(pending).is_file():
                    return self._receive(pending)
                time.sleep(self.poll)
            raise BridgeError(f"Timeout; outcome unknown for {pending['request']}. Use tp_bridge_status/result; do not repeat")


class Worker:
    def __init__(self, root, channel):
        self.directory = channel_path(root, channel)
        self.channel = self.directory.name
        self.session = uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.directory / self.session
        self.tools = []
        self.published_tools = None
        self.poisoned = False

    def publish(self, online=True):
        if self.published_tools != self.tools:
            atomic_json(self.directory / 'current.json', dict(protocol=PROTOCOL, channel=self.channel,
                        session=self.session, heartbeat=time.time(), online=online, tools=self.tools))
            self.published_tools = self.tools
        atomic_json(self.directory / 'heartbeat.json', dict(protocol=PROTOCOL, channel=self.channel,
                    session=self.session, heartbeat=time.time(), online=online))

    async def heartbeat(self):
        while True:
            await asyncio.sleep(5)
            self.publish(not self.poisoned)

    async def once(self, dispatch):
        from mcp.types import CallToolResult, TextContent
        for path in sorted(self.session_dir.glob('*.request.json')):
            request = identifier(path.name.removesuffix('.request.json'))
            running = path.with_name(request + '.running')
            if running.exists() or path.with_name(request + '.response.json').exists():
                continue
            # Atomic claim before executing. .running is never picked up again.
            path.rename(running)
            value = envelope(read_json(running), self.channel, self.session, request)
            expires = value.get('expires')
            if not isinstance(expires, (int, float)) or not math.isfinite(expires):
                raise BridgeError('Invalid expiry')
            if time.time() >= expires:
                result = CallToolResult(isError=True, content=[TextContent(type='text', text='Request expired before execution; no action performed')])
            else:
                if value.get('tool') not in {t['name'] for t in self.tools} or not isinstance(value.get('arguments'), dict):
                    raise BridgeError('Invalid tool/arguments')
                # A transport exception may mean that 1C executed the action. Leave .running,
                # stop this worker and require reconciliation; never manufacture a retryable error.
                result = await dispatch(value['tool'], value['arguments'])
            atomic_json(path.with_name(request + '.response.json'), dict(protocol=PROTOCOL,
                        channel=self.channel, session=self.session, request=request,
                        result=result.model_dump(mode='json', by_alias=True, exclude_none=True)))
            return True
        return False

    async def run(self, env_file=None, profiles=None):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        with exclusive(self.directory / 'worker.lock'):
            self.session_dir.mkdir()
            args = ['-m', '_cli'] + (['--env-file', str(Path(env_file).resolve())] if env_file else [])
            # stdio is internal to the terminal, regardless of the operator's HTTP settings.
            env = {**os.environ, 'TC1C_TRANSPORT': 'stdio'}
            if profiles:
                env['TC1C_PROFILES_FILE'] = str(Path(profiles).resolve())
            params = StdioServerParameters(command=sys.executable, args=args, env=env)
            heartbeat = None
            try:
                async with stdio_client(params) as (reader, writer):
                    async with ClientSession(reader, writer, read_timeout_seconds=timedelta(hours=1)) as client:
                        await client.initialize()
                        self.tools = [t.model_dump(mode='json', by_alias=True, exclude_none=True)
                                      for t in (await client.list_tools()).tools]
                        self.publish()
                        print(f'Worker ready: channel={self.channel} session={self.session}', file=sys.stderr, flush=True)
                        heartbeat = asyncio.create_task(self.heartbeat())
                        while True:
                            if heartbeat.done():
                                heartbeat.result()  # Lost share/lease: stop rather than continue silently.
                            if await self.once(client.call_tool):
                                self.tools = [t.model_dump(mode='json', by_alias=True, exclude_none=True)
                                              for t in (await client.list_tools()).tools]
                                self.publish()
                            await asyncio.sleep(0.25)
            finally:
                self.poisoned = True
                if heartbeat:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                cleanup_io(lambda: self.publish(False))


async def serve(proxy):
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, TextContent, Tool
    server = Server('1c-testpilot-file-bridge')
    control = [Tool(name=PREFIX + 'status', description='File bridge health and unresolved request. Read-only; never resends.',
                    inputSchema={'type': 'object', 'properties': {}, 'additionalProperties': False}),
               Tool(name=PREFIX + 'result', description='Collect a late response by request id, releasing the channel. Never executes an action.',
                    inputSchema={'type': 'object', 'properties': {'request': {'type': 'string'}},
                                 'required': ['request'], 'additionalProperties': False})]

    @server.list_tools()
    async def list_tools():
        try:
            remote = [Tool.model_validate(t) for t in proxy.manifest()['tools'] if not t['name'].startswith(PREFIX)]
        except (OSError, ValueError, KeyError, BridgeError):
            remote = []
        return control + remote

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        try:
            if name == PREFIX + 'status':
                value = await asyncio.to_thread(proxy.status)
                return CallToolResult(content=[TextContent(type='text', text=json.dumps(value))], structuredContent=value)
            if name == PREFIX + 'result':
                return await asyncio.to_thread(proxy.collect, arguments['request'])
            return await asyncio.to_thread(proxy.call, name, arguments)
        except (OSError, ValueError, KeyError, BridgeError) as exc:
            return CallToolResult(isError=True, content=[TextContent(type='text', text=str(exc))])

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['proxy', 'worker'])
    parser.add_argument('--root', required=True, help=r'Shared exchange directory; terminal may use \\tsclient\C\...')
    parser.add_argument('--channel', required=True)
    parser.add_argument('--timeout', type=float, default=120, help='Proxy wait/expiry in seconds; no automatic retries')
    parser.add_argument('--env-file', help='Worker-only Testpilot configuration on the terminal')
    parser.add_argument('--profiles', help='Worker-only YAML/JSON connection profiles on the terminal')
    args = parser.parse_args(argv)
    try:
        if args.mode == 'worker':
            asyncio.run(Worker(args.root, args.channel).run(args.env_file, args.profiles))
        else:
            asyncio.run(serve(Proxy(args.root, args.channel, args.timeout)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if args.mode != 'worker' or not io_failure(exc):
            raise
        report_worker_failure(exc, args.channel)
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
