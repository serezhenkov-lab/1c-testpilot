import asyncio
from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from _file_bridge import BridgeError, Proxy, Worker, atomic_json, channel_path, exclusive, read_json, main
from mcp.types import CallToolResult, TextContent, ImageContent


class FileBridgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.worker = Worker(self.tmp.name, 'test')
        self.worker.session_dir.mkdir()
        self.worker.tools = [{'name': 'tc_test', 'description': 'test', 'inputSchema': {'type': 'object'}}]
        self.worker.publish()
        self.proxy = Proxy(self.tmp.name, 'test', timeout=0.08, poll=0.005)
        self.calls = []

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        return CallToolResult(content=[TextContent(type='text', text='done')], structuredContent={'value': args})

    def pending_request(self):
        with self.assertRaisesRegex(BridgeError, 'Timeout'):
            self.proxy.call('tc_test', {'text': 'Привет'})
        return self.proxy.pending()

    def test_roundtrip_and_clear(self):
        self.proxy.timeout = 2
        def worker_thread():
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if asyncio.run(self.worker.once(self.dispatch)):
                    return
                time.sleep(0.005)
        thread = threading.Thread(target=worker_thread)
        thread.start()
        try:
            result = self.proxy.call('tc_test', {'text': 'Привет'})
        finally:
            thread.join(4)
        self.assertEqual(result.structuredContent, {'value': {'text': 'Привет'}})
        self.assertIsNone(self.proxy.pending())
        self.assertEqual(len(self.calls), 1)

    def test_expired_not_executed(self):
        pending = self.pending_request()
        asyncio.run(self.worker.once(self.dispatch))
        result = self.proxy.collect(pending['request'])
        self.assertTrue(result.isError)
        self.assertEqual(self.calls, [])

    def test_timeout_survives_proxy_restart(self):
        pending = self.pending_request()
        restarted = Proxy(self.tmp.name, 'test')
        with self.assertRaisesRegex(BridgeError, 'Unresolved'):
            restarted.call('tc_test', {})
        self.assertEqual(restarted.pending()['request'], pending['request'])

    def test_late_response_not_reexecuted(self):
        pending = self.pending_request()
        request = self.worker.session_dir / (pending['request'] + '.request.json')
        value = read_json(request)
        value['expires'] = time.time() + 10
        atomic_json(request, value)
        asyncio.run(self.worker.once(self.dispatch))
        with self.assertRaisesRegex(BridgeError, 'Unresolved'):
            self.proxy.call('tc_test', {})
        self.assertFalse(asyncio.run(self.worker.once(self.dispatch)))
        result = self.proxy.collect(pending['request'])
        self.assertFalse(result.isError)
        self.assertEqual(len(self.calls), 1)

    def test_crash_after_claim_never_replayed(self):
        pending = self.pending_request()
        request = self.worker.session_dir / (pending['request'] + '.request.json')
        value = read_json(request)
        value['expires'] = time.time() + 10
        atomic_json(request, value)
        async def fail(name, args):
            self.calls.append(name)
            raise ConnectionError('lost response after action')
        with self.assertRaises(ConnectionError):
            asyncio.run(self.worker.once(fail))
        self.assertFalse(asyncio.run(self.worker.once(self.dispatch)))
        with self.assertRaisesRegex(BridgeError, 'unknown'):
            self.proxy.collect(pending['request'])
        self.assertEqual(len(self.calls), 1)

    def test_worker_restart_isolates_old_queue(self):
        pending = self.pending_request()
        restarted = Worker(self.tmp.name, 'test')
        restarted.session_dir.mkdir()
        restarted.tools = self.worker.tools
        restarted.publish()
        self.assertFalse(asyncio.run(restarted.once(self.dispatch)))
        with self.assertRaisesRegex(BridgeError, 'Unresolved'):
            self.proxy.call('tc_test', {})
        self.assertEqual(self.proxy.pending()['session'], pending['session'])

    def test_wrong_session_response_does_not_release_channel(self):
        pending = self.pending_request()
        atomic_json(self.proxy.response_path(pending), {**pending, 'session': uuid.uuid4().hex, 'result': {'content': []}})
        with self.assertRaisesRegex(BridgeError, 'Session mismatch'):
            self.proxy.collect(pending['request'])
        self.assertIsNotNone(self.proxy.pending())

    def test_invalid_result_does_not_release_channel(self):
        pending = self.pending_request()
        atomic_json(self.proxy.response_path(pending), {**pending, 'result': {'content': 'bad'}})
        with self.assertRaises(ValueError):
            self.proxy.collect(pending['request'])
        self.assertIsNotNone(self.proxy.pending())

    def test_image_and_tool_error_preserved(self):
        pending = self.pending_request()
        result = CallToolResult(isError=True, content=[ImageContent(type='image', mimeType='image/png', data='YWJj')])
        atomic_json(self.proxy.response_path(pending), {**pending, 'result': result.model_dump(mode='json')})
        received = self.proxy.collect(pending['request'])
        self.assertTrue(received.isError)
        self.assertEqual(received.content[0].data, 'YWJj')
        self.assertEqual(self.proxy.collect(pending['request']).content[0].data, 'YWJj')

    def test_other_proxy_cannot_dispatch_while_locked(self):
        with exclusive(self.proxy.directory / 'proxy.lock'):
            with self.assertRaisesRegex(BridgeError, 'busy'):
                Proxy(self.tmp.name, 'test').call('tc_test', {})
        self.assertIsNone(self.proxy.pending())

    def test_wrong_channel_request_not_executed(self):
        pending = self.pending_request()
        request = self.worker.session_dir / (pending['request'] + '.request.json')
        value = read_json(request)
        value.update(channel='different', expires=time.time() + 10)
        atomic_json(request, value)
        with self.assertRaisesRegex(BridgeError, 'channel mismatch'):
            asyncio.run(self.worker.once(self.dispatch))
        self.assertEqual(self.calls, [])

    def test_temporary_files_ignored(self):
        (self.worker.session_dir / 'incomplete.tmp').write_text('{')
        self.assertFalse(asyncio.run(self.worker.once(self.dispatch)))

    def test_stale_worker_rejected_before_publication(self):
        path = self.worker.directory / 'heartbeat.json'
        value = read_json(path)
        value['heartbeat'] -= 100
        atomic_json(path, value)
        with self.assertRaisesRegex(BridgeError, 'offline/stale'):
            self.proxy.call('tc_test', {})
        self.assertIsNone(self.proxy.pending())

    def test_transient_windows_replace_denial_is_retried(self):
        import os
        original = os.replace
        attempts = []
        def replace(source, target):
            attempts.append(target)
            if len(attempts) < 3:
                raise PermissionError('RDP reader temporarily holds file')
            return original(source, target)
        with patch('_file_bridge.os.replace', side_effect=replace):
            self.worker.publish()
        self.assertEqual(len(attempts), 3)
        self.assertTrue(self.proxy.status()['online'])

    def test_rdp_already_exists_is_retried(self):
        import os
        original = os.replace
        attempts = []
        def replace(source, target):
            attempts.append(target)
            if len(attempts) < 3:
                error = FileExistsError('RDP replacement collision')
                error.winerror = 183
                raise error
            return original(source, target)
        with patch('_file_bridge.os.replace', side_effect=replace):
            self.worker.publish()
        self.assertEqual(len(attempts), 3)
        self.assertTrue(self.proxy.status()['online'])

    def test_persistent_rdp_collision_preserves_previous_file(self):
        path = self.worker.directory / 'heartbeat.json'
        before = path.read_bytes()
        error = FileExistsError('RDP replacement collision')
        error.winerror = 183
        with patch('_file_bridge.os.replace', side_effect=error), \
                patch('_file_bridge.time.monotonic', side_effect=[0, 6]):
            with self.assertRaises(FileExistsError):
                self.worker.publish(False)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(path.parent.glob('heartbeat.json.*.tmp')), [])

    def test_other_os_error_is_not_retried(self):
        with patch('_file_bridge.os.replace', side_effect=OSError('disk failure')) as replace:
            with self.assertRaises(OSError):
                self.worker.publish()
        self.assertEqual(replace.call_count, 1)

    def test_heartbeat_does_not_rewrite_schemas(self):
        before = (self.worker.directory / 'current.json').stat().st_mtime_ns
        self.worker.publish(False)
        self.assertEqual((self.worker.directory / 'current.json').stat().st_mtime_ns, before)
        self.assertFalse(self.proxy.status()['online'])

    def test_channels_are_independent(self):
        self.pending_request()
        other = Proxy(self.tmp.name, 'other')
        self.assertIsNone(other.pending())

    def test_exclusive_worker_lock(self):
        path = self.worker.directory / 'worker.lock'
        with exclusive(path):
            with self.assertRaises(BridgeError):
                with exclusive(path):
                    self.fail('Lock acquired twice')
        with exclusive(path):
            pass

    def test_channel_path_restrictions(self):
        for channel in ('../escape', 'a/b', 'a\\b', 'CON', '', 'a' * 65):
            with self.subTest(channel=channel), self.assertRaises(BridgeError):
                channel_path(self.tmp.name, channel)

    def test_disconnected_cleanup_preserves_write_failure(self):
        original = OSError(22, 'redirected drive unavailable')
        with patch('pathlib.Path.open', side_effect=original), \
                patch('pathlib.Path.unlink', side_effect=PermissionError('cleanup denied')):
            with self.assertRaises(OSError) as caught:
                self.worker.publish()
        self.assertIs(caught.exception, original)

    def test_lock_cleanup_preserves_original_failure(self):
        stream = Mock()
        stream.seek.side_effect = [1, 0, OSError('unlock failed')]
        stream.close.side_effect = OSError('close failed')
        original = OSError('heartbeat failed')
        lock = 'msvcrt.locking' if sys.platform == 'win32' else 'fcntl.flock'
        with patch('pathlib.Path.open', return_value=stream), patch(lock):
            with self.assertRaises(OSError) as caught:
                with exclusive('worker.lock'):
                    raise original
        self.assertIs(caught.exception, original)
        stream.close.assert_called_once()

    def test_lock_cleanup_failure_without_original_is_reported(self):
        stream = Mock()
        stream.seek.side_effect = [1, 0, OSError('unlock failed')]
        lock = 'msvcrt.locking' if sys.platform == 'win32' else 'fcntl.flock'
        with patch('pathlib.Path.open', return_value=stream), patch(lock):
            with self.assertRaisesRegex(OSError, 'unlock failed'):
                with exclusive('worker.lock'):
                    pass
        stream.close.assert_called_once()

    def test_worker_disconnect_exits_contexts_without_masking_error(self):
        original = OSError(22, 'heartbeat write failed')
        closed = []
        client = Mock(initialize=AsyncMock(), list_tools=AsyncMock(return_value=Mock(tools=[])))
        @asynccontextmanager
        async def stdio(params):
            try:
                yield None, None
            except Exception as exc:
                raise ExceptionGroup('stdio', [exc])
            finally:
                closed.append('stdio')
        @asynccontextmanager
        async def session(*args, **kwargs):
            try:
                yield client
            finally:
                closed.append('session')
        worker = Worker(self.tmp.name, 'disconnect')
        with patch('mcp.client.stdio.stdio_client', stdio), patch('mcp.ClientSession', session), \
                patch.object(worker, 'publish', side_effect=[None, PermissionError('offline write failed')]), \
                patch.object(worker, 'once', new=AsyncMock(side_effect=original)):
            with self.assertRaises(ExceptionGroup) as caught:
                asyncio.run(worker.run())
        self.assertIs(caught.exception.exceptions[0], original)
        self.assertEqual(closed, ['session', 'stdio'])
        self.assertTrue(worker.poisoned)

    def test_worker_cli_disconnect_has_message_and_local_log(self):
        error = ExceptionGroup('stdio', [ExceptionGroup('session', [OSError(22, 'drive lost')])])
        output = io.StringIO()
        with patch('_file_bridge.Worker.run', new=AsyncMock(side_effect=error)), \
                patch('tempfile.tempdir', self.tmp.name), patch('sys.stderr', output):
            with self.assertRaises(SystemExit) as caught:
                main(['worker', '--root', self.tmp.name, '--channel', 'test'])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn('Reconnect RDP', output.getvalue())
        self.assertIn('Commands are NOT replayed', output.getvalue())
        self.assertNotIn('Traceback', output.getvalue())
        logs = list(Path(self.tmp.name).glob('testpilot-worker-*.log'))
        self.assertEqual(len(logs), 1)
        self.assertIn('drive lost', logs[0].read_text(encoding='utf-8'))

    def test_worker_cli_does_not_hide_mixed_programming_error(self):
        error = ExceptionGroup('mixed', [OSError('drive lost'), ValueError('bug')])
        with patch('_file_bridge.Worker.run', new=AsyncMock(side_effect=error)):
            with self.assertRaises(ExceptionGroup):
                main(['worker', '--root', self.tmp.name, '--channel', 'test'])

    def test_response_publication_failure_does_not_repeat_action(self):
        pending = self.pending_request()
        path = self.worker.session_dir / (pending['request'] + '.request.json')
        value = read_json(path)
        value['expires'] = time.time() + 10
        atomic_json(path, value)
        with patch('_file_bridge.atomic_json', side_effect=OSError('share disconnected')):
            with self.assertRaises(OSError):
                asyncio.run(self.worker.once(self.dispatch))
        self.assertFalse(asyncio.run(self.worker.once(self.dispatch)))
        self.assertEqual(len(self.calls), 1)
        self.assertIsNotNone(self.proxy.pending())


if __name__ == '__main__':
    unittest.main()
