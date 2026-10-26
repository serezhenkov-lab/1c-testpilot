import asyncio
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from _file_bridge import BridgeError, Proxy, Worker, atomic_json, channel_path, exclusive, read_json
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


if __name__ == '__main__':
    unittest.main()
