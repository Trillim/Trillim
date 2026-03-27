"""Tests for STT admission control."""

from __future__ import annotations

import asyncio
import unittest

from trillim.components.stt._admission import TranscriptionAdmission
from trillim.errors import AdmissionRejectedError


class STTAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_acquire_allows_only_one_active_request(self):
        admission = TranscriptionAdmission()
        lease = await admission.acquire()
        self.assertEqual(admission.active_count, 1)
        with self.assertRaisesRegex(AdmissionRejectedError, "STT is busy"):
            await admission.acquire()
        await lease.release()
        self.assertEqual(admission.active_count, 0)

    async def test_start_draining_rejects_new_work_until_finish_starting(self):
        admission = TranscriptionAdmission()
        await admission.start_draining()
        with self.assertRaisesRegex(AdmissionRejectedError, "draining"):
            await admission.acquire()
        await admission.finish_starting()
        lease = await admission.acquire()
        await lease.release()

    async def test_wait_for_idle_blocks_until_release(self):
        admission = TranscriptionAdmission()
        lease = await admission.acquire()
        waiter = asyncio.create_task(admission.wait_for_idle())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        await lease.release()
        await waiter

    async def test_lease_release_is_idempotent(self):
        admission = TranscriptionAdmission()
        lease = await admission.acquire()
        await lease.release()
        await lease.release()
        self.assertEqual(admission.active_count, 0)

    async def test_accepting_property_and_idle_paths_cover_no_active_transitions(self):
        admission = TranscriptionAdmission()

        self.assertTrue(admission.accepting)
        await admission.start_draining()
        self.assertFalse(admission.accepting)
        await admission.wait_for_idle()
        await admission.finish_starting()
        self.assertTrue(admission.accepting)
        await admission._release()
        self.assertEqual(admission.active_count, 0)

    async def test_internal_release_and_finish_starting_cover_active_nonzero_branches(self):
        admission = TranscriptionAdmission()
        admission._active = 2
        admission._idle.clear()

        await admission.finish_starting()
        self.assertFalse(admission._idle.is_set())

        await admission._release()
        self.assertEqual(admission.active_count, 1)
        self.assertFalse(admission._idle.is_set())
