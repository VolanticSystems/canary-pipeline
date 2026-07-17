"""Tests for stuck-host detection and auto-reroll logic.

Added 2026-07-15 after two incidents in one week where a Vast host went bad
mid-provision (one died outright, one retry-looped on a Docker layer for
6+ minutes) and a human had to notice and manually destroy+retry. These
tests exercise the detection state machine in isolation, without touching
the network — `wait_for_ssh` is tested via a fake `vastai()` call sequence.
"""
from unittest.mock import patch

import pytest

import vast_provision as vp


def _info(status, msg=""):
    return {"actual_status": status, "status_msg": msg,
            "ssh_host": "host.example", "ssh_port": 1234}


class TestWaitForSshStuckDetection:
    def test_immediate_running_returns_host_port(self):
        with patch.object(vp, "vastai", return_value='{"actual_status": "running", "status_msg": "ok", "ssh_host": "h", "ssh_port": 22}'), \
             patch.object(vp.time, "sleep"):
            host, port = vp.wait_for_ssh(12345, timeout=100)
            assert host == "h"
            assert port == 22

    def test_offline_raises_stuck_host_error_immediately(self):
        with patch.object(vp, "vastai", return_value='{"actual_status": "offline", "status_msg": "", "ssh_host": null, "ssh_port": null}'.replace("null", "null")), \
             patch.object(vp.time, "sleep"):
            with pytest.raises(vp.StuckHostError, match="offline"):
                vp.wait_for_ssh(12345, timeout=100)

    def test_progressing_pull_never_flagged_stuck(self):
        # Each poll shows a DIFFERENT SHA/message — genuine progress, must
        # never raise StuckHostError even after many polls.
        responses = [
            '{"actual_status": "loading", "status_msg": "aaa: Pull complete"}',
            '{"actual_status": "loading", "status_msg": "bbb: Pull complete"}',
            '{"actual_status": "loading", "status_msg": "ccc: Pull complete"}',
            '{"actual_status": "loading", "status_msg": "ddd: Pull complete"}',
            '{"actual_status": "loading", "status_msg": "eee: Pull complete"}',
            '{"actual_status": "running", "status_msg": "success", "ssh_host": "h", "ssh_port": 22}',
        ]
        with patch.object(vp, "vastai", side_effect=responses), \
             patch.object(vp.time, "sleep"):
            host, port = vp.wait_for_ssh(12345, timeout=1000)
            assert host == "h"

    def test_retry_loop_raises_after_threshold(self):
        # Same "Retrying" message repeated STUCK_RETRY_POLL_THRESHOLD+ times
        # must raise StuckHostError before the overall timeout.
        stuck_msg = '{"actual_status": "loading", "status_msg": "94e7: Retrying in 2 seconds"}'
        responses = [stuck_msg] * (vp.STUCK_RETRY_POLL_THRESHOLD + 2)
        with patch.object(vp, "vastai", side_effect=responses), \
             patch.object(vp.time, "sleep"):
            with pytest.raises(vp.StuckHostError, match="retry-loop"):
                vp.wait_for_ssh(12345, timeout=10000)

    def test_retry_message_that_changes_does_not_trigger(self):
        # Different digit each time — not a real repeat, must not trigger
        # even though every message contains "Retrying".
        responses = [
            '{"actual_status": "loading", "status_msg": "layer1: Retrying in 2 seconds"}',
            '{"actual_status": "loading", "status_msg": "layer2: Retrying in 2 seconds"}',
            '{"actual_status": "loading", "status_msg": "layer3: Retrying in 2 seconds"}',
            '{"actual_status": "loading", "status_msg": "layer4: Retrying in 2 seconds"}',
            '{"actual_status": "loading", "status_msg": "layer5: Retrying in 2 seconds"}',
            '{"actual_status": "running", "status_msg": "success", "ssh_host": "h", "ssh_port": 22}',
        ]
        with patch.object(vp, "vastai", side_effect=responses), \
             patch.object(vp.time, "sleep"):
            host, port = vp.wait_for_ssh(12345, timeout=1000)
            assert host == "h"

    def test_non_retry_repeated_message_does_not_trigger(self):
        # Same message repeated but doesn't contain "retrying" — e.g. a big
        # layer slowly extracting. Should NOT be flagged stuck; it's normal
        # for "Extracting" without a percent to repeat a couple polls.
        responses = [
            '{"actual_status": "loading", "status_msg": "abc: Extracting"}',
            '{"actual_status": "loading", "status_msg": "abc: Extracting"}',
            '{"actual_status": "loading", "status_msg": "abc: Extracting"}',
            '{"actual_status": "loading", "status_msg": "abc: Extracting"}',
            '{"actual_status": "loading", "status_msg": "abc: Extracting"}',
            '{"actual_status": "loading", "status_msg": "abc: Extracting"}',
            '{"actual_status": "running", "status_msg": "success", "ssh_host": "h", "ssh_port": 22}',
        ]
        with patch.object(vp, "vastai", side_effect=responses), \
             patch.object(vp.time, "sleep"):
            host, port = vp.wait_for_ssh(12345, timeout=1000)
            assert host == "h"

    def test_stuck_counter_resets_on_progress(self):
        # Retrying x3 (below threshold), then progress, then retrying x3
        # again — must never trip, because the counter resets on the
        # intervening progress message.
        responses = (
            ['{"actual_status": "loading", "status_msg": "x: Retrying in 2 seconds"}'] * 3
            + ['{"actual_status": "loading", "status_msg": "y: Pull complete"}']
            + ['{"actual_status": "loading", "status_msg": "x: Retrying in 2 seconds"}'] * 3
            + ['{"actual_status": "running", "status_msg": "success", "ssh_host": "h", "ssh_port": 22}']
        )
        with patch.object(vp, "vastai", side_effect=responses), \
             patch.object(vp.time, "sleep"):
            host, port = vp.wait_for_ssh(12345, timeout=1000)
            assert host == "h"


class TestProvisionWithRetry:
    def test_first_host_good_no_retry_needed(self):
        with patch.object(vp, "find_cheapest_offer", return_value={"id": 1}), \
             patch.object(vp, "create_instance", return_value=111), \
             patch.object(vp, "wait_for_ssh", return_value=("goodhost", 22)), \
             patch.object(vp, "wait_ssh_responsive"), \
             patch.object(vp, "destroy_instance") as mock_destroy:
            instance_id, host, port = vp.provision_with_retry(
                "RTX_4090", 50, "img", "tok", max_retries=2)
            assert instance_id == 111
            assert host == "goodhost"
            mock_destroy.assert_not_called()

    def test_first_host_stuck_second_host_good(self):
        with patch.object(vp, "find_cheapest_offer", return_value={"id": 1}), \
             patch.object(vp, "create_instance", side_effect=[111, 222]), \
             patch.object(vp, "wait_for_ssh",
                          side_effect=[vp.StuckHostError("dead"), ("goodhost", 22)]), \
             patch.object(vp, "wait_ssh_responsive"), \
             patch.object(vp, "destroy_instance") as mock_destroy:
            instance_id, host, port = vp.provision_with_retry(
                "RTX_4090", 50, "img", "tok", max_retries=2)
            assert instance_id == 222
            assert host == "goodhost"
            # The bad first instance must have been destroyed.
            mock_destroy.assert_called_once_with(111)

    def test_all_hosts_bad_raises_after_exhausting_retries(self):
        with patch.object(vp, "find_cheapest_offer", return_value={"id": 1}), \
             patch.object(vp, "create_instance", side_effect=[111, 222, 333]), \
             patch.object(vp, "wait_for_ssh",
                          side_effect=[vp.StuckHostError("dead1"),
                                      vp.StuckHostError("dead2"),
                                      vp.StuckHostError("dead3")]), \
             patch.object(vp, "wait_ssh_responsive"), \
             patch.object(vp, "destroy_instance") as mock_destroy:
            with pytest.raises(RuntimeError, match="All 3 host attempts failed"):
                vp.provision_with_retry("RTX_4090", 50, "img", "tok", max_retries=2)
            # All three bad instances destroyed — no orphans left billing.
            assert mock_destroy.call_count == 3

    def test_ssh_unresponsive_also_triggers_retry(self):
        # wait_ssh_responsive can also fail (RuntimeError) even if
        # wait_for_ssh succeeded — e.g. sshd never comes up in the container.
        with patch.object(vp, "find_cheapest_offer", return_value={"id": 1}), \
             patch.object(vp, "create_instance", side_effect=[111, 222]), \
             patch.object(vp, "wait_for_ssh", return_value=("h", 22)), \
             patch.object(vp, "wait_ssh_responsive",
                          side_effect=[RuntimeError("never responsive"), None]), \
             patch.object(vp, "destroy_instance") as mock_destroy:
            instance_id, host, port = vp.provision_with_retry(
                "RTX_4090", 50, "img", "tok", max_retries=2)
            assert instance_id == 222
            mock_destroy.assert_called_once_with(111)

    def test_max_retries_zero_means_single_attempt(self):
        with patch.object(vp, "find_cheapest_offer", return_value={"id": 1}), \
             patch.object(vp, "create_instance", return_value=111), \
             patch.object(vp, "wait_for_ssh", side_effect=vp.StuckHostError("dead")), \
             patch.object(vp, "wait_ssh_responsive"), \
             patch.object(vp, "destroy_instance") as mock_destroy:
            with pytest.raises(RuntimeError, match="All 1 host attempts failed"):
                vp.provision_with_retry("RTX_4090", 50, "img", "tok", max_retries=0)
            mock_destroy.assert_called_once_with(111)
