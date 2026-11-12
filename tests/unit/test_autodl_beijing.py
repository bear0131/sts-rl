"""No real tokens, network, GPU or paid requests. Run with stdlib unittest."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from experiments.launchers.autodl_beijing import (
    AcquireError, Client, Config, NoRedirect, acquire, exclusive_lock,
    main, read_state, save_state,
)


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeAPI:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.rows = []
        self.calls = []

    def instances(self):
        return self.rows.copy()

    def add(self, name, uuid="pro-test1", region="北京B区", **changes):
        row = dict(name=name, uuid=uuid, region_name=region, status="running",
                   charge_type="payg", req_gpu_amount=1, gpu_spec_uuid="5090-p")
        row.update(changes)
        self.rows.append(row)

    def request(self, path, body, timeout=20):
        assert path == "/create"
        self.calls.append(body)
        result = next(self.replies)
        if isinstance(result, BaseException):
            raise result
        if result.get("code") == "Success" and result.get("data"):
            self.add(body["instance_name"], result["data"])
        return result


def ok(instance):
    return {"code": "Success", "data": instance}


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "job.json"
        self.config = Config(job="test", image_uuid="image-fixture", instances=2)
        self.clock = Clock()

    def run_job(self, api, config=None):
        return acquire(api, config or self.config, self.path,
                       clock=self.clock, sleep=self.clock.sleep, emit=lambda _: None)

    def test_only_beijing_and_exact_count_then_restart_does_not_buy_more(self):
        api = FakeAPI([ok("pro-a"), ok("pro-b")])
        self.assertEqual(self.run_job(api), 0)
        self.assertEqual(self.run_job(api), 0)
        self.assertEqual(len(api.calls), 2)
        self.assertTrue(all(c["data_center_list"] == ["beijingDC2"] for c in api.calls))
        self.assertEqual(self.clock.now, 1060)
        self.assertNotEqual(api.calls[0]["instance_name"], api.calls[1]["instance_name"])

    def test_explicit_no_stock_waits_then_succeeds_same_slot(self):
        api = FakeAPI([{"code": "ProviderError", "data": None, "msg": "暂无库存"}, ok("pro-a")])
        self.assertEqual(self.run_job(api, replace(self.config, instances=1)), 0)
        self.assertEqual(self.clock.now, 1060)
        self.assertEqual(api.calls[0], api.calls[1])

    def test_deadline_keeps_partial_success_and_resume_fills_only_missing(self):
        cfg = replace(self.config, max_wait_seconds=60)
        api = FakeAPI([ok("pro-a"), ok("pro-b")])
        self.assertEqual(self.run_job(api, cfg), 2)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(self.run_job(api, cfg), 0)
        self.assertEqual(len(api.calls), 2)

    def test_attempt_cap_persists_across_restarts(self):
        cfg = replace(self.config, max_create_attempts=1)
        api = FakeAPI([{"code": "Empty", "data": None, "msg": "库存不足"}])
        self.assertEqual(self.run_job(api, cfg), 2)
        self.assertEqual(self.run_job(api, cfg), 2)
        self.assertEqual(len(api.calls), 1)

    def test_timeout_without_list_record_never_reposts(self):
        api = FakeAPI([AcquireError("network timeout")])
        with self.assertRaises(AcquireError):
            self.run_job(api)
        with self.assertRaisesRegex(AcquireError, "不重复下单"):
            self.run_job(api)
        self.assertEqual(len(api.calls), 1)
        self.assertEqual(json.loads(self.path.read_text())["slots"][0]["phase"], "pending")

    def test_timeout_after_server_created_is_reconciled_without_duplicate(self):
        cfg = replace(self.config, instances=1)
        api = FakeAPI([AcquireError("network timeout")])
        with self.assertRaises(AcquireError):
            self.run_job(api, cfg)
        api.add(api.calls[0]["instance_name"])
        self.assertEqual(self.run_job(api, cfg), 0)
        self.assertEqual(len(api.calls), 1)

    def test_interrupt_between_intent_and_response_is_persistent(self):
        api = FakeAPI([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(api)
        with self.assertRaises(AcquireError):
            self.run_job(api)
        self.assertEqual(len(api.calls), 1)

    def test_unknown_failure_is_not_treated_as_stock_shortage(self):
        for msg in ["余额不足", "unauthorized", "后台处理中", "库存不足，请重试未知流程"]:
            with self.subTest(msg=msg):
                self.path.unlink(missing_ok=True)
                api = FakeAPI([{"code": "Error", "data": None, "msg": msg}])
                with self.assertRaises(AcquireError):
                    self.run_job(api)
                with self.assertRaises(AcquireError):
                    self.run_job(api)
                self.assertEqual(len(api.calls), 1)

    def test_failure_with_resource_data_remains_ambiguous(self):
        api = FakeAPI([{"code": "Error", "data": "pro-x", "msg": "库存不足"}])
        with self.assertRaises(AcquireError):
            self.run_job(api)
        self.assertEqual(json.loads(self.path.read_text())["slots"][0]["phase"], "pending")

    def test_success_without_uuid_does_not_repost(self):
        api = FakeAPI([ok(None)])
        with self.assertRaises(AcquireError):
            self.run_job(api)
        with self.assertRaises(AcquireError):
            self.run_job(api)
        self.assertEqual(len(api.calls), 1)

    def test_wrong_region_or_gpu_stops_before_another_purchase(self):
        for changes in [{"region_name": "西北区"}, {"gpu_spec_uuid": "h800"}]:
            with self.subTest(changes=changes):
                self.path.unlink(missing_ok=True)
                state = read_state(self.path, self.config)
                api = FakeAPI()
                api.add(state["slots"][0]["name"], **changes)
                with self.assertRaises(AcquireError):
                    self.run_job(api)
                self.assertEqual(api.calls, [])

    def test_duplicate_name_or_changed_config_stops(self):
        state = read_state(self.path, self.config)
        api = FakeAPI()
        for instance in ["pro-a", "pro-b"]:
            api.add(state["slots"][0]["name"], instance)
        with self.assertRaises(AcquireError):
            self.run_job(api)
        with self.assertRaises(AcquireError):
            self.run_job(api, replace(self.config, instances=3))
        self.assertFalse(api.calls)

    def test_corrupt_state_is_not_reset_and_no_request_sent(self):
        self.path.write_text("[]")
        api = FakeAPI()
        with self.assertRaises(AcquireError):
            self.run_job(api)
        self.assertEqual(self.path.read_text(), "[]")
        self.assertFalse(api.calls)

    def test_missing_created_instance_is_not_replaced(self):
        cfg = replace(self.config, instances=1)
        api = FakeAPI([ok("pro-a")])
        self.run_job(api, cfg)
        api.rows = []
        with self.assertRaises(AcquireError):
            self.run_job(api, cfg)
        self.assertEqual(len(api.calls), 1)

    def test_lock_rejects_concurrent_worker_and_releases(self):
        path = self.path.with_suffix(".lock")
        with exclusive_lock(path):
            with self.assertRaises(AcquireError):
                with exclusive_lock(path):
                    self.fail("second worker entered")
        with exclusive_lock(path):
            pass

    def test_default_plan_has_no_network_or_token_prompt(self):
        config = self.path.with_suffix(".config.json")
        config.write_text(json.dumps({"job": "demo", "image_uuid": ""}))
        with patch("experiments.launchers.autodl_beijing.Client") as client, \
                patch("getpass.getpass") as prompt, patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(main(["--config", str(config)]), 0)
            client.assert_not_called()
            prompt.assert_not_called()

    def test_execute_requires_price_acknowledgement_before_token_or_network(self):
        config = self.path.with_suffix(".config.json")
        config.write_text(json.dumps({"job": "demo", "image_uuid": "image-fixture"}))
        with patch("experiments.launchers.autodl_beijing.Client") as client, \
                patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(main(["--config", str(config), "--execute"]), 1)
            client.assert_not_called()


class HTTPTests(unittest.TestCase):
    def test_real_http_adapter_uses_documented_host_header_and_payload(self):
        client = Client("secret-fixture")
        config = Config(job="test", image_uuid="image-fixture")
        with patch.object(client.opener, "open", return_value=io.BytesIO(
                json.dumps(ok("pro-123")).encode())) as request:
            self.assertEqual(client.request("/create", config.payload("test")), ok("pro-123"))
            sent = request.call_args.args[0]
            self.assertEqual(sent.full_url, "https://api.autodl.com/api/v1/dev/instance/pro/create")
            self.assertEqual(sent.get_header("Authorization"), "secret-fixture")
            self.assertEqual(sent.method, "POST")
            self.assertEqual(json.loads(sent.data)["data_center_list"], ["beijingDC2"])

    def test_malformed_response_fails_without_dumping_body(self):
        client = Client("secret-fixture")
        with patch.object(client.opener, "open", return_value=io.BytesIO(b"secret-fixture")):
            with self.assertRaises(AcquireError) as caught:
                client.request("/create", {})
            self.assertNotIn("secret-fixture", str(caught.exception))

    def test_http_error_does_not_leak_body_token_or_retry(self):
        client = Client("secret-fixture")
        client.opener.open = unittest.mock.Mock(side_effect=HTTPError(
            "https://example.invalid/secret-fixture", 429, "secret-fixture", {},
            io.BytesIO(b"secret-fixture")))
        with self.assertRaises(AcquireError) as caught:
            client.request("/create", {})
        self.assertNotIn("secret-fixture", str(caught.exception))
        self.assertIn("429", str(caught.exception))
        self.assertEqual(client.opener.open.call_count, 1)

    def test_redirect_is_not_followed(self):
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid"))

    def test_pagination_collects_all_before_reconciliation(self):
        client = Client("fixture")
        with patch.object(client, "request", side_effect=[
            {"code": "Success", "data": {"list": [{"uuid": "pro-a"}], "max_page": 2}},
            {"code": "Success", "data": {"list": [{"uuid": "pro-b"}], "max_page": 2}},
        ]) as request:
            self.assertEqual(len(client.instances()), 2)
            self.assertEqual([c.args[1]["page_index"] for c in request.call_args_list], [1, 2])

    def test_expired_budget_prevents_http(self):
        client = Client("fixture")
        client.clock = lambda: 20
        client.deadline = 19
        with patch.object(client.opener, "open") as request:
            with self.assertRaises(AcquireError):
                client.request("/create", {})
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
