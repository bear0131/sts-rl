"""Bounded Beijing-only Pro acquisition using AutoDL's documented developer API.

No dependencies beyond Python 3.11+. Default mode is an offline plan.
See README.md#autodl-pro for pricing, retries and recovery limitations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import getpass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid


API_ROOT = "https://api.autodl.com"
PREFIX = "/api/v1/dev/instance/pro"
GPU_SPECS = {"h800", "v-48g", "pro6000-p", "v-32g-p", "v-48g-350w", "5090-p", "4090D"}
# The provider does not publish error codes. Only an explicit, empty-result
# business rejection with one of these exact messages is treated as no stock.
# Unknown wording, HTTP errors and timeouts must never trigger a second POST.
NO_STOCK_MESSAGES = {"暂无库存", "库存不足", "暂无可用库存"}


class AcquireError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    job: str
    image_uuid: str
    instances: int = 1
    gpu_spec_uuid: str = "5090-p"
    gpus_per_instance: int = 1
    cuda_v_from: int = 132
    expand_system_disk_by_gb: int = 0
    interval_seconds: int = 60
    max_wait_seconds: int = 3600
    max_create_attempts: int = 60

    def validate(self, *, execute=False):
        if not isinstance(self.job, str) or not re.fullmatch(r"[a-z0-9-]{1,24}", self.job):
            raise AcquireError("job 必须为 1–24 位小写字母、数字或连字符。")
        if self.gpu_spec_uuid not in GPU_SPECS:
            raise AcquireError("GPU 规格不在已核实的官方列表中。")
        if not isinstance(self.image_uuid, str):
            raise AcquireError("image_uuid 必须是字符串。")
        if execute and not re.fullmatch(r"(?:base-image|image)-[a-z0-9]+", self.image_uuid):
            raise AcquireError("请填写控制台/镜像列表中的真实 image_uuid。")
        bounds = {"instances": (1, 16), "gpus_per_instance": (1, 4),
                  "cuda_v_from": (100, 999), "expand_system_disk_by_gb": (0, 500),
                  "interval_seconds": (60, 3600), "max_wait_seconds": (60, 86400),
                  "max_create_attempts": (1, 1000)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise AcquireError(f"{key} 必须是 {low}–{high} 的整数。")

    def payload(self, name):
        return {"data_center_list": ["beijingDC2"],
                "req_gpu_amount": self.gpus_per_instance,
                "expand_system_disk_by_gb": self.expand_system_disk_by_gb,
                "gpu_spec_uuid": self.gpu_spec_uuid, "image_uuid": self.image_uuid,
                "cuda_v_from": self.cuda_v_from, "instance_name": name}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the developer token to a redirect target.
        return None


class Client:
    def __init__(self, token):
        if not token or any(c in token for c in "\r\n"):
            raise AcquireError("开发者 Token 为空或格式错误。")
        self.token = token
        self.opener = build_opener(NoRedirect())
        self.deadline = None
        self.clock = time.time

    def request(self, path, body, timeout=20):
        if self.deadline is not None:
            remaining = self.deadline - self.clock()
            if remaining <= 0:
                raise AcquireError("等待时间已到；停止，已有实例仍可能计费。")
            timeout = min(timeout, remaining)
        req = Request(API_ROOT + PREFIX + path,
                      data=json.dumps(body).encode(), method="POST",
                      headers={"Authorization": self.token, "Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise AcquireError("API 响应过大；停止，避免误判创建结果。")
            result = json.loads(raw)
        except HTTPError as exc:
            # Do not expose response bodies, URLs, headers or echoed credentials.
            raise AcquireError(f"API HTTP {exc.code}；未自动重试。") from None
        except (URLError, TimeoutError, OSError):
            raise AcquireError("API 网络请求失败/超时；未自动重试。") from None
        except (ValueError, UnicodeError):
            raise AcquireError("API 返回无法解析的数据；未自动重试。") from None
        if not isinstance(result, dict) or not isinstance(result.get("code"), str):
            raise AcquireError("API 响应格式发生变化；停止。")
        return result

    def pages(self, path):
        rows = []
        for page in range(1, 101):
            reply = self.request(path, {"page_index": page, "page_size": 10})
            if reply["code"] != "Success":
                raise AcquireError("读取列表被拒绝；检查实名认证、Token、权限及余额。")
            data = reply.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("list"), list):
                raise AcquireError("列表结构不符合官方文档；停止。")
            max_page = data.get("max_page")
            if type(max_page) is not int or not 0 <= max_page <= 100:
                raise AcquireError("列表页数无效或超过本工具上限；停止。")
            if any(not isinstance(row, dict) for row in data["list"]):
                raise AcquireError("列表记录格式错误；停止。")
            rows.extend(data["list"])
            if page >= max_page:
                return rows
        raise AcquireError("列表未能完整读取；停止。")

    def instances(self):
        return self.pages("/list")


def save_state(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path):
    """OS lock is released on crash; no unsafe stale-PID lock deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise AcquireError("本状态目录已有获取脚本在运行。") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_state(path, config):
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("version") != 1 or state.get("config") != asdict(config):
            raise AcquireError("已有任务配置不同；不要改动旧任务或删除台账后重试。")
        slots = state.get("slots")
        if (not isinstance(slots, list) or len(slots) != config.instances
                or any(not isinstance(s, dict) or s.get("phase") not in
                       {"new", "pending", "created", "rejected"} for s in slots)):
            raise AcquireError("台账损坏；停止，请保留文件并核对控制台。")
        if (type(state.get("attempts")) is not int or state["attempts"] < 0
                or type(state.get("next_attempt_at")) not in (int, float)
                or not math.isfinite(state["next_attempt_at"])
                or len({s.get("name") for s in slots}) != len(slots)):
            raise AcquireError("台账计数/名称损坏；停止。")
        for slot in slots:
            if (not isinstance(slot.get("name"), str)
                    or not re.fullmatch(r"bj-" + re.escape(config.job) + r"-[a-f0-9]{12}-\d{2}", slot["name"])
                    or (slot.get("uuid") is not None and not re.fullmatch(r"pro-[a-z0-9]+", str(slot["uuid"])))
                    or (slot["phase"] == "created" and not slot.get("uuid"))):
                raise AcquireError("台账实例身份损坏；停止。")
        return state
    nonce = uuid.uuid4().hex[:12]
    state = {"version": 1, "config": asdict(config), "attempts": 0,
             "next_attempt_at": 0, "slots": [
                 {"name": f"bj-{config.job}-{nonce}-{i+1:02d}", "phase": "new", "uuid": None}
                 for i in range(config.instances)]}
    save_state(path, state)
    return state


def validate_instance(row, config):
    region = str(row.get("region_sign", ""))
    if not (region.startswith("bj-") or region == "beijingDC2"
            or str(row.get("region_name", "")).startswith("北京")):
        raise AcquireError("已创建实例的地区无法确认是北京；停止，请检查控制台（可能已计费）。")
    if (row.get("gpu_spec_uuid") != config.gpu_spec_uuid
            or row.get("req_gpu_amount") != config.gpus_per_instance
            or row.get("charge_type") != "payg"):
        raise AcquireError("已创建实例配置不一致；停止，请检查控制台（可能已计费）。")
    if not isinstance(row.get("uuid"), str) or not re.fullmatch(r"pro-[a-z0-9]+", row["uuid"]):
        raise AcquireError("实例 UUID 格式异常；停止。")


def reconcile(client, config, state, path):
    rows = client.instances()
    for slot in state["slots"]:
        matches = [r for r in rows if r.get("name") == slot["name"]]
        if len(matches) > 1:
            raise AcquireError("发现同名重复实例；停止，请检查控制台。")
        if matches:
            row = matches[0]
            validate_instance(row, config)
            if slot["uuid"] and row["uuid"] != slot["uuid"]:
                raise AcquireError("实例身份与台账冲突；停止。")
            slot.update(phase="created", uuid=row["uuid"], status=row.get("status"))
            save_state(path, state)
        elif slot["phase"] in {"pending", "created"}:
            raise AcquireError(
                "之前的创建结果尚无法核实；即使列表暂无记录也不重复下单。"
                "保留台账，稍后原命令重跑以核对，或联系 AutoDL 确认。")
        if slot["phase"] == "rejected":
            raise AcquireError("此前创建被拒绝且非明确缺货；停止，请核对控制台/API 权限。")


def acquire(client, config, path, *, clock=time.time, sleep=time.sleep, emit=print):
    """Caller holds directory lock. No automatic replacement/deletion of instances."""
    config.validate(execute=True)
    state = read_state(path, config)
    deadline = clock() + config.max_wait_seconds
    client.deadline, client.clock = deadline, clock
    reconcile(client, config, state, path)
    while any(s["phase"] != "created" for s in state["slots"]):
        remaining = deadline - clock()
        if remaining <= 0 or state["attempts"] >= config.max_create_attempts:
            emit("等待/累计尝试达到上限；已获取实例仍保留并可能继续计费。")
            return 2
        delay = max(0, state["next_attempt_at"] - clock())
        if delay:
            sleep(min(delay, remaining))
            continue
        # Re-check before every mutation, including previously rejected no-stock.
        reconcile(client, config, state, path)
        slot = next((s for s in state["slots"] if s["phase"] == "new"), None)
        if slot is None:
            break
        if clock() >= deadline:
            continue
        slot["phase"] = "pending"
        state["attempts"] += 1
        state["next_attempt_at"] = clock() + config.interval_seconds
        # Persist intent BEFORE the paid request, including Ctrl-C/crash cases.
        save_state(path, state)
        try:
            reply = client.request("/create", config.payload(slot["name"]),
                                   timeout=max(1, min(20, deadline - clock())))
        except AcquireError:
            emit("创建请求结果不明，已保留待核对状态；不会重发创建请求。")
            raise
        if reply.get("code") == "Success":
            instance = reply.get("data")
            if not isinstance(instance, str) or not re.fullmatch(r"pro-[a-z0-9]+", instance):
                raise AcquireError("创建返回成功但实例 ID 不明；保留待核对状态，不重试。")
            slot.update(phase="created", uuid=instance)
            save_state(path, state)
            emit(f"已创建 {instance}（尚待核验地区/运行状态），开始计费以平台为准。")
            # Allow eventual list consistency through a later manual rerun, never
            # by issuing another create while this identity is unverified.
            reconcile(client, config, state, path)
        elif (reply.get("code") and reply.get("data") is None
              and reply.get("msg") in NO_STOCK_MESSAGES):
            slot["phase"] = "new"
            save_state(path, state)
            emit(f"北京暂无所选库存；至少 {config.interval_seconds} 秒后再试。")
        else:
            # Failure with nonempty data might still refer to a created resource.
            slot["phase"] = "rejected" if reply.get("data") is None else "pending"
            save_state(path, state)
            raise AcquireError("创建返回未识别的业务错误；保留状态并停止，不盲目重试。")
    emit(json.dumps({"created": [{k: s.get(k) for k in ("name", "uuid", "status")}
                                 for s in state["slots"]]}, ensure_ascii=False))
    emit("指定数量已创建。请在控制台核对 running/SSH；本工具不部署训练，也不自动关机。")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="本地 JSON 配置；默认仅预演")
    parser.add_argument("--execute", action="store_true", help="实际创建付费 Pro 实例")
    parser.add_argument("--accept-current-price", action="store_true",
                        help="接受平台当前按量价；官方 API 无下单前限价字段")
    parser.add_argument("--list-images", action="store_true", help="只读查询私有镜像 UUID")
    parser.add_argument("--state-dir", type=Path, default=Path("runs/local/autodl-acquire"))
    args = parser.parse_args(argv)
    try:
        if args.list_images and (args.execute or args.config):
            raise AcquireError("--list-images 请单独使用。")
        if args.list_images:
            token = os.environ.get("AUTODL_TOKEN") or getpass.getpass("AutoDL 开发者 Token（隐藏输入）: ")
            rows = Client(token).pages("/image/private/list")
            print(json.dumps([{k: r.get(k) for k in ("image_uuid", "name", "status")}
                              for r in rows], ensure_ascii=False, indent=2))
            return 0
        if not args.config:
            raise AcquireError("请使用 --config 指定配置，或用 --list-images 查询镜像。")
        config = Config(**json.loads(args.config.read_text(encoding="utf-8-sig")))
        config.validate(execute=args.execute)
        print(json.dumps({"mode": "execute" if args.execute else "offline-plan",
                          "instances": config.instances,
                          "create_payload": config.payload(f"bj-{config.job}-<task>-01"),
                          "max_wait_seconds": config.max_wait_seconds,
                          "max_create_attempts": config.max_create_attempts,
                          "interval_seconds": config.interval_seconds,
                          "price_cap_supported": False}, ensure_ascii=False, indent=2))
        if not args.execute:
            print("仅预演，未联网/未租用。镜像与价格须先在控制台确认。")
            return 0
        if not args.accept_current_price:
            raise AcquireError("创建会按平台当前价格计费，无自动限价；确认后添加 --accept-current-price。")
        token = os.environ.get("AUTODL_TOKEN") or getpass.getpass("AutoDL 开发者 Token（隐藏输入）: ")
        with exclusive_lock(args.state_dir / ".lock"):
            return acquire(Client(token), config, args.state_dir / f"{config.job}.json")
    except KeyboardInterrupt:
        print("已停止等待；已有实例不会自动关机，未决创建保留在台账。", file=sys.stderr)
        return 130
    except (AcquireError, OSError, ValueError, TypeError):
        # Configuration/I/O errors may contain file contents or private paths.
        exc = sys.exc_info()[1]
        message = str(exc) if isinstance(exc, AcquireError) else "配置/台账读写失败；请检查文件与字段类型。"
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
