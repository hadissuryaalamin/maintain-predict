import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest


def exercise_module():
    path = Path(__file__).resolve().parents[1] / "scripts/14_docker.py"
    spec = importlib.util.spec_from_file_location("docker_exercise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def container_info(module, owned=True):
    return {
        "Id": "synthetic-container", "Image": "synthetic-image", "State": {"Running": True},
        "Config": {"Labels": {module.OWNER_LABEL: module.OWNER_VALUE} if owned else {}},
        "NetworkSettings": {"Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8001"}]}},
    }


def test_build_and_run_are_separate_local_steps_without_mounts_or_remote_image_pull(monkeypatch):
    module = exercise_module()
    calls = []
    monkeypatch.setattr(module, "docker", lambda *args, **kwargs: calls.append(args))
    module.build()
    assert len(calls) == 1 and calls[0][0] == "build"
    calls.clear()
    module.run()
    assert len(calls) == 1 and calls[0][0] == "run"
    command = calls[0]
    assert command[command.index("--publish") + 1] == "127.0.0.1:8001:8000"
    assert command[command.index("--pull") + 1] == "never"
    assert "--volume" not in command and "--mount" not in command and "--privileged" not in command


def test_stop_refuses_unrelated_container_and_removes_only_owned_container(monkeypatch):
    module = exercise_module()
    calls = []
    info = container_info(module, owned=False)

    def docker(*args, **kwargs):
        calls.append(args)
        return json.dumps([info]) if args[0] == "container" else ""

    monkeypatch.setattr(module, "docker", docker)
    with pytest.raises(RuntimeError, match="exercise label"):
        module.stop()
    assert len(calls) == 1
    calls.clear()
    info["Config"]["Labels"][module.OWNER_LABEL] = module.OWNER_VALUE
    module.stop()
    assert calls[-2:] == [("stop", module.CONTAINER), ("rm", module.CONTAINER)]


def test_smoke_check_waits_for_readiness_and_records_expected_result(tmp_path, monkeypatch):
    module = exercise_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "owned_container", lambda: container_info(module))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    calls = []
    result = {
        "failure_score": 0.0005919649847783148, "predicted_label": 0,
        "threshold": 0.5, "target_policy": "failure_evidence_or_v1",
    }

    def request(path, payload=None):
        calls.append((path, payload))
        if len(calls) == 1:
            raise URLError("Application is starting")
        if path == "/health":
            return {"status": "ok"}
        if payload["type"] == "unknown":
            raise HTTPError("http://localhost", 422, "Invalid type", {}, None)
        return result

    monkeypatch.setattr(module, "request_json", request)
    module.test()
    receipt = json.loads((tmp_path / "artifacts/docker/test_receipt.json").read_text())
    assert receipt["prediction"] == result
    assert receipt["container_id"] == "synthetic-container"
    assert receipt["invalid_type_status"] == 422
    assert receipt["model_refitted"] is receipt["dataset_evaluated"] is False
    assert [path for path, _ in calls] == ["/health", "/health", "/predict", "/predict"]


def test_wrong_prediction_does_not_create_success_receipt(tmp_path, monkeypatch):
    module = exercise_module()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "owned_container", lambda: container_info(module))
    monkeypatch.setattr(module, "request_json", lambda path, payload=None:
                        {"status": "ok"} if path == "/health" else {
                            "failure_score": 0.9, "predicted_label": 1, "threshold": 0.5,
                            "target_policy": "failure_evidence_or_v1",
                        })
    with pytest.raises(RuntimeError, match="prediction differs"):
        module.test()
    assert not (tmp_path / "artifacts/docker/test_receipt.json").exists()


def test_wrong_port_and_readiness_timeout_prevent_prediction(monkeypatch):
    module = exercise_module()
    info = container_info(module)
    info["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"] = "8000"
    monkeypatch.setattr(module, "owned_container", lambda: info)
    with pytest.raises(RuntimeError, match="port mapping"):
        module.test()
    info["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"] = "8001"
    ticks = iter([0, 1000])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))

    def unavailable(path, payload=None):
        assert path == "/health"
        raise URLError("Not ready")

    monkeypatch.setattr(module, "request_json", unavailable)
    with pytest.raises(RuntimeError, match="did not become ready"):
        module.test()
