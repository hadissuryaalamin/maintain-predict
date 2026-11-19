r"""Manual Docker exercise: run one named step at a time from the project root.

    .\.venv\Scripts\python.exe .\scripts\14_docker.py check
    .\.venv\Scripts\python.exe .\scripts\14_docker.py build
    .\.venv\Scripts\python.exe .\scripts\14_docker.py run
    .\.venv\Scripts\python.exe .\scripts\14_docker.py test
    .\.venv\Scripts\python.exe .\scripts\14_docker.py logs
    .\.venv\Scripts\python.exe .\scripts\14_docker.py stop

Only the Python standard library is needed to run this script.
No step trains a model, uploads an image, or creates cloud resources.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE = "maintain-predict:local"
CONTAINER = "maintain-predict-local"
OWNER_LABEL = "io.maintain-predict.exercise"
OWNER_VALUE = "local-docker"
BASE_URL = "http://127.0.0.1:8001"
EXAMPLE = {
    "type": "L", "air_temp_k": 300, "process_temp_k": 310,
    "rpm": 1500, "torque_nm": 40, "tool_wear_min": 100,
}


def docker(*arguments, capture=False):
    command = ["docker", *arguments]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=capture, text=True)
    if result.returncode:
        message = result.stderr.strip() if capture else "See Docker's output above."
        raise RuntimeError(f"Docker command failed: {message}")
    return result.stdout.strip() if capture else ""


def owned_container():
    """Never inspect/test/stop an unrelated container with a coinciding name."""
    info = json.loads(docker("container", "inspect", CONTAINER, capture=True))[0]
    labels = info.get("Config", {}).get("Labels") or {}
    if labels.get(OWNER_LABEL) != OWNER_VALUE:
        raise RuntimeError("This container lacks the exercise label; it will not be modified.")
    return info


def request_json(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(BASE_URL + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def check():
    """Step 1: check the CLI, Docker engine, and Linux container mode."""
    docker("--version")
    docker("version")
    operating_system = docker("info", "--format", "{{.OSType}}", capture=True)
    if operating_system != "linux":
        raise RuntimeError("Switch Docker Desktop to Linux containers, then repeat check.")
    print("Docker engine is ready for Linux containers.")


def build():
    """Step 2: turn Dockerfile and source code into a local Linux image."""
    docker("build", "--platform", "linux/amd64", "--progress", "plain", "--tag", IMAGE, ".")
    print("Built image:", IMAGE)


def run():
    """Step 3: start the image; startup fetches the pinned public model files."""
    # No project directories, credentials, or host Python environment are mounted.
    # Existing containers are never automatically replaced.
    docker(
        "run", "--detach", "--pull", "never", "--name", CONTAINER,
        "--label", f"{OWNER_LABEL}={OWNER_VALUE}",
        "--publish", "127.0.0.1:8001:8000", IMAGE,
    )
    print("Container started. Model download and startup may still be in progress.")
    print("Next: run the test step. Browser docs:", BASE_URL + "/docs")


def test(timeout=180):
    """Step 4: wait for readiness, predict one fabricated row, check validation."""
    info = owned_container()
    if not info["State"]["Running"]:
        raise RuntimeError("Container is not running. Run the logs step to inspect the failure.")
    bindings = info.get("NetworkSettings", {}).get("Ports", {}).get("8000/tcp") or []
    if not any(item.get("HostIp") == "127.0.0.1" and item.get("HostPort") == "8001" for item in bindings):
        raise RuntimeError("Container port mapping differs from this exercise; test stopped.")

    deadline = time.monotonic() + timeout
    print("Waiting for /health (up to", timeout, "seconds)...", flush=True)
    while True:
        try:
            if request_json("/health") == {"status": "ok"}:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("API did not become ready. Run logs; inspect model download/startup errors.")
        time.sleep(1)

    print("Sending fabricated inputs:", json.dumps(EXAMPLE))
    result = request_json("/predict", EXAMPLE)
    if not (
        math.isclose(result["failure_score"], 0.0005919649847783148, rel_tol=1e-5, abs_tol=1e-8)
        and result["predicted_label"] == 0 and result["threshold"] == 0.5
        and result["target_policy"] == "failure_evidence_or_v1"
    ):
        raise RuntimeError(f"Container prediction differs from the expected selected model: {result}")
    print(json.dumps(result, indent=2))
    # Invalid input should be rejected before model inference.
    try:
        request_json("/predict", {**EXAMPLE, "type": "unknown"})
    except HTTPError as error:
        if error.code != 422:
            raise RuntimeError(f"Expected HTTP 422, received {error.code}.") from error
    else:
        raise RuntimeError("Invalid product type was unexpectedly accepted.")

    output = PROJECT_ROOT / "artifacts/docker"
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "container_id": info["Id"], "image_id": info["Image"], "base_url": BASE_URL,
        "health": "ok", "fabricated_input": EXAMPLE, "prediction": result,
        "invalid_type_status": 422, "model_refitted": False, "dataset_evaluated": False,
    }
    (output / "test_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Passed: readiness, example prediction, and HTTP 422 validation.")
    print("Receipt:", output / "test_receipt.json")


def logs():
    """Optional: inspect the last 100 lines without starting another container."""
    owned_container()
    docker("logs", "--tail", "100", CONTAINER)


def stop():
    """Final step: stop and remove only this exercise container; keep its image."""
    info = owned_container()
    if info["State"]["Running"]:
        docker("stop", CONTAINER)
    docker("rm", CONTAINER)
    print("Exercise container removed. The image remains available for another run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["check", "build", "run", "test", "logs", "stop"])
    arguments = parser.parse_args()
    if shutil.which("docker") is None:
        parser.exit(1, "Docker CLI not found. Install Docker Desktop, open it, then repeat check.\n")
    try:
        {"check": check, "build": build, "run": run, "test": test, "logs": logs, "stop": stop}[arguments.step]()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
