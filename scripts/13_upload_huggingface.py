r"""Inspect the prepared package; publish publicly only when run with --upload.

    .\.venv\Scripts\python.exe .\scripts\13_upload_huggingface.py
    .\.venv\Scripts\python.exe .\scripts\13_upload_huggingface.py --upload

Login interactively first with .\.venv\Scripts\hf.exe auth login.
Do not put tokens in this script or in the model directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "hadissuryaalamin/maintain-predict"
PACKAGE_FILES = {
    "README.md", "LICENSE", "model.ubj", "preprocessor.skops", "predict.py",
    "model_contract.json", "requirements.txt", "export_manifest.json",
}


def main(project_root=PROJECT_ROOT, *, upload=False, api=None):
    root = Path(project_root)
    folder = root / "artifacts/huggingface/model"
    # 1. Check the prepared file allowlist and hashes; no training or prediction.
    manifest = json.loads((folder / "export_manifest.json").read_text(encoding="utf-8"))
    if manifest["destination_repo_id"] != REPO_ID or manifest["license"] != "mit":
        raise ValueError("Unexpected repository or model licence in the prepared export.")
    if set(manifest["sha256"]) != PACKAGE_FILES - {"export_manifest.json"}:
        raise ValueError("Export manifest does not contain the expected release files.")
    fingerprints = {}
    for name in sorted(PACKAGE_FILES):
        fingerprints[name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        if name != "export_manifest.json" and fingerprints[name] != manifest["sha256"][name]:
            raise ValueError(f"Prepared file changed after export: {name}")
    print("Destination:", REPO_ID, "(PUBLIC model repository)")
    print("Files to upload:", ", ".join(sorted(PACKAGE_FILES)))
    if not upload:
        print("Local check passed. No network calls or upload. Run with --upload to publish.")
        return {"uploaded": False, "sha256": fingerprints}

    # 2. Use the login saved by hf auth login; never print or save its token.
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    api = api if api is not None else HfApi()
    if api.whoami()["name"] != REPO_ID.split("/")[0]:
        raise ValueError("Logged-in account differs from the planned personal repository owner.")
    try:
        existing = api.repo_info(repo_id=REPO_ID, repo_type="model")
    except RepositoryNotFoundError:
        # Creating a public repository happens only after explicit --upload.
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=False)
        existing = api.repo_info(repo_id=REPO_ID, repo_type="model")
    if existing.private:
        raise ValueError("The existing repository is private. Its visibility was not changed.")

    # 3. Publish only the eight selected files, retaining any unrelated remote files.
    commit = api.upload_folder(
        repo_id=REPO_ID, repo_type="model", folder_path=folder,
        allow_patterns=sorted(PACKAGE_FILES),
        commit_message="Publish selected maintain-predict v1 model package (MIT)",
        parent_commit=existing.sha,
    )
    remote_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="model", revision=commit.oid))
    if not PACKAGE_FILES <= remote_files:
        raise RuntimeError("Upload returned a commit, but expected files are missing; inspect the Hub repository.")

    # 4. Save a separate receipt; the export manifest remains a preparation snapshot.
    receipt = {
        "repo_id": REPO_ID, "url": f"https://huggingface.co/{REPO_ID}",
        "commit_sha": commit.oid, "commit_url": str(commit.commit_url),
        "uploaded": True, "private": False, "license": "mit",
        "sha256": fingerprints, "model_refitted": False, "dataset_evaluated": False,
        "verification": "Expected files listed at the returned commit; download/inference not yet checked.",
    }
    path = root / "artifacts/huggingface/upload_receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Uploaded:", receipt["url"])
    print("Commit:", receipt["commit_sha"])
    print("Receipt:", path)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="Publish the verified package to the PUBLIC Hub repository.")
    main(upload=parser.parse_args().upload)
