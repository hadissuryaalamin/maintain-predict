import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_explain import saved_selected
from test_final_test import final_case
from test_registry import registry_case
from test_huggingface_export import export_case


def upload_module():
    path = Path(__file__).resolve().parents[1] / "scripts/13_upload_huggingface.py"
    spec = importlib.util.spec_from_file_location("hub_upload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeApi:
    def __init__(self, module, *, private=False, account="hadissuryaalamin"):
        self.module, self.private, self.account = module, private, account
        self.writes = []

    def whoami(self):
        return {"name": self.account}

    def repo_info(self, **kwargs):
        return SimpleNamespace(private=self.private, sha="previous-commit")

    def upload_folder(self, **kwargs):
        self.writes.append(kwargs)
        return SimpleNamespace(oid="uploaded-commit", commit_url="https://huggingface.co/example/commit")

    def list_repo_files(self, **kwargs):
        assert kwargs["revision"] == "uploaded-commit"
        return sorted(self.module.PACKAGE_FILES)


def test_default_is_offline_and_explicit_upload_uses_file_allowlist(export_case):
    root, preparation, _ = export_case
    preparation.main(root)
    module = upload_module()

    class NoNetwork:
        def __getattr__(self, name):
            raise AssertionError("A local check must never use the API.")

    assert module.main(root, api=NoNetwork())["uploaded"] is False
    assert not (root / "artifacts/huggingface/upload_receipt.json").exists()
    folder = root / "artifacts/huggingface/model"
    (folder / ".env").write_text("synthetic file that must not be uploaded")
    before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    api = FakeApi(module)
    receipt = module.main(root, upload=True, api=api)
    assert len(api.writes) == 1
    call = api.writes[0]
    assert set(call["allow_patterns"]) == module.PACKAGE_FILES
    assert ".env" not in call["allow_patterns"]
    assert call["parent_commit"] == "previous-commit" and "delete_patterns" not in call
    assert receipt["uploaded"] is True and receipt["private"] is False
    assert receipt["commit_sha"] == "uploaded-commit"
    assert json.loads((root / "artifacts/huggingface/upload_receipt.json").read_text()) == receipt
    assert before == {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    (folder / "predict.py").write_text("changed code")
    with pytest.raises(ValueError, match="changed after export"):
        module.main(root, upload=True, api=NoNetwork())


@pytest.mark.parametrize("private,account", [(True, "hadissuryaalamin"), (False, "someone-else")])
def test_upload_refuses_unexpected_visibility_or_account_before_writing(export_case, private, account):
    root, preparation, _ = export_case
    preparation.main(root)
    module = upload_module()
    api = FakeApi(module, private=private, account=account)
    with pytest.raises(ValueError):
        module.main(root, upload=True, api=api)
    assert api.writes == []
