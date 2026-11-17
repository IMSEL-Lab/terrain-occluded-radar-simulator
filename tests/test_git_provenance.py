import shutil
import subprocess

import pytest

from scripts.git_provenance import checkout_revision


@pytest.fixture
def checkout(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required to create a checkout")
    subprocess.run([git, "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            git,
            "-c",
            "user.name=j-vaught",
            "-c",
            "user.email=jvaught@sc.edu",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--allow-empty",
            "--quiet",
            "-m",
            "Fixture commit",
        ],
        cwd=tmp_path,
        check=True,
    )
    revision = subprocess.check_output(
        [git, "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    return tmp_path, revision


def test_revision_identifies_the_source_checkout(checkout):
    root, revision = checkout
    assert checkout_revision(root) == revision


def test_installed_source_excludes_parent_checkout(checkout, capfd):
    root, _ = checkout
    installed = root / ".venv" / "lib" / "python3.12" / "site-packages"
    installed.mkdir(parents=True)
    assert checkout_revision(installed) is None
    assert capfd.readouterr() == ("", "")
