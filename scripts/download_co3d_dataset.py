from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


CO3D_REPO_URL = "https://github.com/facebookresearch/co3d.git"
MIN_DOWNLOADER_DEPS = ["requests", "tqdm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CO3Dv2 dataset by invoking the official downloader."
    )
    parser.add_argument(
        "--download-folder",
        type=Path,
        default=Path(r"D:\datasets\co3dv2"),
        help="Target folder for CO3Dv2 files (use a short path on Windows).",
    )
    parser.add_argument(
        "--co3d-repo-dir",
        type=Path,
        default=Path("external/co3d"),
        help="Local path to clone/store the official co3d repository.",
    )
    parser.add_argument(
        "--single-sequence-subset",
        action="store_true",
        help="Download the small single-sequence subset (~8.9GB).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full CO3Dv2 dataset (~5.5TB zipped).",
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Install minimal downloader dependencies before download (requests, tqdm).",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used to run the official downloader.",
    )
    return parser.parse_args()


def _warn_path_length(path: Path) -> None:
    if len(str(path.resolve())) > 180:
        print(
            f"[WARN] Path is long ({len(str(path.resolve()))} chars): {path.resolve()}\n"
            "       Prefer a shorter root path (e.g. D:\\datasets\\co3dv2) on Windows."
        )


def _check_long_paths_enabled() -> None:
    if sys.platform != "win32":
        return
    try:
        import winreg
    except Exception:
        return

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        if int(value) != 1:
            print(
                "[WARN] Windows LongPathsEnabled is not 1. "
                "Very long paths can fail during extraction."
            )
    except PermissionError:
        print("[WARN] Cannot read LongPathsEnabled (permission denied).")
    except OSError:
        print("[WARN] Cannot read LongPathsEnabled from registry.")


def _ensure_co3d_repo(repo_dir: Path) -> None:
    if (repo_dir / "co3d" / "download_dataset.py").exists():
        print(f"Using existing co3d repo: {repo_dir}")
        return

    if shutil.which("git") is None:
        raise RuntimeError("git not found in PATH. Please install Git for Windows first.")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", CO3D_REPO_URL, str(repo_dir)]
    print("Cloning official co3d repository...")
    subprocess.run(cmd, check=True)


def _missing_downloader_deps(python_exe: str) -> list[str]:
    code = (
        "import importlib.util\n"
        f"mods={MIN_DOWNLOADER_DEPS!r}\n"
        "missing=[m for m in mods if importlib.util.find_spec(m) is None]\n"
        "print(','.join(missing))"
    )
    result = subprocess.run(
        [python_exe, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = result.stdout.strip()
    if not payload:
        return []
    return [m for m in payload.split(",") if m]


def _ensure_downloader_deps(args: argparse.Namespace, repo_dir: Path) -> None:
    missing = _missing_downloader_deps(args.python_exe)
    if not missing:
        return

    if args.install_requirements:
        print(f"Installing minimal downloader dependencies: {', '.join(missing)}")
        subprocess.run([args.python_exe, "-m", "pip", "install", *missing], check=True)
        return

    missing_str = ", ".join(missing)
    raise RuntimeError(
        "Missing downloader dependencies: "
        f"{missing_str}. Run:\n"
        f"  {args.python_exe} -m pip install {missing_str}\n"
        "or rerun with --install-requirements"
    )


def main() -> int:
    args = parse_args()

    if args.single_sequence_subset and args.full:
        raise ValueError("Choose only one of --single-sequence-subset or --full.")

    if not args.single_sequence_subset and not args.full:
        args.single_sequence_subset = True

    _check_long_paths_enabled()
    args.download_folder.mkdir(parents=True, exist_ok=True)
    repo_dir = args.co3d_repo_dir.resolve()
    _warn_path_length(args.download_folder)
    _warn_path_length(repo_dir)

    _ensure_co3d_repo(repo_dir)
    _ensure_downloader_deps(args, repo_dir)

    downloader = (repo_dir / "co3d" / "download_dataset.py").resolve()
    if not downloader.exists():
        raise FileNotFoundError(f"Official downloader not found: {downloader}")

    cmd = [
        args.python_exe,
        str(downloader),
        "--download_folder",
        str(args.download_folder.resolve()),
    ]
    if args.single_sequence_subset:
        cmd.append("--single_sequence_subset")

    mode = "single-sequence subset (~8.9GB)" if args.single_sequence_subset else "full dataset (~5.5TB)"
    print(f"Starting CO3Dv2 download: {mode}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print("\nDownload finished.")
    print(f"Dataset folder: {args.download_folder.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
