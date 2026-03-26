import argparse
import os
import shutil
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload an exported model directory to ModelScope.")
    parser.add_argument("--local-dir", required=True, help="Local exported model directory.")
    parser.add_argument("--repo-id", default="qding98/TLM_QSH", help="ModelScope repo id, e.g. username/model-name")
    parser.add_argument(
        "--token",
        default=None,
        help="ModelScope access token. Falls back to MODELSCOPE_API_TOKEN when omitted.",
    )
    parser.add_argument("--commit-message", default="Upload model from TLM", help="Commit message.")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and print the upload plan only.")
    args = parser.parse_args()

    try:
        from modelscope.hub.api import HubApi  # type: ignore
        from modelscope.hub.repository import Repository  # type: ignore
    except ImportError as exc:
        raise ImportError("Please install ModelScope first: pip install modelscope") from exc

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    token = args.token or os.environ.get("MODELSCOPE_API_TOKEN")
    if args.dry_run:
        print(f"[dry-run] Would upload {local_dir} to {args.repo_id}")
        return
    if not token:
        raise ValueError("ModelScope token is required. Pass --token or set MODELSCOPE_API_TOKEN.")

    api = HubApi()
    api.login(token)

    create_repo_errors = []
    for create_fn_name in ("create_model", "create_repo"):
        create_fn = getattr(api, create_fn_name, None)
        if create_fn is None:
            continue
        try:
            create_fn(args.repo_id)
            break
        except Exception as exc:  # repo may already exist
            create_repo_errors.append(f"{create_fn_name}: {exc}")

    with tempfile.TemporaryDirectory(prefix="modelscope_upload_") as tmpdir:
        repo_dir = Path(tmpdir) / args.repo_id.split("/")[-1]
        repo = Repository(str(repo_dir), clone_from=args.repo_id)

        for item in repo_dir.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        for item in local_dir.iterdir():
            target = repo_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

        repo.push(args.commit_message)

    if create_repo_errors:
        print("Repo creation messages:")
        for message in create_repo_errors:
            print(f"  - {message}")
    print(f"Upload complete: {args.repo_id}")


if __name__ == "__main__":
    main()
