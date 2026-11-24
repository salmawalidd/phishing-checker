# Feedback from users ("was this right?") – saved as JSON lines.
# On Hugging Face, the file is pushed every few minutes to a PRIVATE dataset repo (FEEDBACK_REPO), so it survives restarts.
# Without HF_TOKEN / FEEDBACK_REPO it is only kept on the Space's disk (lost when the Space restarts).
import os, json, uuid, datetime as dt
from contextlib import nullcontext
from pathlib import Path

FOLDER = Path("feedback_data")
FOLDER.mkdir(exist_ok=True)
FILE = FOLDER / f"feedback-{uuid.uuid4()}.jsonl"      # one file per app start, so restarts never overwrite each other

scheduler, MODE = None, "local"
_repo, _token = os.environ.get("FEEDBACK_REPO", "").strip(), os.environ.get("HF_TOKEN", "").strip()
if _repo and _token:
    try:
        from huggingface_hub import CommitScheduler
        scheduler = CommitScheduler(repo_id=_repo, repo_type="dataset", folder_path=FOLDER, path_in_repo="data",
                                    every=5, private=True, token=_token)
        MODE = "hub"
    except Exception as e:                      # wrong token / repo name: keep working, save locally
        print("Feedback: could not connect to the dataset repo, saving locally only:", e)

def save(record):
    record = {"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **record}
    with (scheduler.lock if scheduler else nullcontext()):
        with FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return MODE
