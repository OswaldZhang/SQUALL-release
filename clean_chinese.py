from pathlib import Path
import re

ROOT = Path(".")
BACKUP = True

#  + 
CHINESE_RE = re.compile(
    r"[\u4e00-\u9fff"
    r"\u3400-\u4dbf"
    r"\uf900-\ufaff"
    r"\u3000-\u303f"
    r"\uff00-\uffef]"
)

SKIP_DIRS = {".git", "__pycache__", "__MACOSX", ".ipynb_checkpoints"}
SKIP_PREFIXES = ("._",)
SKIP_SUFFIXES = (".bak", ".zhbak")

def should_skip(path: Path):
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    if path.name.startswith(SKIP_PREFIXES):
        return True
    if path.name.endswith(SKIP_SUFFIXES):
        return True
    return False

changed = 0

for path in ROOT.rglob("*.py"):
    if should_skip(path):
        continue

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print("[skip non-utf8]", path)
        continue

    new_text = CHINESE_RE.sub("", text)

    if new_text != text:
        if BACKUP:
            backup_path = path.with_suffix(path.suffix + ".zhbak")
            if not backup_path.exists():
                backup_path.write_text(text, encoding="utf-8")

        path.write_text(new_text, encoding="utf-8")
        print("[cleaned]", path)
        changed += 1

print("Done. changed files:", changed)
