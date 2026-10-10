
from pathlib import Path
from shutil import copy2
import hashlib
import os

from app.services.document_registry import get_document

DOCUMENT_ID = "doc_260827cbbff7465c916b4f582fd69f05"
STAGING = Path("data_rebuild_test8")
BACKUP = Path("data_backup_test8_before_replacement")

KEYS = (
    "extracted_file_path",
    "chunks_file_path",
    "embeddings_file_path",
    "faiss_index_path",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


document = get_document(DOCUMENT_ID)

if document is None:
    raise SystemExit("Document not found.")

files = []

# Validate all paths and backups before changing anything.
for key in KEYS:
    target = Path(document[key])
    staged = STAGING / target.name
    backup = BACKUP / target.name

    if not all(p.is_file() for p in (target, staged, backup)):
        raise RuntimeError(f"Missing file for {key}")

    if sha256(target) != sha256(backup):
        raise RuntimeError(
            f"Original and backup differ: {key}"
        )

    files.append((key, staged, target))

print("Pre-install checks passed.")

# Prepare temporary files beside their destinations.
prepared = []

try:
    for key, staged, target in files:
        temp = target.with_name(target.name + ".rebuild_tmp")

        if temp.exists():
            raise RuntimeError(
                f"Temporary file already exists: {temp}"
            )

        copy2(staged, temp)

        if sha256(staged) != sha256(temp):
            raise RuntimeError(
                f"Copy verification failed: {key}"
            )

        prepared.append((key, temp, target))

    print("All replacement files prepared.")

    # Replace files only after every copy is verified.
    for key, temp, target in prepared:
        os.replace(temp, target)
        print("Installed:", key)

except Exception:
    print("Installation interrupted or failed.")
    print("Original backups remain available.")
    raise

# Verify installed files.
for key, staged, target in files:
    if sha256(staged) != sha256(target):
        raise RuntimeError(
            f"Installed file verification failed: {key}"
        )

print("\nPASS: All four rebuilt artifacts installed.")
print("Document ID preserved:", DOCUMENT_ID)
print("Database record unchanged.")
