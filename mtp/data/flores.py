"""Original FLORES-200 dev/devtest text, using its public official archive.

This is an explicit alternative to the gated HF loader, never an automatic
replacement with a newer corpus version. Downloads occur only on function call.
Only the requested member is read; archive paths are never extracted.
Source: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
"""

from pathlib import Path
import tarfile
import tempfile
import os

OFFICIAL_URL = "https://tinyurl.com/flores200dataset"


def _member(archive, split, code):
    matches = [
        m
        for m in archive.getmembers()
        if m.name.lstrip("./").endswith(f"{split}/{code}.{split}") and m.isfile()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one original FLORES member: {split}/{code}.{split}")
    return matches[0]


def load_flores_text(lang, split="devtest", archive_path=None):
    if lang not in {"hi", "mr"} or split not in {"dev", "devtest"}:
        raise ValueError("unsupported language/split")
    code = {"hi": "hin_Deva", "mr": "mar_Deva"}[lang]
    path = (
        Path(archive_path)
        if archive_path
        else Path(
            os.environ.get(
                "MTP_FLORES_ARCHIVE",
                Path.home() / ".cache" / "mtp" / "flores200_dataset.tar.gz",
            )
        )
    )
    if not path.exists():
        from urllib.request import urlopen

        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(OFFICIAL_URL, timeout=60) as response:
            fd, temp = tempfile.mkstemp(dir=path.parent, prefix="flores-")
            os.close(fd)
            try:
                with open(temp, "wb") as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                with tarfile.open(temp, "r:gz") as archive:
                    member = _member(archive, split, code)
                    if not member.isfile():
                        raise ValueError("requested member is not a regular file")
                os.replace(temp, path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
    with tarfile.open(path, "r:gz") as archive:
        member = _member(archive, split, code)
        file = archive.extractfile(member)
        if file is None:
            raise ValueError("requested member not readable")
        return file.read().decode("utf-8").splitlines()
