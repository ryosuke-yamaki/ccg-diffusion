"""Drop the Qwen3.8-Flash-Next checkpoint's clean pages from the page cache (no root needed)."""
import glob
import os

snapshots = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/*/*")
for path in glob.glob(snapshots):
    fd = os.open(os.path.realpath(path), os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
