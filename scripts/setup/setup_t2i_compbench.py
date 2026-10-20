"""Install the official T2I-CompBench++ evaluation stack as a pinned, isolated environment.

The official implementation pins torch 2.0.1 / transformers 4.30.2 / spacy 3.5.3, which are
incompatible with the CCG Diffusion runtime (torch 2.13 / transformers 5.14).  Everything this script
builds therefore lives under ``external/t2i_compbench/`` (git-ignored) and is driven through a
separate Python 3.10 interpreter -- the *eval venv*.  The main interpreter is never touched.

The script is idempotent: each step verifies what is already on disk (commit, sha256, byte count,
import smoke) and skips the work when it matches.  Re-running after a partial failure resumes.

Steps
  1. clone the official repo at ``T2ICB_COMMIT`` (or verify an existing clone is at that commit)
  2. create the eval venv on CPython 3.10 via ``uv``
  3. install torch/torchvision from the cu118 index (the official pin)
  4. install every other package of the tracked freeze ``docker/t2i_compbench/venv_freeze.txt``
     (the venv of the paper evaluation; derived from the official list as in ``requirements.eval.txt``)
  5. install detectron2 at its pinned commit as a CPU extension build
  6. download the four evaluation weights, verifying byte count and sha256
  7. link the weights where the official scripts look for them
  8. write ``venv_freeze.txt`` and require it to equal the tracked freeze; check the tracked
     source-pin mirror JSON, which is left unchanged (its sha256 is pinned by the dataset contract)

The official clone is treated as read-only: this script only adds weight files under
``UniDet_eval/experts/expert_weights/`` and never edits tracked files inside it.

HOW TO LAUNCH THE EVAL VENV
    The container exports a ``LD_LIBRARY_PATH`` whose first entry is the main environment's
    python3.12 torch, and ``PYTHONPATH=src``.  An eval venv interpreter that inherits either
    one loads the wrong ``libtorch_python.so`` and dies immediately with
    ``ImportError: undefined symbol: PyObject_CallOneArg``.  Never invoke the venv python
    bare.  Either go through ``eval_venv_env()`` from this module::

        subprocess.run([str(venv_python()), ...], env=eval_venv_env())

    or, from a shell::

        env -u PYTHONPATH -u LD_LIBRARY_PATH external/t2i_compbench/.venv/bin/python ...

    Both forms are verified: torch reports ``cuda_available=True`` either way.
    ``--smoke`` runs the bundled sample images through three metrics using this environment,
    which is the quickest way to confirm a fresh setup end to end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ccg_diffusion.evaluation.t2i_compbench.dataset import EVALUATOR_WEIGHTS, SOURCE_COMMIT, SOURCE_REPOSITORY


REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Official sources -------------------------------------------------------------------------
T2ICB_REPOSITORY = SOURCE_REPOSITORY
T2ICB_COMMIT = SOURCE_COMMIT
DETECTRON2_REPOSITORY = "https://github.com/facebookresearch/detectron2.git"
DETECTRON2_COMMIT = "5aeb252b194b93dc2879b4ac34bc51a31b5aee13"

# --- Eval venv --------------------------------------------------------------------------------
# The interpreter of the tracked source pin (venv/python_version), which validate_official_venv requires.
EVAL_PYTHON_VERSION = "3.10.20"
TORCH_CU118_INDEX = "https://download.pytorch.org/whl/cu118"
TORCH_REQUIREMENTS = ("torch==2.0.1+cu118", "torchvision==0.15.2+cu118")
# torch 2.0.1's cpp_extension imports pkg_resources, dropped in setuptools >= 81.
BUILD_SETUPTOOLS_REQUIREMENT = "setuptools==69.5.1"
# opencv-python needs libGL.so.1; this container has no root to install it.  Same OpenCV
# version, headless wheel -- the eval path uses only decode/resize, never highgui.
HEADLESS_OPENCV_REQUIREMENT = "opencv-python-headless==4.7.0.72"

# Lines of the tracked freeze installed by their own step: torch/torchvision from the cu118 index (step 3)
# and detectron2 without build isolation (step 5).
FREEZE_SEPARATE_PREFIXES = ("torch==", "torchvision==", "detectron2 @ ")

# --- Evaluation weights -----------------------------------------------------------------------
# ``link`` is relative to the clone root, or None for the torch.hub cache (BLIP resolves its
# checkpoint URL through timm's download_cached_file, which reads torch.hub.get_dir()).
WEIGHTS: tuple[dict[str, Any], ...] = EVALUATOR_WEIGHTS

# --- Prompt files -----------------------------------------------------------------------------
# Every category ships 300 val prompts.  Six of the eight files have no trailing newline, so
# `wc -l` reports 299 for them; count with splitlines() instead.
CATEGORIES = ("color", "shape", "texture", "spatial", "3d_spatial", "non_spatial", "complex", "numeracy")
EXPECTED_VAL_PROMPTS = 300

# --- Paths ------------------------------------------------------------------------------------
EXTERNAL_ROOT = REPO_ROOT / "external/t2i_compbench"
CLONE_DIR = EXTERNAL_ROOT / "T2I-CompBench"
VENV_DIR = EXTERNAL_ROOT / ".venv"
WEIGHTS_DIR = EXTERNAL_ROOT / "weights"
FREEZE_REQUIREMENTS = EXTERNAL_ROOT / "requirements.freeze.txt"
FREEZE_PATH = EXTERNAL_ROOT / "venv_freeze.txt"
TRACKED_FREEZE_PATH = REPO_ROOT / "docker/t2i_compbench/venv_freeze.txt"
PIN_MIRROR_PATH = REPO_ROOT / "output/analysis/t2i_compbench/t2icb_source_pin.json"
# Fields of the source pin that a new setup changes without changing the evaluator.
VOLATILE_PIN_FIELDS = ("retrieved_at",)

# The container exports a LD_LIBRARY_PATH that puts the main environment's (python3.12) torch shared
# objects first, and a PYTHONPATH pointing at the repository's src.  Either one makes the eval venv's torch load
# the wrong libtorch_python.so.  Keep only the driver/CUDA entries.
KEPT_LIBRARY_PATHS = ("/usr/local/cuda/compat/lib", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64")


def eval_venv_env(*, hide_gpu: bool = False, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return the environment for running the eval venv's interpreter.

    Drops PYTHONPATH and the main environment's torch entries from LD_LIBRARY_PATH so the eval venv resolves
    its own torch.  With *hide_gpu* the CUDA devices are masked, which is what makes detectron2's
    setup.py pick CppExtension instead of a CUDAExtension it cannot compile here.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["LD_LIBRARY_PATH"] = ":".join(KEPT_LIBRARY_PATHS)
    if hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["FORCE_CUDA"] = "0"
    if extra:
        env.update(extra)
    return env


def venv_python() -> Path:
    """Path to the eval venv interpreter."""
    return VENV_DIR / "bin/python"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """Run *command*, streaming its output, and raise on a non-zero exit."""
    print(f"$ {' '.join(str(part) for part in command)}", flush=True)
    subprocess.run(list(command), cwd=None if cwd is None else str(cwd), env=env, check=True)


def uv_executable() -> str:
    """Locate the uv binary, installing it into the user site if absent."""
    found = shutil.which("uv") or shutil.which("uv", path=str(Path.home() / ".local/bin"))
    if found:
        return found
    print("uv not found; installing into the user site", flush=True)
    try:
        run([sys.executable, "-m", "pip", "install", "--user", "uv"])
    except subprocess.CalledProcessError:
        run([sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "uv"])
    found = shutil.which("uv") or shutil.which("uv", path=str(Path.home() / ".local/bin"))
    if not found:
        raise RuntimeError("uv is still not on PATH after installation")
    return found


def uv_pip(uv: str, *arguments: str, hide_gpu: bool = False) -> None:
    """Run ``uv pip`` against the eval venv."""
    run(
        [uv, "pip", *arguments, "--python", str(venv_python())],
        env=eval_venv_env(hide_gpu=hide_gpu, extra={"UV_LINK_MODE": "copy"}),
    )


# --- Step 1: pinned clone ----------------------------------------------------------------------
def ensure_clone() -> None:
    """Clone the official repo at the pinned commit, or verify an existing clone."""
    if CLONE_DIR.exists():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(CLONE_DIR),
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if head != T2ICB_COMMIT:
            raise RuntimeError(f"{CLONE_DIR} is at {head}, expected {T2ICB_COMMIT}")
        print(f"clone already at {T2ICB_COMMIT}")
        return
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", T2ICB_REPOSITORY, str(CLONE_DIR)])
    run(["git", "checkout", "--quiet", T2ICB_COMMIT], cwd=CLONE_DIR)


# --- Step 2-5: eval venv ------------------------------------------------------------------------
def freeze_requirements() -> list[str]:
    """The lines of the tracked freeze that step 4 installs (everything but ``FREEZE_SEPARATE_PREFIXES``)."""
    lines = TRACKED_FREEZE_PATH.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() and not line.startswith(FREEZE_SEPARATE_PREFIXES)]


def ensure_venv(uv: str) -> None:
    """Create the eval venv and install exactly the packages of the tracked freeze.

    Every package is installed with ``--no-deps`` at its frozen version, so the
    venv holds the frozen set (including the headless OpenCV wheel, the CLIP
    commit, the spaCy model and the setuptools/wheel used by the detectron2 build).
    """
    if not venv_python().exists():
        run([uv, "venv", str(VENV_DIR), "--python", EVAL_PYTHON_VERSION])

    FREEZE_REQUIREMENTS.write_text("\n".join(freeze_requirements()) + "\n", encoding="utf-8")

    uv_pip(uv, "install", "--no-deps", *TORCH_REQUIREMENTS, "--index-url", TORCH_CU118_INDEX)
    uv_pip(uv, "install", "--no-deps", "-r", str(FREEZE_REQUIREMENTS))

    # detectron2 compiles against the venv's torch, so build isolation has to be off.  The GPU is
    # masked for the build only: with it visible, setup.py selects a CUDAExtension and the build
    # fails on the system nvcc 12.6 vs torch cu118 mismatch.
    uv_pip(
        uv, "install", "--no-deps", "--no-build-isolation",
        f"detectron2 @ git+{DETECTRON2_REPOSITORY}@{DETECTRON2_COMMIT}",
        hide_gpu=True,
    )


def verify_imports() -> dict[str, str]:
    """Import the whole eval stack in the venv and return the versions it reports."""
    probe = """
import json
import torch, torchvision, detectron2, transformers, timm, spacy, numpy, cv2, diffusers, accelerate
from detectron2 import _C  # noqa: F401  -- fails loudly if the C++ extension is missing
from detectron2.layers import DeformConv, ModulatedDeformConv, ROIAlign, nms_rotated  # noqa: F401
import clip, pycocotools, word2number  # noqa: F401
spacy.load("en_core_web_sm")
print(json.dumps({
    "python": ".".join(str(p) for p in __import__("sys").version_info[:3]),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": str(torch.cuda.is_available()),
    "detectron2": detectron2.__version__,
    "transformers": transformers.__version__,
    "timm": timm.__version__,
    "spacy": spacy.__version__,
    "numpy": numpy.__version__,
    "opencv": cv2.__version__,
    "diffusers": diffusers.__version__,
    "accelerate": accelerate.__version__,
}))
"""
    completed = subprocess.run(
        [str(venv_python()), "-c", probe],
        env=eval_venv_env(), check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


# --- Step 6-7: weights ---------------------------------------------------------------------------
def weight_is_present(spec: dict[str, Any]) -> bool:
    """True when the weight is on disk with the pinned byte count and digest."""
    path = WEIGHTS_DIR / str(spec["name"])
    if not path.exists() or path.stat().st_size != spec["bytes"]:
        return False
    return sha256_file(path) == spec["sha256"]


def ensure_weights() -> None:
    """Download every evaluation weight and verify it against its pin."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in WEIGHTS:
        path = WEIGHTS_DIR / str(spec["name"])
        if weight_is_present(spec):
            print(f"{spec['name']}: already present and verified")
            continue
        print(f"{spec['name']}: downloading from {spec['url']}", flush=True)
        with tempfile.NamedTemporaryFile(dir=WEIGHTS_DIR, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(str(spec["url"])) as response:
                shutil.copyfileobj(response, temporary)
        try:
            actual_bytes = temporary_path.stat().st_size
            if actual_bytes != spec["bytes"]:
                raise ValueError(f"{spec['name']}: expected {spec['bytes']} bytes, got {actual_bytes}")
            actual_sha = sha256_file(temporary_path)
            if actual_sha != spec["sha256"]:
                raise ValueError(f"{spec['name']}: expected sha256 {spec['sha256']}, got {actual_sha}")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def torch_hub_checkpoint_dir() -> Path:
    """Ask the eval venv's torch where timm's download_cached_file would put a checkpoint."""
    completed = subprocess.run(
        [str(venv_python()), "-c", "import torch; print(torch.hub.get_dir())"],
        env=eval_venv_env(), check=True, capture_output=True, text=True,
    )
    return Path(completed.stdout.strip()) / "checkpoints"


def link_weights() -> None:
    """Point the official scripts at the downloaded weights without editing the clone."""
    torch_hub_checkpoints = torch_hub_checkpoint_dir()
    for spec in WEIGHTS:
        source = WEIGHTS_DIR / str(spec["name"])
        link = spec["link"]
        target = (CLONE_DIR / str(link)) if link else (torch_hub_checkpoints / str(spec["name"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        print(f"linked {target} -> {source}")


# --- Step 8: records ------------------------------------------------------------------------------
def freeze_differences(live: str, tracked: str) -> list[str]:
    """``-line`` for each tracked freeze line missing from the live freeze, ``+line`` for each extra one."""
    live_lines, tracked_lines = set(live.splitlines()), set(tracked.splitlines())
    return ([f"-{line}" for line in sorted(tracked_lines - live_lines)]
            + [f"+{line}" for line in sorted(live_lines - tracked_lines)])


def write_freeze() -> str:
    """Write the eval venv's frozen requirement list, require it to equal the tracked freeze, return its sha256."""
    uv = uv_executable()
    completed = subprocess.run(
        [uv, "pip", "freeze", "--color", "never", "--python", str(venv_python())],
        env=eval_venv_env(), check=True, capture_output=True, text=True,
    )
    FREEZE_PATH.write_text(completed.stdout, encoding="utf-8")
    tracked = TRACKED_FREEZE_PATH.read_text(encoding="utf-8")
    if completed.stdout != tracked:
        differences = freeze_differences(completed.stdout, tracked)
        raise RuntimeError(f"the eval venv ({FREEZE_PATH}) differs from the tracked freeze {TRACKED_FREEZE_PATH}: "
                           f"{differences or 'same packages, different bytes'}")
    return sha256_file(FREEZE_PATH)


def collect_val_txt() -> dict[str, dict[str, Any]]:
    """Digest and count every category's val prompt file.

    Records two independent quirks of the distributed files.  Both happen to split the eight
    categories 6:2, which makes them easy to confuse, so they are stored under separate keys:

    ``line_ending``       six files are CRLF, 3d_spatial and numeracy are LF.
    ``trailing_newline``  only shape and non_spatial end in a newline; for the other six
                          ``wc -l`` reports 299 while there are really 300 prompts.

    Neither one perturbs a prompt string as long as the file is read in text mode: universal
    newlines fold CRLF to LF, and splitlines() does not invent a final empty line.  Reading in
    binary, or with newline='', would leave a stray CR inside the prompt and therefore inside
    every generated filename.
    """
    result: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        path = CLONE_DIR / "examples/dataset" / f"{category}_val.txt"
        raw = path.read_bytes()
        count = len(raw.decode("utf-8").splitlines())
        if count != EXPECTED_VAL_PROMPTS:
            raise ValueError(f"{path} has {count} prompts, expected {EXPECTED_VAL_PROMPTS}")
        crlf = raw.count(b"\r\n")
        result[category] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "count": count,
            "line_ending": "CRLF" if crlf else "LF",
            "crlf_count": crlf,
            "trailing_newline": raw.endswith(b"\n"),
        }
    return result


def deviation_records() -> list[dict[str, str]]:
    """Every departure from the official requirements, with the reason it was needed."""
    return [
        {
            "item": "opencv-python",
            "official": "opencv-python==4.7.0.72",
            "installed": HEADLESS_OPENCV_REQUIREMENT,
            "reason": "opencv-python needs libGL.so.1; this container has no root to install it. "
                      "Same OpenCV version, headless wheel. The eval path uses decode/resize only.",
        },
        {
            "item": "detectron2 build",
            "official": f"git+{DETECTRON2_REPOSITORY}@{DETECTRON2_COMMIT} (default CUDA extension)",
            "installed": "same commit, CPU extension build (CUDA_VISIBLE_DEVICES='' FORCE_CUDA=0)",
            "reason": "System nvcc is 12.6 while torch is cu118, so a CUDA extension build fails "
                      "torch's _check_cuda_version. UniDet's RS200/R50 configs set no "
                      "DEFORM_ON_PER_STAGE, so no deformable-conv CUDA kernel is reached; "
                      "ROIAlign/NMS come from torchvision and still run on GPU.",
        },
        {
            "item": "setuptools",
            "official": "unpinned (implicit build dependency)",
            "installed": BUILD_SETUPTOOLS_REQUIREMENT,
            "reason": "torch 2.0.1's cpp_extension imports pkg_resources, removed in setuptools>=81. "
                      "Build-time only; no runtime pin is affected.",
        },
        {
            "item": "Unified_learned_OCIM_R50_6x+2x.pth source",
            "official": "gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq",
            "installed": "huggingface.co/datasets/nqbinh/misc_upload (byte- and sha256-verified)",
            "reason": "The official Google Drive id refuses anonymous access, and shikunl/prismer "
                      "hosts only the RS200 weight. The mirror is pinned by sha256 and was checked "
                      "to load into the official R50 config with no missing or mismatched keys.",
        },
        {
            "item": "word2number",
            "official": "absent from requirements.txt",
            "installed": "word2number==1.1",
            "reason": "numeracy_eval.py imports it; the official requirements omit it.",
        },
    ]


def pin_payload(versions: dict[str, str], freeze_sha256: str) -> dict[str, Any]:
    """The source pin of this setup: every pinned external identity."""
    return {
        "repo": T2ICB_REPOSITORY,
        "commit": T2ICB_COMMIT,
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clone_path": str(CLONE_DIR.relative_to(REPO_ROOT)),
        "weights": [
            {
                "name": spec["name"],
                "url": spec["url"],
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "used_by": spec["used_by"],
                "official_source": spec["official_source"],
            }
            for spec in WEIGHTS
        ],
        "val_txt": collect_val_txt(),
        "venv": {
            "path": str(VENV_DIR.relative_to(REPO_ROOT)),
            "interpreter": str((VENV_DIR / "bin/python").relative_to(REPO_ROOT)),
            # Written verbatim into the tracked pin mirror, whose sha256 the T2I-CompBench++ dataset
            # contract pins (source.pin_sha256): do not reword it.
            "launch_requirement": (
                "Do not invoke this interpreter bare. PYTHONPATH must be unset and the "
                "container's LD_LIBRARY_PATH must be dropped or stripped of "
                "/usr/local/lib/python3.12/dist-packages/torch/lib; otherwise it loads holsd's "
                "torch 2.13 libtorch_python.so and aborts with "
                "'ImportError: undefined symbol: PyObject_CallOneArg'. Use "
                "scripts.setup.setup_t2i_compbench.eval_venv_env(), or "
                "'env -u PYTHONPATH -u LD_LIBRARY_PATH <interpreter>'."
            ),
            "launch_env_kept_library_paths": list(KEPT_LIBRARY_PATHS),
            "python_version": versions["python"],
            "torch": versions["torch"],
            "torchvision": versions["torchvision"],
            "torch_cuda": versions["torch_cuda"],
            "detectron2": versions["detectron2"],
            "detectron2_commit": DETECTRON2_COMMIT,
            "detectron2_build": "CPU extension (CUDA_VISIBLE_DEVICES='' FORCE_CUDA=0)",
            "transformers": versions["transformers"],
            "timm": versions["timm"],
            "spacy": versions["spacy"],
            "numpy": versions["numpy"],
            "opencv": versions["opencv"],
            "diffusers": versions["diffusers"],
            "accelerate": versions["accelerate"],
            "freeze_sha256": freeze_sha256,
        },
        "deviations": deviation_records(),
    }


def pin_differences(new: dict[str, Any], tracked: dict[str, Any]) -> list[str]:
    """Dotted paths at which a new source pin differs from the tracked one, ``VOLATILE_PIN_FIELDS`` excluded."""
    missing = object()

    def walk(a: Any, b: Any, path: str) -> list[str]:
        if isinstance(a, dict) and isinstance(b, dict):
            return [found for key in sorted(set(a) | set(b))
                    for found in walk(a.get(key, missing), b.get(key, missing), f"{path}.{key}")]
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [found for index, (x, y) in enumerate(zip(a, b)) for found in walk(x, y, f"{path}[{index}]")]
        return [] if a == b else [path]

    keys = sorted((set(new) | set(tracked)) - set(VOLATILE_PIN_FIELDS))
    return [found for key in keys for found in walk(new.get(key, missing), tracked.get(key, missing), key)]


def write_pin_mirror(versions: dict[str, str], freeze_sha256: str) -> None:
    """Check the git-tracked mirror of every pinned external identity, or write it when it is missing.

    The tracked mirror is pinned by sha256 in the T2I-CompBench++ dataset contract
    (``source.pin_sha256``), so it is left untouched when this setup's pin equals
    it apart from ``VOLATILE_PIN_FIELDS``; any other difference is an error.
    """
    payload = pin_payload(versions, freeze_sha256)
    if PIN_MIRROR_PATH.exists():
        differences = pin_differences(payload, json.loads(PIN_MIRROR_PATH.read_text(encoding="utf-8")))
        if differences:
            raise RuntimeError(f"this setup differs from the tracked source pin {PIN_MIRROR_PATH} at {differences}; "
                               "the pin is part of the dataset contract and is not rewritten")
        print(f"{PIN_MIRROR_PATH} matches this setup (apart from {', '.join(VOLATILE_PIN_FIELDS)}); left unchanged")
        return
    PIN_MIRROR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=PIN_MIRROR_PATH.parent, delete=False, encoding="utf-8"
    ) as temporary:
        json.dump(payload, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(PIN_MIRROR_PATH)
    print(f"wrote {PIN_MIRROR_PATH}")


SMOKE_DEFAULT_DIR = REPO_ROOT / "output/vqa/20260803_t2icb_sd14ref/t2icb_work/smoke"


def run_smoke(smoke_dir: Path) -> dict[str, float]:
    """Score the four bundled sample images with three metrics and return what they printed.

    This exercises the whole chain -- eval venv launch environment, BLIP checkpoint resolution
    through the torch.hub cache, the CPU-extension detectron2 against the UniDet RS200 weight,
    and CLIP's own download -- so a green run means a fresh setup is genuinely usable.  The
    values are not compared against anything: they only have to be numbers.
    """
    samples = smoke_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for image in sorted((CLONE_DIR / "examples/samples").glob("*.png")):
        shutil.copy2(image, samples / image.name)

    blip_dir = CLONE_DIR / "BLIPvqa_eval"
    unidet_dir = CLONE_DIR / "UniDet_eval"
    run([str(venv_python()), "BLIP_vqa.py", f"--out_dir={smoke_dir}"], cwd=blip_dir, env=eval_venv_env())
    run([str(venv_python()), "2D_spatial_eval.py", f"--outpath={smoke_dir}"], cwd=unidet_dir, env=eval_venv_env())
    run(
        [str(venv_python()), str(CLONE_DIR / "CLIPScore_eval/CLIP_similarity.py"), f"--outpath={smoke_dir}"],
        env=eval_venv_env(),
    )

    def read_score(path: Path, prefix: str) -> float:
        text = path.read_text(encoding="utf-8").strip()
        if prefix not in text:
            raise RuntimeError(f"{path} does not contain {prefix!r}: {text!r}")
        return float(text.split(prefix, 1)[1].strip())

    # 2D_spatial writes "score avg:" into avg_score.txt even though it prints "avg score:".
    scores = {
        "blip_vqa": read_score(smoke_dir / "annotation_blip/blip_vqa_score.txt", "BLIP-VQA score:"),
        "unidet_2d_spatial": read_score(smoke_dir / "labels/annotation_obj_detection_2d/avg_score.txt", "score avg:"),
        "clip_score": read_score(smoke_dir / "annotation_clip/score_avg.txt", "score avg:"),
    }
    print(json.dumps(scores, indent=2))
    return scores


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full setup, or the subset selected on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-venv", action="store_true", help="Assume the eval venv is already built")
    parser.add_argument("--skip-weights", action="store_true", help="Assume the weights are already present")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Check the clone, imports and weight digests without installing or downloading",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Score the bundled sample images with BLIP-VQA, UniDet 2D spatial and CLIPScore (needs a GPU)",
    )
    parser.add_argument(
        "--smoke-dir", type=Path, default=SMOKE_DEFAULT_DIR,
        help=f"Where --smoke writes its samples and official scratch (default: {SMOKE_DEFAULT_DIR})",
    )
    arguments = parser.parse_args(argv)

    ensure_clone()

    if arguments.smoke:
        run_smoke(arguments.smoke_dir)
        return 0

    if arguments.verify_only:
        versions = verify_imports()
        missing = [str(spec["name"]) for spec in WEIGHTS if not weight_is_present(spec)]
        print(json.dumps({"versions": versions, "unverified_weights": missing}, indent=2))
        return 1 if missing else 0

    if not arguments.skip_venv:
        ensure_venv(uv_executable())
    if not arguments.skip_weights:
        ensure_weights()
    link_weights()

    versions = verify_imports()
    write_pin_mirror(versions, write_freeze())
    print(json.dumps(versions, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
