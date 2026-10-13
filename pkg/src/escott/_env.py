"""Run-time environment checks and progress messages for the command line."""
import sys
import time


def lapack_names():
    """LAPACK library names numpy and scipy were built against (empty if the build info is unavailable)."""
    names = {}
    for mod_name in ("numpy", "scipy"):
        try:
            mod = __import__(mod_name)
            cfg = mod.show_config(mode="dicts")
            names[mod_name] = str(cfg["Build Dependencies"]["lapack"]["name"])
        except Exception:
            pass
    return names


def accelerate_warning(names=None, platform=None):
    """Warning text if numpy or scipy links Apple Accelerate on macOS (see README, 'Install'), else None."""
    platform = sys.platform if platform is None else platform
    names = lapack_names() if names is None else names
    bad = [m for m, n in names.items() if "accelerate" in n.lower()]
    if platform != "darwin" or not bad:
        return None
    return (f"warning: {' and '.join(bad)} use(s) Apple Accelerate for LAPACK. With Accelerate we saw segmentation "
            f"faults and silently wrong log-likelihoods in long tensor-train runs. Install OpenBLAS builds of numpy "
            f"and scipy (see the README, 'Install') before trusting tensor-train results.")


class Progress:
    """Prints one-line progress messages with elapsed time to stderr."""

    def __init__(self, quiet=False):
        self.t0 = time.time()
        self.quiet = quiet

    def __call__(self, msg):
        if not self.quiet:
            print(f"[{time.time() - self.t0:7.1f}s] {msg}", file=sys.stderr, flush=True)
