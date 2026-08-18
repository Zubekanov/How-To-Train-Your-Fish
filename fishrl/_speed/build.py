"""Build the _fastenc extension in place:  python -m fishrl._speed.build

Runs setuptools' build_ext --inplace from the repo root, dropping
fishrl/_speed/_fastenc.<abi>.pyd (or .so) next to its source. Needs a C
compiler; on Windows that is MSVC Build Tools (vcvars is located automatically
by setuptools). Verifies the result imports and is wired into the encoder.
"""
import os
import sys

from setuptools import Extension, setup


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(repo)
    setup(
        name="fishrl-speed",
        script_args=["build_ext", "--inplace"],
        ext_modules=[Extension("fishrl._speed._fastenc",
                               sources=["fishrl/_speed/_fastenc.c"])],
    )
    from fishrl._speed import _fastenc            # noqa: F401 -- import proves the build
    from fishrl.obs import encoder
    active = getattr(encoder._fill_zone, "__module__", "")
    print(f"\n_fastenc built and importable; encoder fast path active: "
          f"{'YES' if 'fastenc' in repr(encoder._fill_zone) or active.endswith('_fastenc') else 'NO (restart python)'}")


if __name__ == "__main__":
    main()
