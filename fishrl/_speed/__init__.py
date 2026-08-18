"""Optional C fast paths for the collection hot loop.

`_fastenc` is a compiled extension (see `_fastenc.c`); build it in place with

    python -m fishrl._speed.build

which needs a C compiler (MSVC on Windows, cc on Linux). Everything here is
OPTIONAL by design: importers (fishrl.obs.encoder) fall back to the pure-Python
twin with bit-identical output when the module isn't built, so unbuilt checkouts
-- the Linux deploy scripts, the ODROID, a fresh clone -- run unchanged, just
without the speedup. The compiled .pyd/.so is git-ignored (build artefact).
"""
