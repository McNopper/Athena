"""
Athena package.

Athena is named after the Greek goddess of knowledge, wisdom, and intelligence.

Educational note:
    The modules in this project print rich progress output using Unicode
    characters (checkmarks, arrows, emojis) to make the learning experience
    friendlier. On some platforms (notably Windows consoles that default to a
    legacy code page such as cp1252) writing those characters raises a
    UnicodeEncodeError. To keep every script and ``python -m src.<module>``
    self-test runnable everywhere, we reconfigure the standard streams to UTF-8
    as soon as the package is imported.
"""

import sys as _sys

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(_sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Stream may be detached or non-reconfigurable; fall back silently.
            pass
