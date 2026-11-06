from __future__ import annotations

import ast
import tempfile
from dataclasses import fields, replace
from pathlib import Path


class ConfigBase:
    def with_overrides(self, items: list[str]):
        known = {f.name for f in fields(self)}
        updates = {}
        for item in items:
            key, sep, raw = item.partition("=")
            if not sep or key not in known:
                raise ValueError(f"Invalid override: {item!r} (valid keys: {sorted(known)})")
            try:
                updates[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                updates[key] = raw
        return replace(self, **updates)

    def for_run(self, output_dir: Path):
        if not self.debug:
            return self, output_dir
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return self.debug_variant(), debug_dir

    def cache_dir_or(self, name: str) -> Path:
        return Path(self.cache_dir) if self.cache_dir else Path(tempfile.gettempdir()) / name
