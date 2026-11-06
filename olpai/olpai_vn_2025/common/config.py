from __future__ import annotations

import ast
from dataclasses import fields, replace

from common.paths import find_data_root, resolve_output_dir


class ConfigBase:
    def with_overrides(self, items: list[str]):
        known = {f.name for f in fields(self)}
        updates = {}
        for item in items:
            key, sep, raw = item.partition("=")
            if not sep or key not in known:
                raise ValueError(f"invalid parameter: {item!r} (valid keys: {sorted(known)})")
            try:
                updates[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                updates[key] = raw
        return replace(self, **updates)

    def with_paths(self, track: str, marker: str, **extra):
        return replace(self, data_root=str(find_data_root(self.data_root, track, marker)),
                       output_dir=str(resolve_output_dir(self.output_dir, track)), **extra)
