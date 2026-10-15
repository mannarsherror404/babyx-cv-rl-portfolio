#!/usr/bin/env python3
# ============================================================
# memory.py  — BabyX label→name memory
# ------------------------------------------------------------
# Stores what the user teaches:
#   base_label (YOLO) → list of taught names
#
# Example baby_memory.json:
# {
#   "cell phone": ["my phone", "Keshav phone", "Jhanvi phone"],
#   "bottle": ["baby bottle"]
# }
#
# • memory.teach(label, name)   → appends to the list
# • memory.recall(label)        → returns the most recent name
# ============================================================

import json
import os

MEMORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "baby_memory.json")


class BabyMemory:
    def __init__(self):
        # internal store: {label: [name1, name2, ...]}
        self.data = {}
        self._load()

    # --------------------------------------------------------
    # LOAD / SAVE
    # --------------------------------------------------------
    def _load(self):
        """Load from baby_memory.json, support old format too."""
        if not os.path.exists(MEMORY_PATH):
            print("🍼 No previous memory found, starting fresh.")
            self.data = {}
            return

        try:
            with open(MEMORY_PATH, "r") as f:
                raw = json.load(f)

            fixed = {}
            for label, value in raw.items():
                # old format: "cell phone": "my phone"
                if isinstance(value, str):
                    fixed[label] = [value]
                # new format: list of names
                elif isinstance(value, list):
                    # ensure all entries are strings
                    fixed[label] = [str(x) for x in value if isinstance(x, (str, int, float))]
                else:
                    # ignore weird entries
                    continue

            self.data = fixed
            print("🧠 Loaded memory from baby_memory.json")
        except Exception as e:
            print(f"⚠️ Failed to load memory: {e}")
            self.data = {}

    def _save(self):
        """Save to baby_memory.json."""
        try:
            with open(MEMORY_PATH, "w") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            # you can comment this out if it’s too spammy
            # print(f"💾 Saved memory → {MEMORY_PATH}")
        except Exception as e:
            print(f"⚠️ Failed to save memory: {e}")

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------
    def teach(self, base_label: str, taught_name: str):
        """
        Add a new taught name under a label.
        Does NOT overwrite previous names, just appends.
        """
        if not base_label or not taught_name:
            return

        base_label = str(base_label).strip()
        taught_name = str(taught_name).strip()

        if not base_label or not taught_name:
            return

        names = self.data.get(base_label)
        if not isinstance(names, list):
            names = []

        names.append(taught_name)
        self.data[base_label] = names
        self._save()

        print(f"👶 Learned: {base_label} → {taught_name}")

    def recall(self, base_label: str):
        """
        Return the most recent taught name for this label.
        (So existing code still works the same.)
        """
        names = self.data.get(base_label)
        if isinstance(names, list) and names:
            return names[-1]  # last taught
        if isinstance(names, str):
            return names      # super old format, just in case
        return None

    def all_names(self, base_label: str):
        """
        Return the full list of taught names for this label.
        Handy for debugging / UI later.
        """
        names = self.data.get(base_label)
        if isinstance(names, list):
            return names[:]
        if isinstance(names, str):
            return [names]
        return []

    def dump(self):
        """Print everything (for quick debugging in Python REPL)."""
        print(json.dumps(self.data, indent=2, ensure_ascii=False))


# quick test
if __name__ == "__main__":
    mem = BabyMemory()
    print("Current memory:")
    mem.dump()

    # example:
    # mem.teach("cell phone", "my phone")
    # mem.teach("cell phone", "Keshav phone")
    # mem.dump()