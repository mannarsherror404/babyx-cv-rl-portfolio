#!/usr/bin/env python3
# ============================================================
# rl_agent.py — Reinforcement Learning Memory for BabyX
# ============================================================
# This module teaches BabyX through feedback (👍 / 👎).
# - state_key → describes the context (e.g. "greeting|face")
# - action    → what Baby said
# - reward    → +1 (good) / -1 (bad) / 0 (neutral)
#
# Each feedback updates an internal policy that is saved to
# rl_policy.json, so Baby gradually learns what replies work best.
# ============================================================

import json
import os
import random
from collections import defaultdict

# Keep the policy beside this module, regardless of the launch directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(BASE_DIR, "rl_policy.json")


class RLAgent:
    def __init__(self):
        # policy[state_key][action] = {"score": float, "count": int}
        self.policy = defaultdict(lambda: defaultdict(lambda: {"score": 0.0, "count": 0}))
        self._load()

    # ------------------------------------------------------------
    # LOAD EXISTING POLICY
    # ------------------------------------------------------------
    def _load(self):
        if os.path.exists(POLICY_PATH):
            try:
                with open(POLICY_PATH, "r") as f:
                    raw = json.load(f)
                # rebuild defaultdict structure
                for state_key, actions in raw.items():
                    for act, stats in actions.items():
                        self.policy[state_key][act]["score"] = float(stats.get("score", 0.0))
                        self.policy[state_key][act]["count"] = int(stats.get("count", 0))
                print("🧠 RL policy loaded from rl_policy.json")
            except Exception as e:
                print(f"⚠️ RL load failed: {e}")
        else:
            print("ℹ️ No previous RL policy found — starting fresh.")

    # → opens Baby’s old memory file so it remembers what it learned before

    # ------------------------------------------------------------
    # SAVE POLICY TO DISK
    # ------------------------------------------------------------
    def _save(self):
        try:
            data = {}
            for state_key, actions in self.policy.items():
                data[state_key] = {}
                for act, stats in actions.items():
                    data[state_key][act] = {
                        "score": stats["score"],
                        "count": stats["count"],
                    }
            with open(POLICY_PATH, "w") as f:
                json.dump(data, f, indent=2)
            print(f"💾 RL policy saved → {POLICY_PATH}")
        except Exception as e:
            print(f"⚠️ RL save failed: {e}")

    # → stores Baby’s new learning safely in a file after every change

    # ------------------------------------------------------------
    # CHOOSE BEST ACTION (Epsilon-Greedy)
    # ------------------------------------------------------------
    def choose_action(self, state_key, candidate_actions, epsilon=0.2):
        """
        Select what Baby should say for a given situation.

        Exploration (ε): try new replies (20% of the time).
        Exploitation: choose the reply with the best average reward.
        """
        if not candidate_actions:
            return None

        # Exploration: be playful sometimes
        if random.random() < epsilon:
            return random.choice(candidate_actions)

        # Exploitation: pick highest average reward
        best_action = None
        best_score = None
        if state_key in self.policy:
            for act, stats in self.policy[state_key].items():
                if act not in candidate_actions:
                    continue
                avg = stats["score"] / max(stats["count"], 1)
                if (best_score is None) or (avg > best_score):
                    best_score = avg
                    best_action = act

        # fallback if unknown state
        return best_action if best_action else random.choice(candidate_actions)

    # → helps Baby decide what to say — sometimes tries new things, sometimes picks the best one

    # ------------------------------------------------------------
    # RECORD OUTCOME (USER FEEDBACK)
    # ------------------------------------------------------------
    def record_outcome(self, state_key, action, reward):
        """
        Store feedback from user (RL update).
        Each call:
          - updates the score for (state_key, action)
          - increments how many times this action was tried
          - auto-saves rl_policy.json after every change
        """
        st = self.policy[state_key][action]
        st["score"] += float(reward)
        st["count"] += 1

        # ✅ Auto-save after every feedback (so teacher sees it live)
        self._save()

        print(f"📈 RL updated → state={state_key!r}, action={action!r}, reward={reward}")

# → learns from your thumbs-up or thumbs-down and updates its memory

# ------------------------------------------------------------
# Quick manual test
# ------------------------------------------------------------
if __name__ == "__main__":
    rl = RLAgent()
    s = "greeting|face"
    act = rl.choose_action(s, ["hi", "hello there!", "I am learning."])
    print("🤖 Baby choice before feedback:", act)
    rl.record_outcome(s, act, reward=1.0)
    print("✅ RL test complete.")

# → small test to check if Baby is learning properly when you run the code
