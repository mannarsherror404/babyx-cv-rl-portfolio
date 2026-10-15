import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import memory
import rl_agent


class CoreTests(unittest.TestCase):
    def test_policy_path_is_project_relative(self):
        self.assertEqual(Path(rl_agent.POLICY_PATH).parent, Path(rl_agent.__file__).parent)

    def test_feedback_round_trip_and_candidate_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            policy_file = str(Path(temp_dir) / "policy.json")
            with patch.object(rl_agent, "POLICY_PATH", policy_file):
                agent = rl_agent.RLAgent()
                agent.record_outcome("greeting", "old reply", 5)
                agent.record_outcome("greeting", "current reply", 1)

                self.assertEqual(
                    agent.choose_action("greeting", ["current reply"], epsilon=0),
                    "current reply",
                )
                self.assertEqual(
                    agent.choose_action("greeting", ["new reply"], epsilon=0),
                    "new reply",
                )

                reloaded = rl_agent.RLAgent()
                self.assertEqual(reloaded.policy["greeting"]["old reply"]["score"], 5)

    def test_taught_name_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_file = str(Path(temp_dir) / "baby_memory.json")
            with patch.object(memory, "MEMORY_PATH", memory_file):
                learned = memory.BabyMemory()
                learned.teach("cup", "my red cup")
                self.assertEqual(memory.BabyMemory().recall("cup"), "my red cup")


if __name__ == "__main__":
    unittest.main()
