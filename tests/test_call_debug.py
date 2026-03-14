import tempfile
import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from call_debug import CallDebugLogger


class CallDebugLoggerTests(unittest.TestCase):
    def test_logger_writes_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "active_call_debug.log"
            logger = CallDebugLogger(path=path)

            logger.log("turn", "user_finished_speaking", new_state="listening")
            logger.log("transcript", "conversation_item_added", role="user", text="hello")
            logger.close(cleanup=False)

            contents = path.read_text(encoding="utf-8")
            self.assertIn("TURN | user_finished_speaking", contents)
            self.assertIn('role="user"', contents)
            self.assertIn('text="hello"', contents)

    def test_logger_cleans_file_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "active_call_debug.log"
            logger = CallDebugLogger(path=path)

            logger.log("tool", "check_meeting_slot.start", duration_minutes=30)
            time.sleep(0.05)
            logger.close(cleanup=True)

            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
