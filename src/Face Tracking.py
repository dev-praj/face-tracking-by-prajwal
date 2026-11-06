from pathlib import Path
import runpy


tracker_script = Path(__file__).resolve().parents[1] / "tracker.py"
runpy.run_path(str(tracker_script), run_name="__main__")
