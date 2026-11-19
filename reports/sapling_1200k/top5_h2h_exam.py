"""Run the Top-5 head-to-head student on the frozen elite_exam map and both seats."""
from pathlib import Path

import elite_exam


if __name__ == "__main__":
    elite_exam.ARMS["top5_h2h_v2"] = "top5_h2h_v2.npz"
    elite_exam.SOURCE_FILES.append(Path(__file__).resolve())
    elite_exam.main()
