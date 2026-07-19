"""fixed/, student_parts/가 패키지로 설치되어 있지 않아도 import되도록 repo root를 sys.path에 둡니다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
