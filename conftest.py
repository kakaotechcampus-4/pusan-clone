"""pytest가 프로젝트 루트를 import 경로에 포함하도록 합니다."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
