import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixed.config import CONFIG
from student_parts.week04_retrieve_nanas_memory import add_personal_reference

# @tool 실행 시 -> invoke 필요
print(add_personal_reference.invoke({
    "title": "테스트 참고자료",
    "content": "이것은 테스트용 참고자료입니다.",
    "tags": ["테스트", "참고자료"]
}))

