import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fixed.config import CONFIG
from student_parts.week04_retrieve_nanas_memory import add_personal_reference

# @tool 실행 시 -> invoke 필요
print(add_personal_reference.invoke({
    "title": "검증용 참고 문서",
    "content": "시스템 테스트를 위한 참고 문서입니다.",
    "tags": ["검증", "문서"]
}))

print(add_personal_reference.invoke({
    "title": "시범 운영용 참조 정보",
    "content": "본 내용은 기능 확인을 목적으로 작성된 참조 정보입니다.",
    "tags": ["시범운영", "참조정보"]
}))

print(add_personal_reference.invoke({
    "title": "Test Reference Data",
    "content": "This is sample data for testing purposes.",
    "tags": ["test", "sample"]
}))

print(add_personal_reference.invoke({
    "title":"고양이 사료 배급 시간 기억하기",
    "content":"고양이에게 사료를 배급할때는 아침 7시, 저녁 7시에 배급을 하여야 합니다.",
    "tags":["반려동물", "고양이"]
    }))

print(add_personal_reference.invoke({
    "title":"고양이 사료 배급 시간 기억하기",
    "content":"고양이에게 사료를 배급할때는 아침 7시, 저녁 7시에 배급을 하여야 합니다.",
    "tags":["반려동물", "고양이"]
    }))

print(add_personal_reference.invoke({
    "title":"산책 시간",
    "content":"매일 산책을 나가는 개인 시간은 아침 8시입니다.",
    "tags":["산책"]
    }))

print(add_personal_reference.invoke({
    "title":"걷기 시간",
    "content":"매일 밖으로 나가 걷기를 시작하는 시간은 아침 8시입니다.",
    "tags":["걷기"]
    }))

