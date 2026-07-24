import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student_parts.week04_retrieve_nanas_memory import search_personal_reference_hits, REFERENCE_STORE

reference_store = REFERENCE_STORE
print(search_personal_reference_hits(reference_store, query="테스트", top_k=1))

print(search_personal_reference_hits(reference_store, query="반려동물", top_k=1))

print(search_personal_reference_hits(reference_store, query="회의 점심시간", top_k=1))

print(search_personal_reference_hits(reference_store, query="회의", top_k=2))