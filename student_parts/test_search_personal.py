import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from student_parts.week04_retrieve_nanas_memory import search_personal_references, search_personal_reference_hits, REFERENCE_STORE

print(search_personal_references.invoke({
    "query":"회의",
    "top_k": 2
}))

print(search_personal_references.invoke({
    "query":"반려동물",
    "top_k": 2
}))