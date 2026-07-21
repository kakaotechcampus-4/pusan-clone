import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from week03_build_nanas_logbook import *

result = extract_schedule_request.invoke({
    "query":"내일 3시에 회의"
})

print(extract_schedule_request.name)
print(extract_schedule_request.__name__)
