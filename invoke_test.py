from student_parts.week01_wake_up_nana import personal_create_schedule, personal_delete_schedule, personal_list_schedules

print(personal_create_schedule.invoke({
    "title": "",
    "date": "2026-07-04",
    "start_time": "10:00",
}))
print(personal_list_schedules.invoke({"date_from": "2026-07-31", "date_to": "2026-07-04"}))
print(personal_delete_schedule.invoke({"schedule_id": ""}))