from typing import List
from sqlmodel import Session, select
from models.schema import Attendance, People


def get_attendance_for_meeting(session: Session, meeting_id: int) -> List[Attendance]:
    statement = select(Attendance).where(Attendance.meeting_id == meeting_id)
    return session.exec(statement).all()


def get_attendance_for_person(session: Session, person_id: int) -> List[Attendance]:
    statement = select(Attendance).where(Attendance.person_id == person_id)
    return session.exec(statement).all()


def mark_attendance(session: Session, meeting_id: int, person_id: int, present: bool) -> Attendance:
    statement = select(Attendance).where(
        Attendance.meeting_id == meeting_id,
        Attendance.person_id == person_id
    )
    existing = session.exec(statement).first()

    if existing:
        existing.present = present
        session.add(existing)
    else:
        attendance = Attendance(
            meeting_id=meeting_id,
            person_id=person_id,
            present=present
        )
        session.add(attendance)
        existing = attendance

    session.commit()
    session.refresh(existing)
    return existing


def get_attendees_for_meeting(session: Session, meeting_id: int) -> List[People]:
    statement = (
        select(People)
        .join(Attendance)
        .where(Attendance.meeting_id == meeting_id)
        .where(Attendance.present == True)
    )
    return session.exec(statement).all()


def bulk_mark_attendance(session: Session, meeting_id: int, attendance_data: List[tuple[int, bool]]) -> List[Attendance]:
    results = []
    for person_id, present in attendance_data:
        attendance = mark_attendance(session, meeting_id, person_id, present)
        results.append(attendance)
    return results