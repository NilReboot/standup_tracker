from typing import List, Optional
from sqlmodel import Session, select
from models.schema import Meetings
from datetime import date


def create_meeting(session: Session, meeting_date: date, notes: Optional[str] = None) -> Meetings:
    meeting = Meetings(meeting_date=meeting_date, notes=notes)
    session.add(meeting)
    session.commit()
    session.refresh(meeting)
    return meeting


def get_meeting_by_date(session: Session, meeting_date: date) -> Optional[Meetings]:
    statement = select(Meetings).where(Meetings.meeting_date == meeting_date)
    return session.exec(statement).first()


def get_meeting_by_id(session: Session, meeting_id: int) -> Optional[Meetings]:
    return session.get(Meetings, meeting_id)


def get_all_meetings(session: Session, limit: Optional[int] = None) -> List[Meetings]:
    statement = select(Meetings).order_by(Meetings.meeting_date.desc())
    if limit:
        statement = statement.limit(limit)
    return session.exec(statement).all()


def update_meeting_notes(session: Session, meeting_id: int, notes: str) -> Optional[Meetings]:
    meeting = session.get(Meetings, meeting_id)
    if meeting:
        meeting.notes = notes
        session.add(meeting)
        session.commit()
        session.refresh(meeting)
    return meeting


def get_or_create_meeting(session: Session, meeting_date: date, notes: Optional[str] = None) -> Meetings:
    meeting = get_meeting_by_date(session, meeting_date)
    if not meeting:
        meeting = create_meeting(session, meeting_date, notes)
    return meeting


def reset_meeting(session: Session, meeting_id: int) -> bool:
    """
    Reset a meeting by deleting all passes and attendance records for that meeting.
    Returns True if any data was deleted, False otherwise.
    """
    from models.schema import Passes, Attendance

    # Delete all passes for this meeting
    passes_statement = select(Passes).where(Passes.meeting_id == meeting_id)
    passes_to_delete = session.exec(passes_statement).all()
    for pass_record in passes_to_delete:
        session.delete(pass_record)

    # Delete all attendance records for this meeting
    attendance_statement = select(Attendance).where(Attendance.meeting_id == meeting_id)
    attendance_to_delete = session.exec(attendance_statement).all()
    for attendance_record in attendance_to_delete:
        session.delete(attendance_record)

    session.commit()
    return len(passes_to_delete) > 0 or len(attendance_to_delete) > 0