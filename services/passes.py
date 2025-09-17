from typing import List, Optional, Tuple
from sqlmodel import Session, select
from models.schema import Passes, People, Meetings
from datetime import date


def create_pass(session: Session, meeting_id: int, from_person_id: int, to_person_id: int, seq: int) -> Passes:
    pass_record = Passes(
        meeting_id=meeting_id,
        from_person_id=from_person_id,
        to_person_id=to_person_id,
        seq=seq
    )
    session.add(pass_record)
    session.commit()
    session.refresh(pass_record)
    return pass_record


def get_passes_for_meeting(session: Session, meeting_id: int) -> List[Passes]:
    statement = select(Passes).where(Passes.meeting_id == meeting_id).order_by(Passes.seq)
    return session.exec(statement).all()


def get_passes_from_person(session: Session, person_id: int) -> List[Passes]:
    statement = select(Passes).where(Passes.from_person_id == person_id)
    return session.exec(statement).all()


def get_passes_to_person(session: Session, person_id: int) -> List[Passes]:
    statement = select(Passes).where(Passes.to_person_id == person_id)
    return session.exec(statement).all()


def get_all_passes(session: Session) -> List[Passes]:
    statement = select(Passes).order_by(Passes.meeting_id, Passes.seq)
    return session.exec(statement).all()


def get_next_pass_sequence(session: Session, meeting_id: int) -> int:
    statement = select(Passes.seq).where(Passes.meeting_id == meeting_id).order_by(Passes.seq.desc()).limit(1)
    result = session.exec(statement).first()
    return (result + 1) if result else 1


def get_pass_history_for_pair(session: Session, from_person_id: int, to_person_id: int) -> List[Passes]:
    statement = select(Passes).where(
        Passes.from_person_id == from_person_id,
        Passes.to_person_id == to_person_id
    ).order_by(Passes.meeting_id, Passes.seq)
    return session.exec(statement).all()


def get_recent_passes(session: Session, days: int = 30) -> List[Passes]:
    from datetime import date, timedelta
    cutoff_date = date.today() - timedelta(days=days)

    statement = (
        select(Passes)
        .join(Meetings, Passes.meeting_id == Meetings.meeting_id)
        .where(Meetings.meeting_date >= cutoff_date)
        .order_by(Meetings.meeting_date, Passes.seq)
    )
    return session.exec(statement).all()


def undo_last_pass(session: Session, meeting_id: int) -> bool:
    """
    Delete the last pass for a specific meeting (highest seq number).
    Returns True if a pass was deleted, False if no passes exist.
    """
    # Get the pass with the highest sequence number for this meeting
    statement = (
        select(Passes)
        .where(Passes.meeting_id == meeting_id)
        .order_by(Passes.seq.desc())
        .limit(1)
    )
    last_pass = session.exec(statement).first()

    if last_pass:
        session.delete(last_pass)
        session.commit()
        return True

    return False