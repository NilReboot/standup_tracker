from typing import List, Optional
from sqlmodel import Session, select
from models.schema import Turns, People
from datetime import date


def create_turn(session: Session, meeting_id: int, seq: int, speaker_id: int, duration_sec: Optional[int] = None) -> Turns:
    turn = Turns(
        meeting_id=meeting_id,
        seq=seq,
        speaker_id=speaker_id,
        duration_sec=duration_sec
    )
    session.add(turn)
    session.commit()
    session.refresh(turn)
    return turn


def get_turns_for_meeting(session: Session, meeting_id: int) -> List[Turns]:
    statement = select(Turns).where(Turns.meeting_id == meeting_id).order_by(Turns.seq)
    return session.exec(statement).all()


def get_turn_by_id(session: Session, turn_id: int) -> Optional[Turns]:
    return session.get(Turns, turn_id)


def update_turn_duration(session: Session, turn_id: int, duration_sec: int) -> Optional[Turns]:
    turn = session.get(Turns, turn_id)
    if turn:
        turn.duration_sec = duration_sec
        session.add(turn)
        session.commit()
        session.refresh(turn)
    return turn


def get_all_turns_for_person(session: Session, person_id: int) -> List[Turns]:
    statement = select(Turns).where(Turns.speaker_id == person_id).order_by(Turns.meeting_id, Turns.seq)
    return session.exec(statement).all()


def get_next_sequence_number(session: Session, meeting_id: int) -> int:
    statement = select(Turns.seq).where(Turns.meeting_id == meeting_id).order_by(Turns.seq.desc()).limit(1)
    result = session.exec(statement).first()
    return (result + 1) if result else 1