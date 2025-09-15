from typing import List, Optional
from sqlmodel import Session, select, and_, or_
from models.schema import People, MembershipPeriods
from datetime import date


def create_person(session: Session, name: str, team: Optional[str] = None, role: Optional[str] = None) -> People:
    person = People(name=name, team=team, role=role, active=True)
    session.add(person)
    session.commit()
    session.refresh(person)

    # Create initial membership period
    membership = MembershipPeriods(
        person_id=person.person_id,
        start_date=date.today(),
        end_date=None
    )
    session.add(membership)
    session.commit()

    return person


def get_person_by_id(session: Session, person_id: int) -> Optional[People]:
    return session.get(People, person_id)


def get_person_by_name(session: Session, name: str) -> Optional[People]:
    statement = select(People).where(People.name == name)
    return session.exec(statement).first()


def get_all_people(session: Session, active_only: bool = False) -> List[People]:
    statement = select(People)
    if active_only:
        statement = statement.where(People.active == True)
    return session.exec(statement).all()


def update_person(session: Session, person_id: int, name: Optional[str] = None,
                 team: Optional[str] = None, role: Optional[str] = None) -> Optional[People]:
    person = session.get(People, person_id)
    if person:
        if name is not None:
            person.name = name
        if team is not None:
            person.team = team
        if role is not None:
            person.role = role
        session.add(person)
        session.commit()
        session.refresh(person)
    return person


def deactivate_person(session: Session, person_id: int, end_date: date = None) -> Optional[People]:
    if end_date is None:
        end_date = date.today()

    person = session.get(People, person_id)
    if not person:
        return None

    # Close current membership period
    statement = select(MembershipPeriods).where(
        and_(
            MembershipPeriods.person_id == person_id,
            MembershipPeriods.end_date.is_(None)
        )
    )
    current_period = session.exec(statement).first()

    if current_period:
        current_period.end_date = end_date
        session.add(current_period)

    # Mark person as inactive
    person.active = False
    session.add(person)
    session.commit()
    session.refresh(person)

    return person


def reactivate_person(session: Session, person_id: int, start_date: date = None) -> Optional[People]:
    if start_date is None:
        start_date = date.today()

    person = session.get(People, person_id)
    if not person:
        return None

    # Create new membership period
    new_period = MembershipPeriods(
        person_id=person_id,
        start_date=start_date,
        end_date=None
    )
    session.add(new_period)

    # Mark person as active
    person.active = True
    session.add(person)
    session.commit()
    session.refresh(person)

    return person


def get_active_people_on_date(session: Session, target_date: date) -> List[People]:
    statement = (
        select(People)
        .join(MembershipPeriods)
        .where(
            and_(
                MembershipPeriods.start_date <= target_date,
                or_(
                    MembershipPeriods.end_date.is_(None),
                    MembershipPeriods.end_date >= target_date
                )
            )
        )
        .distinct()
    )
    return session.exec(statement).all()


def get_membership_periods(session: Session, person_id: int) -> List[MembershipPeriods]:
    statement = select(MembershipPeriods).where(MembershipPeriods.person_id == person_id).order_by(MembershipPeriods.start_date)
    return session.exec(statement).all()


def is_person_active_on_date(session: Session, person_id: int, target_date: date) -> bool:
    statement = select(MembershipPeriods).where(
        and_(
            MembershipPeriods.person_id == person_id,
            MembershipPeriods.start_date <= target_date,
            or_(
                MembershipPeriods.end_date.is_(None),
                MembershipPeriods.end_date >= target_date
            )
        )
    )
    return session.exec(statement).first() is not None