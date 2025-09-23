from datetime import date
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship


class People(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}

    person_id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    active: bool = Field(default=True)
    team: Optional[str] = Field(default=None)
    role: Optional[str] = Field(default=None)

    membership_periods: list["MembershipPeriods"] = Relationship(back_populates="person")
    attendance_records: list["Attendance"] = Relationship(back_populates="person")
    turns: list["Turns"] = Relationship(back_populates="speaker")
    passes_from: list["Passes"] = Relationship(back_populates="from_person", sa_relationship_kwargs={"foreign_keys": "Passes.from_person_id"})
    passes_to: list["Passes"] = Relationship(back_populates="to_person", sa_relationship_kwargs={"foreign_keys": "Passes.to_person_id"})


class MembershipPeriods(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}

    period_id: Optional[int] = Field(default=None, primary_key=True)
    person_id: int = Field(foreign_key="people.person_id")
    start_date: date
    end_date: Optional[date] = Field(default=None)

    person: People = Relationship(back_populates="membership_periods")


class Meetings(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}

    meeting_id: Optional[int] = Field(default=None, primary_key=True)
    meeting_date: date = Field(unique=True)
    notes: Optional[str] = Field(default=None)

    attendance_records: list["Attendance"] = Relationship(back_populates="meeting")
    turns: list["Turns"] = Relationship(back_populates="meeting")
    passes: list["Passes"] = Relationship(back_populates="meeting")


class Attendance(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}

    meeting_id: int = Field(foreign_key="meetings.meeting_id", primary_key=True)
    person_id: int = Field(foreign_key="people.person_id", primary_key=True)
    present: bool

    meeting: Meetings = Relationship(back_populates="attendance_records")
    person: People = Relationship(back_populates="attendance_records")


class Turns(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}

    turn_id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.meeting_id")
    seq: int = Field(ge=1)
    speaker_id: int = Field(foreign_key="people.person_id")
    duration_sec: Optional[int] = Field(default=None)

    meeting: Meetings = Relationship(back_populates="turns")
    speaker: People = Relationship(back_populates="turns")


class Passes(SQLModel, table=True):
    __table_args__ = {'extend_existing': True}

    pass_id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meetings.meeting_id")
    from_person_id: int = Field(foreign_key="people.person_id")
    to_person_id: int = Field(foreign_key="people.person_id")
    seq: int

    meeting: Meetings = Relationship(back_populates="passes")
    from_person: People = Relationship(back_populates="passes_from", sa_relationship_kwargs={"foreign_keys": "Passes.from_person_id"})
    to_person: People = Relationship(back_populates="passes_to", sa_relationship_kwargs={"foreign_keys": "Passes.to_person_id"})