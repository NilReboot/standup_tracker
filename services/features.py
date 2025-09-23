from typing import List, Dict, Optional
from sqlmodel import Session, select
from models.schema import Passes, People, Meetings
from datetime import date, timedelta


def calculate_turns_since_spoke(session: Session, person_id: int, current_meeting_id: int,
                              current_pass_seq: int, lookback_meetings: int = 5) -> int:
    """
    Calculate number of turns (passes) since this person last spoke.
    Looks across current meeting and previous meetings up to lookback_meetings.
    """
    # Get current meeting date
    current_meeting = session.get(Meetings, current_meeting_id)
    if not current_meeting:
        return float('inf')  # Unknown, assign high value

    # Get recent meetings including current one
    cutoff_date = current_meeting.meeting_date - timedelta(days=lookback_meetings * 7)  # Assume ~weekly meetings

    meetings_stmt = (
        select(Meetings)
        .where(Meetings.meeting_date >= cutoff_date)
        .where(Meetings.meeting_date <= current_meeting.meeting_date)
        .order_by(Meetings.meeting_date.desc(), Meetings.meeting_id.desc())
    )
    recent_meetings = session.exec(meetings_stmt).all()

    turns_count = 0

    # Check current meeting first (up to current pass)
    if current_meeting_id in [m.meeting_id for m in recent_meetings]:
        current_passes_stmt = (
            select(Passes)
            .where(Passes.meeting_id == current_meeting_id)
            .where(Passes.seq < current_pass_seq)  # Only passes before current position
            .order_by(Passes.seq.desc())
        )
        current_passes = session.exec(current_passes_stmt).all()

        for pass_record in current_passes:
            if pass_record.from_person_id == person_id:
                return turns_count
            turns_count += 1

    # Check previous meetings
    for meeting in recent_meetings:
        if meeting.meeting_id == current_meeting_id:
            continue  # Already handled above

        passes_stmt = (
            select(Passes)
            .where(Passes.meeting_id == meeting.meeting_id)
            .order_by(Passes.seq.desc())
        )
        meeting_passes = session.exec(passes_stmt).all()

        for pass_record in meeting_passes:
            if pass_record.from_person_id == person_id:
                return turns_count
            turns_count += 1

    return turns_count  # If person never spoke, return total turns since we started looking


def calculate_spoke_today(session: Session, person_id: int, meeting_id: int) -> bool:
    """
    Binary indicator if this person has already spoken in today's meeting.
    """
    passes_stmt = (
        select(Passes)
        .where(Passes.meeting_id == meeting_id)
        .where(Passes.from_person_id == person_id)
    )
    passes = session.exec(passes_stmt).all()
    return len(passes) > 0


def calculate_same_team_as_current(session: Session, person_id: int, current_speaker_id: int) -> bool:
    """
    Binary indicator if this person is on the same team as current speaker.
    """
    person = session.get(People, person_id)
    current_speaker = session.get(People, current_speaker_id)

    if not person or not current_speaker:
        return False

    # Handle None team assignments
    if person.team is None or current_speaker.team is None:
        return False

    return person.team == current_speaker.team


def calculate_days_since_current_picked_them(session: Session, person_id: int,
                                           current_speaker_id: int, lookback_days: int = 90) -> int:
    """
    Days since current speaker last selected this person.
    Returns a large number if never selected.
    """
    cutoff_date = date.today() - timedelta(days=lookback_days)

    # Find most recent pass from current_speaker to person
    passes_stmt = (
        select(Passes, Meetings)
        .join(Meetings, Passes.meeting_id == Meetings.meeting_id)
        .where(Passes.from_person_id == current_speaker_id)
        .where(Passes.to_person_id == person_id)
        .where(Meetings.meeting_date >= cutoff_date)
        .order_by(Meetings.meeting_date.desc(), Passes.seq.desc())
    )

    recent_pass = session.exec(passes_stmt).first()

    if not recent_pass:
        return lookback_days + 1  # Return max + 1 if never picked

    pass_record, meeting = recent_pass
    days_since = (date.today() - meeting.meeting_date).days
    return days_since


def calculate_typical_position(session: Session, person_id: int, lookback_meetings: int = 20) -> float:
    """
    Calculate person's average speaking position in historical meetings.
    Returns value between 0 (always speaks first) and 1 (always speaks last).
    """
    # Get recent meetings
    cutoff_date = date.today() - timedelta(days=lookback_meetings * 7)  # Assume ~weekly meetings

    meetings_stmt = (
        select(Meetings)
        .where(Meetings.meeting_date >= cutoff_date)
        .order_by(Meetings.meeting_date.desc())
    )
    recent_meetings = session.exec(meetings_stmt).all()

    positions = []

    for meeting in recent_meetings:
        # Get all passes for this meeting
        passes_stmt = (
            select(Passes)
            .where(Passes.meeting_id == meeting.meeting_id)
            .order_by(Passes.seq)
        )
        meeting_passes = session.exec(passes_stmt).all()

        if not meeting_passes:
            continue

        # Find person's speaking position
        total_speakers = len(set(p.from_person_id for p in meeting_passes))
        if total_speakers <= 1:
            continue  # Skip meetings with only one or no speakers

        # Find when this person first spoke
        for idx, pass_record in enumerate(meeting_passes):
            if pass_record.from_person_id == person_id:
                # Position as fraction (0 = first, 1 = last)
                position = idx / (total_speakers - 1) if total_speakers > 1 else 0.5
                positions.append(position)
                break

    if not positions:
        return 0.5  # Default to middle if no history

    return sum(positions) / len(positions)


def calculate_current_position_in_queue(remaining_speakers: int, total_attendees: int) -> float:
    """
    Calculate how many people are left / total people (0-1 scale).
    0 = early in meeting, 1 = late in meeting.
    """
    if total_attendees <= 1:
        return 0.5

    # Current position as fraction through the meeting
    people_who_spoke = total_attendees - remaining_speakers
    return people_who_spoke / (total_attendees - 1) if total_attendees > 1 else 0.5


def build_feature_matrix(session: Session, meeting_date: date, current_speaker_id: int,
                        candidate_ids: List[int], exclude_person_ids: Optional[List[int]] = None) -> List[Dict]:
    """
    Build feature matrix for all candidate next speakers.

    Args:
        session: Database session
        meeting_date: Date of current meeting
        current_speaker_id: ID of person currently speaking
        candidate_ids: List of person IDs who could speak next
        exclude_person_ids: List of person IDs to exclude (already spoken)

    Returns:
        List of feature dictionaries, one per candidate
    """
    if exclude_person_ids is None:
        exclude_person_ids = []

    # Get current meeting
    from services.meetings import get_meeting_by_date
    meeting = get_meeting_by_date(session, meeting_date)
    if not meeting:
        return []

    # Get current pass sequence number
    current_passes_stmt = (
        select(Passes)
        .where(Passes.meeting_id == meeting.meeting_id)
        .order_by(Passes.seq.desc())
    )
    all_passes = session.exec(current_passes_stmt).all()
    current_pass_seq = len(all_passes) + 1  # Next sequence number

    # Get total attendees for current meeting
    from services.attendance import get_attendees_for_meeting
    attendees = get_attendees_for_meeting(session, meeting.meeting_id)
    total_attendees = len(attendees)
    remaining_speakers = len([p for p in attendees if p.person_id not in exclude_person_ids])

    features = []

    for candidate_id in candidate_ids:
        if candidate_id in exclude_person_ids:
            continue

        feature_dict = {
            'person_id': candidate_id,
            'turns_since_spoke': calculate_turns_since_spoke(
                session, candidate_id, meeting.meeting_id, current_pass_seq
            ),
            'spoke_today': calculate_spoke_today(session, candidate_id, meeting.meeting_id),
            'same_team_as_current': calculate_same_team_as_current(
                session, candidate_id, current_speaker_id
            ),
            'days_since_current_picked_them': calculate_days_since_current_picked_them(
                session, candidate_id, current_speaker_id
            ),
            'typical_position': calculate_typical_position(session, candidate_id),
            'current_position_in_queue': calculate_current_position_in_queue(
                remaining_speakers, total_attendees
            )
        }

        features.append(feature_dict)

    return features


def normalize_features(features: List[Dict]) -> List[Dict]:
    """
    Normalize features to have similar scales for the model.

    Args:
        features: List of feature dictionaries

    Returns:
        List of feature dictionaries with normalized values
    """
    if not features:
        return features

    # Features that need normalization (continuous values)
    continuous_features = ['turns_since_spoke', 'days_since_current_picked_them', 'typical_position', 'current_position_in_queue']

    # Calculate min/max for each continuous feature
    feature_ranges = {}
    for feature_name in continuous_features:
        values = [f[feature_name] for f in features if feature_name in f]
        if values:
            min_val = min(values)
            max_val = max(values)
            feature_ranges[feature_name] = (min_val, max_val)

    # Normalize features
    normalized_features = []
    for feature_dict in features:
        normalized_dict = feature_dict.copy()

        for feature_name in continuous_features:
            if feature_name in feature_ranges and feature_name in normalized_dict:
                min_val, max_val = feature_ranges[feature_name]
                if max_val > min_val:
                    # Min-max normalization to [0, 1]
                    normalized_dict[feature_name] = (normalized_dict[feature_name] - min_val) / (max_val - min_val)
                else:
                    # All values are the same
                    normalized_dict[feature_name] = 0.5

        normalized_features.append(normalized_dict)

    return normalized_features