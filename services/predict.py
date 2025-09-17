from typing import List, Dict, Tuple, Optional
from sqlmodel import Session, select
from models.schema import Passes, People, Meetings, Attendance
from collections import defaultdict, Counter
from datetime import date, timedelta
import math


def get_pass_history_for_prediction(session: Session, min_history_days: int = 30) -> List[Passes]:
    cutoff_date = date.today() - timedelta(days=min_history_days)

    statement = (
        select(Passes)
        .join(Meetings, Passes.meeting_id == Meetings.meeting_id)
        .where(Meetings.meeting_date >= cutoff_date)
        .order_by(Meetings.meeting_date, Passes.seq)
    )
    return session.exec(statement).all()


def build_transition_matrix(passes: List[Passes], alpha: float = 1.0, lambda_decay: float = 0.95) -> Dict[int, Dict[int, float]]:
    """
    Build first-order Markov transition matrix with Laplace smoothing and exponential decay.
    """
    # Count transitions with exponential decay based on recency
    transition_counts = defaultdict(lambda: defaultdict(float))
    all_people = set()

    # Sort passes by meeting date (most recent first for decay calculation)
    passes_by_meeting = defaultdict(list)
    for pass_record in passes:
        passes_by_meeting[pass_record.meeting_id].append(pass_record)
        all_people.add(pass_record.from_person_id)
        all_people.add(pass_record.to_person_id)

    # Calculate decay weights based on how recent each meeting is
    meeting_dates = []
    for meeting_id in passes_by_meeting:
        meeting_passes = passes_by_meeting[meeting_id]
        # Use the first pass to get meeting reference (they all have same meeting_id)
        meeting_dates.append((meeting_id, meeting_passes))

    # Sort by meeting_id (assuming higher ID = more recent)
    meeting_dates.sort(key=lambda x: x[0], reverse=True)

    # Apply exponential decay and count transitions
    for idx, (meeting_id, meeting_passes) in enumerate(meeting_dates):
        decay_factor = lambda_decay ** idx
        for pass_record in meeting_passes:
            transition_counts[pass_record.from_person_id][pass_record.to_person_id] += decay_factor

    # Apply Laplace smoothing and normalize
    transition_matrix = {}
    for from_person in all_people:
        transition_matrix[from_person] = {}

        # Calculate total count for normalization (including smoothing)
        total_count = sum(transition_counts[from_person].values()) + alpha * len(all_people)

        for to_person in all_people:
            if from_person != to_person:  # Can't pass to yourself
                smoothed_count = transition_counts[from_person][to_person] + alpha
                transition_matrix[from_person][to_person] = smoothed_count / total_count
            else:
                transition_matrix[from_person][to_person] = 0.0

    return transition_matrix


def predict_next_speakers(session: Session, meeting_date: date, current_speaker_id: int,
                         alpha: float = 1.0, lambda_decay: float = 0.95,
                         min_history_days: int = 30, top_k: int = 3,
                         exclude_person_ids: Optional[List[int]] = None) -> List[Tuple[People, float]]:
    """
    Predict the top-k most likely next speakers using Markov model.
    Only considers people who are present and active on the meeting date.
    Excludes people in exclude_person_ids (e.g., those who have already spoken today).
    """
    if exclude_person_ids is None:
        exclude_person_ids = []
    # Get attendees for this meeting who are active on the meeting date
    from services.people import get_active_people_on_date
    from services.attendance import get_attendees_for_meeting
    from services.meetings import get_meeting_by_date

    meeting = get_meeting_by_date(session, meeting_date)
    if not meeting:
        return []

    # Get people who are both present at meeting and active on the date
    attendees = get_attendees_for_meeting(session, meeting.meeting_id)
    active_people = get_active_people_on_date(session, meeting_date)

    # Find intersection - people who are both present and active, excluding those who have already spoken
    active_attendee_ids = {p.person_id for p in active_people}.intersection({p.person_id for p in attendees})
    active_attendee_ids = active_attendee_ids - set(exclude_person_ids)

    if not active_attendee_ids or current_speaker_id not in (active_attendee_ids | set(exclude_person_ids)):
        return []

    # Get historical pass data for building the model
    passes = get_pass_history_for_prediction(session, min_history_days)

    if not passes:
        # No historical data - return random prediction from active attendees (excluding those who have spoken)
        other_attendees = [p for p in attendees if p.person_id != current_speaker_id and p.person_id in active_attendee_ids]
        equal_prob = 1.0 / len(other_attendees) if other_attendees else 0.0
        return [(person, equal_prob) for person in other_attendees[:top_k]]

    # Build transition matrix
    transition_matrix = build_transition_matrix(passes, alpha, lambda_decay)

    # Get predictions for current speaker
    if current_speaker_id not in transition_matrix:
        # Current speaker has no history - return equal probabilities (excluding those who have spoken)
        other_attendees = [p for p in attendees if p.person_id != current_speaker_id and p.person_id in active_attendee_ids]
        equal_prob = 1.0 / len(other_attendees) if other_attendees else 0.0
        return [(person, equal_prob) for person in other_attendees[:top_k]]

    # Filter predictions to only include active attendees (excluding current speaker and those who have spoken)
    predictions = []
    for person_id, prob in transition_matrix[current_speaker_id].items():
        if person_id in active_attendee_ids and person_id != current_speaker_id:
            person = session.get(People, person_id)
            if person:
                predictions.append((person, prob))

    # Renormalize probabilities since we've excluded some people
    if predictions:
        total_prob = sum(prob for _, prob in predictions)
        if total_prob > 0:
            predictions = [(person, prob / total_prob) for person, prob in predictions]

    # Sort by probability (descending) and return top-k
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]


def get_pass_statistics(session: Session, person_id: int, days: int = 30) -> Dict:
    """
    Get pass statistics for a person over the last N days.
    """
    cutoff_date = date.today() - timedelta(days=days)

    # Passes from this person
    statement_from = (
        select(Passes)
        .join(Meetings, Passes.meeting_id == Meetings.meeting_id)
        .where(Passes.from_person_id == person_id)
        .where(Meetings.meeting_date >= cutoff_date)
    )
    passes_from = session.exec(statement_from).all()

    # Passes to this person
    statement_to = (
        select(Passes)
        .join(Meetings, Passes.meeting_id == Meetings.meeting_id)
        .where(Passes.to_person_id == person_id)
        .where(Meetings.meeting_date >= cutoff_date)
    )
    passes_to = session.exec(statement_to).all()

    # Count by recipient/sender
    passes_to_counts = Counter(p.from_person_id for p in passes_from)
    passes_from_counts = Counter(p.to_person_id for p in passes_to)

    return {
        'total_passes_given': len(passes_from),
        'total_passes_received': len(passes_to),
        'most_passed_to': passes_to_counts.most_common(3),
        'most_received_from': passes_from_counts.most_common(3),
        'days_analyzed': days
    }