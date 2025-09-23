from typing import List, Dict, Tuple, Optional
from sqlmodel import Session, select
from models.schema import Passes, People, Meetings, Attendance
from datetime import date, timedelta
import numpy as np
from pathlib import Path

from .features import build_feature_matrix, normalize_features
from .plackett_luce import PlackettLuceModel, get_model_path


def collect_training_data(session: Session, min_history_days: int = 90) -> List[Dict]:
    """
    Collect historical data for training the Plackett-Luce model.

    Args:
        session: Database session
        min_history_days: How far back to look for training data

    Returns:
        List of training instances with features and chosen outcomes
    """
    cutoff_date = date.today() - timedelta(days=min_history_days)

    # Get historical meetings
    meetings_stmt = (
        select(Meetings)
        .where(Meetings.meeting_date >= cutoff_date)
        .order_by(Meetings.meeting_date)
    )
    historical_meetings = session.exec(meetings_stmt).all()

    training_data = []

    for meeting in historical_meetings:
        # Get all passes for this meeting
        passes_stmt = (
            select(Passes)
            .where(Passes.meeting_id == meeting.meeting_id)
            .order_by(Passes.seq)
        )
        meeting_passes = session.exec(passes_stmt).all()

        if len(meeting_passes) < 2:  # Need at least 2 passes for meaningful training
            continue

        # Get attendees for this meeting
        from .attendance import get_attendees_for_meeting
        attendees = get_attendees_for_meeting(session, meeting.meeting_id)
        attendee_ids = [p.person_id for p in attendees]

        # For each pass, create a training instance
        for i, pass_record in enumerate(meeting_passes):
            current_speaker_id = pass_record.from_person_id
            chosen_person_id = pass_record.to_person_id

            # Get people who had already spoken before this pass
            people_who_spoke = set()
            for prev_pass in meeting_passes[:i]:
                people_who_spoke.add(prev_pass.from_person_id)

            # Candidate pool: attendees who haven't spoken yet (excluding current speaker)
            candidate_ids = [
                p_id for p_id in attendee_ids
                if p_id not in people_who_spoke and p_id != current_speaker_id
            ]

            if len(candidate_ids) < 2 or chosen_person_id not in candidate_ids:
                continue

            # Build features for all candidates at this decision point
            features = build_feature_matrix(
                session=session,
                meeting_date=meeting.meeting_date,
                current_speaker_id=current_speaker_id,
                candidate_ids=candidate_ids,
                exclude_person_ids=list(people_who_spoke)
            )

            if not features:
                continue

            # Normalize features
            features = normalize_features(features)

            training_instance = {
                'features': features,
                'chosen_person_id': chosen_person_id,
                'meeting_context': {
                    'meeting_id': meeting.meeting_id,
                    'meeting_date': meeting.meeting_date,
                    'pass_sequence': pass_record.seq,
                    'current_speaker_id': current_speaker_id
                }
            }

            training_data.append(training_instance)

    return training_data


def train_or_load_model(session: Session, force_retrain: bool = False,
                       min_history_days: int = 90) -> PlackettLuceModel:
    """
    Train a new Plackett-Luce model or load existing one.

    Args:
        session: Database session
        force_retrain: Whether to force retraining even if model exists
        min_history_days: Days of history to use for training

    Returns:
        Trained PlackettLuceModel
    """
    model_path = get_model_path()

    # Try to load existing model if not forcing retrain
    if not force_retrain and model_path.exists():
        try:
            model = PlackettLuceModel()
            model.load_model(str(model_path))
            return model
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            print("Training new model...")

    # Collect training data
    training_data = collect_training_data(session, min_history_days)

    if len(training_data) < 10:  # Need minimum amount of training data
        print(f"Warning: Only {len(training_data)} training instances found. Model may not be reliable.")

    # Initialize and train model
    model = PlackettLuceModel(
        regularization_strength=1.0,
        max_iter=1000,
        feature_scaling=True,
        random_state=42
    )

    if training_data:
        model.fit(training_data)

        # Save the trained model
        try:
            model.save_model(str(model_path))
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    return model


def predict_next_speakers_pl(session: Session, meeting_date: date, current_speaker_id: int,
                           top_k: int = 3, exclude_person_ids: Optional[List[int]] = None,
                           force_retrain: bool = False) -> List[Tuple[People, float]]:
    """
    Predict the top-k most likely next speakers using Plackett-Luce model.

    Args:
        session: Database session
        meeting_date: Date of the meeting
        current_speaker_id: ID of current speaker
        top_k: Number of top predictions to return
        exclude_person_ids: Person IDs to exclude (already spoken)
        force_retrain: Whether to force model retraining

    Returns:
        List of (Person, probability) tuples, ordered by probability (descending)
    """
    if exclude_person_ids is None:
        exclude_person_ids = []

    # Get attendees for this meeting who are active on the meeting date
    from .people import get_active_people_on_date
    from .attendance import get_attendees_for_meeting
    from .meetings import get_meeting_by_date

    meeting = get_meeting_by_date(session, meeting_date)
    if not meeting:
        return []

    # Get people who are both present at meeting and active on the date
    attendees = get_attendees_for_meeting(session, meeting.meeting_id)
    active_people = get_active_people_on_date(session, meeting_date)

    # Find intersection - people who are both present and active, excluding those who have already spoken
    active_attendee_ids = {p.person_id for p in active_people}.intersection({p.person_id for p in attendees})
    candidate_ids = list(active_attendee_ids - set(exclude_person_ids) - {current_speaker_id})

    if not candidate_ids:
        return []

    # Load or train model
    try:
        model = train_or_load_model(session, force_retrain)
    except Exception as e:
        print(f"Error with Plackett-Luce model: {e}")
        return _fallback_to_simple_prediction(session, attendees, candidate_ids, top_k)

    # Build features for current candidates
    try:
        features = build_feature_matrix(
            session=session,
            meeting_date=meeting_date,
            current_speaker_id=current_speaker_id,
            candidate_ids=candidate_ids,
            exclude_person_ids=exclude_person_ids
        )

        if not features:
            return _fallback_to_simple_prediction(session, attendees, candidate_ids, top_k)

        # Normalize features
        features = normalize_features(features)

        # Get predictions
        probabilities = model.predict_proba(features)

        if len(probabilities) == 0:
            return _fallback_to_simple_prediction(session, attendees, candidate_ids, top_k)

        # Create results with Person objects
        predictions = []
        for i, prob in enumerate(probabilities):
            person_id = features[i]['person_id']
            person = session.get(People, person_id)
            if person:
                predictions.append((person, float(prob)))

        # Sort by probability (descending) and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    except Exception as e:
        print(f"Error during prediction: {e}")
        return _fallback_to_simple_prediction(session, attendees, candidate_ids, top_k)


def _fallback_to_simple_prediction(session: Session, attendees: List[People],
                                 candidate_ids: List[int], top_k: int) -> List[Tuple[People, float]]:
    """
    Fallback to simple equal probability prediction when model fails.
    """
    candidates = [p for p in attendees if p.person_id in candidate_ids]
    if not candidates:
        return []

    equal_prob = 1.0 / len(candidates)
    predictions = [(person, equal_prob) for person in candidates]
    return predictions[:top_k]


def get_model_training_info(session: Session) -> Dict:
    """
    Get information about the current model training status.

    Returns:
        Dictionary with training information
    """
    model_path = get_model_path()

    info = {
        'model_exists': model_path.exists(),
        'model_path': str(model_path),
        'last_modified': None,
        'training_data_count': 0
    }

    if model_path.exists():
        import os
        from datetime import datetime
        stat = os.stat(model_path)
        info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    # Count available training data
    training_data = collect_training_data(session)
    info['training_data_count'] = len(training_data)

    return info


def retrain_model(session: Session, min_history_days: int = 90) -> Dict:
    """
    Force retrain the model and return training results.

    Returns:
        Dictionary with training results
    """
    try:
        model = train_or_load_model(session, force_retrain=True, min_history_days=min_history_days)

        info = get_model_training_info(session)
        info['retrain_success'] = True
        info['retrain_message'] = "Model retrained successfully"

        if hasattr(model, 'get_feature_importance'):
            try:
                info['feature_importance'] = model.get_feature_importance()
            except:
                info['feature_importance'] = {}

        return info

    except Exception as e:
        return {
            'retrain_success': False,
            'retrain_message': f"Failed to retrain model: {str(e)}",
            'model_exists': False
        }