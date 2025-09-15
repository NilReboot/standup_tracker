import streamlit as st
from datetime import date
from sqlmodel import Session
from db.engine import engine
from services.meetings import get_or_create_meeting
from services.people import get_active_people_on_date, get_person_by_id
from services.attendance import get_attendees_for_meeting, bulk_mark_attendance
from services.passes import create_pass, get_passes_for_meeting, get_next_pass_sequence
from services.predict import predict_next_speakers


st.set_page_config(page_title="Today's Standup", page_icon="📅", layout="wide")

st.title("📅 Today's Standup")

# Get today's date
today = date.today()
st.subheader(f"Meeting Date: {today.strftime('%A, %B %d, %Y')}")

with Session(engine) as session:
    # Get or create today's meeting
    meeting = get_or_create_meeting(session, today)

    # Get active people for today
    active_people = get_active_people_on_date(session, today)
    current_attendees = get_attendees_for_meeting(session, meeting.meeting_id)

    if not active_people:
        st.warning("No active team members found for today. Please add people in the People page first.")
        st.stop()

    # Attendance section
    st.header("👥 Attendance")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Mark Attendance")

        # Create checkboxes for each active person
        attendance_data = {}
        current_attendee_ids = {p.person_id for p in current_attendees}

        for person in active_people:
            is_present = person.person_id in current_attendee_ids
            attendance_data[person.person_id] = st.checkbox(
                f"{person.name} ({person.team or 'No team'})",
                value=is_present,
                key=f"attendance_{person.person_id}"
            )

        if st.button("Update Attendance", type="primary"):
            # Bulk update attendance
            attendance_updates = [(person_id, present) for person_id, present in attendance_data.items()]
            bulk_mark_attendance(session, meeting.meeting_id, attendance_updates)
            st.success("Attendance updated!")
            st.rerun()

    with col2:
        st.subheader("Present Today")
        present_count = sum(1 for present in attendance_data.values() if present)
        st.metric("Attendees", f"{present_count} / {len(active_people)}")

        if current_attendees:
            st.write("**Currently marked present:**")
            for person in current_attendees:
                st.write(f"• {person.name}")

    # Pass tracking section
    if current_attendees and len(current_attendees) > 1:
        st.header("🎯 Pass Tracking")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Record a Pass")

            # Select from person (who is passing)
            from_person = st.selectbox(
                "From (current speaker):",
                options=current_attendees,
                format_func=lambda x: f"{x.name} ({x.team or 'No team'})",
                key="from_person_select"
            )

            # Select to person (who receives the pass)
            to_options = [p for p in current_attendees if p.person_id != from_person.person_id]
            to_person = st.selectbox(
                "To (next speaker):",
                options=to_options,
                format_func=lambda x: f"{x.name} ({x.team or 'No team'})",
                key="to_person_select"
            )

            if st.button("Record Pass", type="primary"):
                seq = get_next_pass_sequence(session, meeting.meeting_id)
                pass_record = create_pass(session, meeting.meeting_id, from_person.person_id, to_person.person_id, seq)
                st.success(f"Pass recorded: {from_person.name} → {to_person.name}")
                st.rerun()

        with col2:
            st.subheader("AI Predictions")

            if 'from_person_select' in st.session_state and st.session_state.from_person_select:
                current_speaker = st.session_state.from_person_select

                # Get predictions
                predictions = predict_next_speakers(
                    session, today, current_speaker.person_id,
                    alpha=1.0, lambda_decay=0.95, min_history_days=30, top_k=3
                )

                if predictions:
                    st.write(f"**Top 3 likely next speakers after {current_speaker.name}:**")
                    for idx, (person, prob) in enumerate(predictions, 1):
                        confidence = f"{prob*100:.1f}%"
                        st.write(f"{idx}. **{person.name}** ({confidence})")
                else:
                    st.info("No predictions available yet. Record more passes to improve predictions!")
            else:
                st.info("Select a current speaker to see AI predictions.")

    # Today's passes
    st.header("📋 Today's Passes")
    passes = get_passes_for_meeting(session, meeting.meeting_id)

    if passes:
        st.write(f"**{len(passes)} passes recorded today:**")
        for idx, pass_record in enumerate(passes, 1):
            from_person = get_person_by_id(session, pass_record.from_person_id)
            to_person = get_person_by_id(session, pass_record.to_person_id)
            st.write(f"{idx}. {from_person.name} → {to_person.name}")
    else:
        st.info("No passes recorded yet for today.")

    # Meeting notes
    st.header("📝 Meeting Notes")
    notes = st.text_area(
        "Optional notes for today's standup:",
        value=meeting.notes or "",
        height=100,
        key="meeting_notes"
    )

    if st.button("Save Notes"):
        from services.meetings import update_meeting_notes
        update_meeting_notes(session, meeting.meeting_id, notes)
        st.success("Notes saved!")