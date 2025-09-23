import streamlit as st
import json
import os
import pandas as pd
from pathlib import Path
from datetime import date
from sqlmodel import Session, select
from db.engine import engine, get_database_path
from services.people import get_all_people
from services.meetings import get_all_meetings
from utils.cache import clear_all_cache
from models.schema import Attendance, Passes, Turns


st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")

# Prediction parameters
st.header("🤖 Plackett-Luce Model Parameters")

st.markdown("""
Configure the Plackett-Luce model for speaker predictions:
- **Regularization Strength**: Controls model complexity (0.1-10.0, higher = simpler model)
- **Training History Days**: How many days of historical data to use for training
- **Feature Scaling**: Whether to normalize features before training
- **Max Iterations**: Maximum optimization iterations during training
""")

col1, col2 = st.columns(2)

with col1:
    regularization_strength = st.slider("Regularization Strength", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    min_history_days = st.slider("Training History Days", min_value=30, max_value=180, value=90, step=15)

with col2:
    feature_scaling = st.checkbox("Feature Scaling", value=True)
    max_iterations = st.slider("Max Training Iterations", min_value=100, max_value=2000, value=1000, step=100)

# Model management
st.subheader("Model Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Save Model Settings"):
        settings = {
            "regularization_strength": regularization_strength,
            "min_history_days": min_history_days,
            "feature_scaling": feature_scaling,
            "max_iterations": max_iterations,
            "updated": date.today().isoformat()
        }

        # Save to a settings file
        settings_path = Path("data") / "pl_model_settings.json"
        settings_path.parent.mkdir(exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)

        st.success("Model settings saved!")

with col2:
    if st.button("Retrain Model Now", type="primary"):
        from services.predict_pl import retrain_model

        with st.spinner("Retraining model with new settings..."):
            with Session(engine) as session:
                retrain_info = retrain_model(session, min_history_days)

        if retrain_info['retrain_success']:
            st.success("Model retrained successfully!")

            # Show feature importance if available
            if 'feature_importance' in retrain_info:
                st.write("**Feature Importance:**")
                for feature, importance in retrain_info['feature_importance'].items():
                    st.write(f"• {feature}: {importance:.3f}")
        else:
            st.error(f"Retraining failed: {retrain_info['retrain_message']}")

with col3:
    if st.button("Model Status", type="secondary"):
        from services.predict_pl import get_model_training_info

        with Session(engine) as session:
            model_info = get_model_training_info(session)

        if model_info['model_exists']:
            st.success("✅ Model is trained and ready")
            st.write(f"Training data points: {model_info['training_data_count']}")
            if model_info['last_modified']:
                st.write(f"Last updated: {model_info['last_modified']}")
        else:
            st.warning("⚠️ No trained model found")
            st.write(f"Available training data: {model_info['training_data_count']} points")

st.divider()

# Database management
st.header("🗄️ Database Management")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Database Info")
    db_path = get_database_path()
    st.write(f"**Location**: {db_path}")

    # Check if database exists and get size
    if os.path.exists(db_path):
        db_size = os.path.getsize(db_path)
        st.write(f"**Size**: {db_size:,} bytes ({db_size/1024:.1f} KB)")

        with Session(engine) as session:
            # Count records
            people_count = len(get_all_people(session))
            meetings_count = len(get_all_meetings(session))

            st.write(f"**People**: {people_count}")
            st.write(f"**Meetings**: {meetings_count}")
    else:
        st.write("Database not yet created")

with col2:
    st.subheader("Cache Management")
    st.write("Clear cached data to force refresh of all pages.")

    if st.button("Clear All Caches", type="secondary"):
        clear_all_cache()
        st.success("All caches cleared!")

st.divider()

# Data export/import
st.header("📤 Data Export")

st.write("Export all data as CSV files for backup or analysis in external tools.")

with Session(engine) as session:
    people = get_all_people(session)
    meetings = get_all_meetings(session)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("People Export")
        if st.button("Export People", key="export_people"):
            import pandas as pd
            from services.people import get_membership_periods

            people_data = []
            for person in people:
                periods = get_membership_periods(session, person.person_id)
                for period in periods:
                    people_data.append({
                        'person_id': person.person_id,
                        'name': person.name,
                        'team': person.team,
                        'role': person.role,
                        'active': person.active,
                        'start_date': period.start_date,
                        'end_date': period.end_date
                    })

            df = pd.DataFrame(people_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download people_with_periods.csv",
                data=csv,
                file_name=f"people_with_periods_{date.today()}.csv",
                mime="text/csv"
            )

    with col2:
        st.subheader("Meetings Export")
        if st.button("Export Meetings", key="export_meetings"):

            meetings_data = []
            for meeting in meetings:
                # Get attendance count
                attendance_stmt = select(Attendance).where(
                    Attendance.meeting_id == meeting.meeting_id,
                    Attendance.present == True
                )
                attendance_count = len(list(session.exec(attendance_stmt)))

                meetings_data.append({
                    'meeting_id': meeting.meeting_id,
                    'meeting_date': meeting.meeting_date,
                    'attendance_count': attendance_count,
                    'notes': meeting.notes
                })

            df = pd.DataFrame(meetings_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download meetings.csv",
                data=csv,
                file_name=f"meetings_{date.today()}.csv",
                mime="text/csv"
            )

    with col3:
        st.subheader("Full Export")
        if st.button("Export All Data", key="export_all"):

            # Create comprehensive export
            export_data = {
                'people': [],
                'meetings': [],
                'attendance': [],
                'passes': [],
                'turns': []
            }

            # People
            for person in people:
                export_data['people'].append({
                    'person_id': person.person_id,
                    'name': person.name,
                    'team': person.team,
                    'role': person.role,
                    'active': person.active
                })

            # Meetings
            for meeting in meetings:
                export_data['meetings'].append({
                    'meeting_id': meeting.meeting_id,
                    'meeting_date': meeting.meeting_date.isoformat(),
                    'notes': meeting.notes
                })

            # Attendance
            attendance_stmt = select(Attendance)
            for attendance in session.exec(attendance_stmt):
                export_data['attendance'].append({
                    'meeting_id': attendance.meeting_id,
                    'person_id': attendance.person_id,
                    'present': attendance.present
                })

            # Passes
            passes_stmt = select(Passes)
            for pass_record in session.exec(passes_stmt):
                export_data['passes'].append({
                    'pass_id': pass_record.pass_id,
                    'meeting_id': pass_record.meeting_id,
                    'from_person_id': pass_record.from_person_id,
                    'to_person_id': pass_record.to_person_id,
                    'seq': pass_record.seq
                })

            # Turns
            turns_stmt = select(Turns)
            for turn in session.exec(turns_stmt):
                export_data['turns'].append({
                    'turn_id': turn.turn_id,
                    'meeting_id': turn.meeting_id,
                    'seq': turn.seq,
                    'speaker_id': turn.speaker_id,
                    'duration_sec': turn.duration_sec
                })

            # Create JSON export
            json_export = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download full_export.json",
                data=json_export,
                file_name=f"standup_tracker_export_{date.today()}.json",
                mime="application/json"
            )

st.divider()

# Danger zone
st.header("⚠️ Danger Zone")

st.warning("These actions cannot be undone. Use with caution!")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Reset Database")
    st.write("This will delete all data and recreate empty tables.")

    if st.button("Reset Database", type="secondary"):
        confirm = st.checkbox("I understand this will delete all data", key="confirm_reset")
        if confirm and st.button("Confirm Reset", type="secondary"):
            try:
                # Delete the database file
                db_path = get_database_path()
                if os.path.exists(db_path):
                    os.remove(db_path)

                # Recreate tables
                from db.engine import create_all_tables
                create_all_tables()

                # Clear caches
                clear_all_cache()

                st.success("Database reset successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")

with col2:
    st.subheader("Load Demo Data")
    st.write("Populate the database with sample data for testing.")

    if st.button("Load Demo Data", type="secondary"):
        try:
            from db.seed import seed_demo_data
            seed_demo_data()
            clear_all_cache()
            st.success("Demo data loaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading demo data: {str(e)}")

# App info
st.divider()
st.header("ℹ️ About")

st.markdown("""
**Standup Tracker** v1.0

A local Streamlit application for tracking daily "popcorn" standups with AI-powered predictions.

**Features:**
- Track attendance and speaking order
- AI predictions using Plackett-Luce choice model
- Feature-based ranking with 6 key behavioral factors
- Historical analytics and model insights
- Team member management with history
- Data export capabilities

**Technology Stack:**
- Streamlit for the web interface
- SQLModel + SQLAlchemy for database ORM
- SQLite for data storage
- Python for backend logic

Built with ❤️ for agile teams everywhere.
""")

# Load existing settings if they exist
settings_path = Path("data") / "settings.json"
if settings_path.exists():
    try:
        with open(settings_path, "r") as f:
            saved_settings = json.load(f)
        st.write(f"**Last settings update**: {saved_settings.get('updated', 'Unknown')}")
    except Exception:
        pass