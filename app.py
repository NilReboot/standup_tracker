import streamlit as st
from datetime import date, timedelta
from sqlmodel import Session, select
from db.engine import create_all_tables, engine
from services.people import get_all_people
from services.meetings import get_all_meetings
from models.schema import Passes


def main():
    st.set_page_config(
        page_title="Standup Tracker",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    create_all_tables()

    st.title("🎙️ Standup Tracker")
    st.markdown("""
    Track daily "popcorn" standups and predict likely next speakers using AI.

    **Features:**
    - 📅 **Today**: Record attendance and passes for today's standup
    - 👥 **People**: Manage team roster and membership periods
    - 📈 **Analytics**: View attendance trends and pass patterns
    - ⚙️ **Settings**: Configure prediction parameters and export data

    Navigate using the sidebar to get started!
    """)

    # Show quick stats
    col1, col2, col3 = st.columns(3)

    with Session(engine) as session:
        # Get active team members count
        all_people = get_all_people(session)
        active_count = len([p for p in all_people if p.active])

        # Get recent meetings count (last 30 days)
        cutoff_date = date.today() - timedelta(days=30)
        all_meetings = get_all_meetings(session)
        recent_meetings = [m for m in all_meetings if m.meeting_date >= cutoff_date]

        # Get total passes count
        passes_stmt = select(Passes)
        total_passes = len(list(session.exec(passes_stmt)))

    with col1:
        st.metric("Active Team Members", active_count)

    with col2:
        st.metric("Recent Meetings (30d)", len(recent_meetings))

    with col3:
        st.metric("Total Passes Recorded", total_passes)


if __name__ == "__main__":
    main()