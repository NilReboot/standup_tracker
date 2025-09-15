import streamlit as st
import pandas as pd
from datetime import date, timedelta
from sqlmodel import Session, select
from db.engine import engine
from models.schema import Attendance, Passes, People
from services.meetings import get_all_meetings
from services.people import get_all_people
from services.predict import get_pass_statistics


st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

st.title("📈 Analytics Dashboard")

with Session(engine) as session:
    # Time range selector
    time_per_col, inact_mems_col, col3 = st.columns([1, 1, 2])

    with time_per_col:
        days_back = st.selectbox("Time period", [7, 14, 30, 60, 90], index=2, key="days_back")

    with inact_mems_col:
        include_inactive = st.checkbox("Include inactive members", value=False)

    cutoff_date = date.today() - timedelta(days=days_back)

    # Get data for the selected period
    meetings = get_all_meetings(session)
    recent_meetings = [m for m in meetings if m.meeting_date >= cutoff_date]

    all_people = get_all_people(session, active_only=not include_inactive)

    if not recent_meetings:
        st.warning(f"No meetings found in the last {days_back} days.")
        st.stop()

    # Overview metrics
    st.header("📊 Overview")

    recent_mtgs_col, total_passes_col, avg_attend_col, active_mems_col = st.columns(4)

    with recent_mtgs_col:
        st.metric("Meetings", len(recent_meetings))

    with total_passes_col:
        # Calculate total passes in period
        total_passes = 0
        for meeting in recent_meetings:
            passes_stmt = select(Passes).where(Passes.meeting_id == meeting.meeting_id)
            passes_count = len(list(session.exec(passes_stmt)))
            total_passes += passes_count
        st.metric("Total Passes", total_passes)

    with avg_attend_col:
        # Calculate average attendance
        total_attendance = 0
        for meeting in recent_meetings:
            attendance_stmt = select(Attendance).where(
                Attendance.meeting_id == meeting.meeting_id,
                Attendance.present == True
            )
            attendance_count = len(list(session.exec(attendance_stmt)))
            total_attendance += attendance_count

        avg_attendance = total_attendance / len(recent_meetings) if recent_meetings else 0
        st.metric("Avg Attendance", f"{avg_attendance:.1f}")

    with active_mems_col:
        st.metric("Active Members", len([p for p in all_people if p.active]))

    # Attendance analysis
    st.header("👥 Attendance Analysis")

    if recent_meetings:
        # Build attendance data
        attendance_data = []

        for meeting in recent_meetings:
            attendance_stmt = select(Attendance, People).join(People).where(
                Attendance.meeting_id == meeting.meeting_id
            )
            attendances = session.exec(attendance_stmt).all()

            for attendance, person in attendances:
                if not include_inactive and not person.active:
                    continue

                attendance_data.append({
                    'Date': meeting.meeting_date,
                    'Person': person.name,
                    'Team': person.team or 'No team',
                    'Present': attendance.present,
                    'Person_ID': person.person_id
                })

        if attendance_data:
            df_attendance = pd.DataFrame(attendance_data)

            # Attendance by person
            attend_rate_col, daily_attend_col = st.columns(2)

            with attend_rate_col:
                st.subheader("Attendance Rate by Person")
                person_attendance = df_attendance.groupby('Person').agg({
                    'Present': ['sum', 'count']
                }).round(2)
                person_attendance.columns = ['Present', 'Total']
                person_attendance['Rate'] = (person_attendance['Present'] / person_attendance['Total'] * 100).round(1)
                person_attendance = person_attendance.sort_values('Rate', ascending=False)

                st.dataframe(person_attendance, width='stretch')

            with daily_attend_col:
                st.subheader("Daily Attendance")
                daily_attendance = df_attendance[df_attendance['Present']].groupby('Date').size()
                st.line_chart(daily_attendance)

    # Pass analysis
    st.header("🎯 Pass Analysis")

    if recent_meetings:
        # Build pass data
        pass_data = []

        for meeting in recent_meetings:
            passes_stmt = select(Passes).where(Passes.meeting_id == meeting.meeting_id)
            passes = session.exec(passes_stmt).all()

            for pass_record in passes:
                from_person = session.get(People, pass_record.from_person_id)
                to_person = session.get(People, pass_record.to_person_id)

                if not include_inactive and (not from_person.active or not to_person.active):
                    continue

                pass_data.append({
                    'Date': meeting.meeting_date,
                    'From': from_person.name,
                    'To': to_person.name,
                    'From_Team': from_person.team or 'No team',
                    'To_Team': to_person.team or 'No team',
                    'From_ID': from_person.person_id,
                    'To_ID': to_person.person_id
                })

        if pass_data:
            df_passes = pd.DataFrame(pass_data)

            act_pass_col, freq_recip_col = st.columns(2)

            with act_pass_col:
                st.subheader("Most Active Passers")
                from_counts = df_passes['From'].value_counts().head(10)
                st.bar_chart(from_counts)

            with freq_recip_col:
                st.subheader("Most Frequent Recipients")
                to_counts = df_passes['To'].value_counts().head(10)
                st.bar_chart(to_counts)

            # Pass patterns
            st.subheader("Pass Patterns")
            col1, col2 = st.columns(2)

            with col1:
                # Most common pass pairs
                st.write("**Most Common Pass Pairs:**")
                pass_pairs = df_passes.groupby(['From', 'To']).size().sort_values(ascending=False).head(10)
                for (from_person, to_person), count in pass_pairs.items():
                    st.write(f"{from_person} → {to_person}: {count} times")

            with col2:
                # Team-to-team passes
                st.write("**Cross-Team Passes:**")
                cross_team = df_passes[df_passes['From_Team'] != df_passes['To_Team']]
                team_passes = cross_team.groupby(['From_Team', 'To_Team']).size().sort_values(ascending=False).head(5)
                for (from_team, to_team), count in team_passes.items():
                    st.write(f"{from_team} → {to_team}: {count} times")

    # Individual statistics
    st.header("👤 Individual Statistics")

    selected_person = st.selectbox(
        "Select person for detailed stats:",
        options=all_people,
        format_func=lambda x: f"{x.name} ({x.team or 'No team'})",
        key="person_stats_select"
    )

    if selected_person:
        person_stats_col, attend_person_col = st.columns(2)

        with person_stats_col:
            # Pass statistics for selected person
            stats = get_pass_statistics(session, selected_person.person_id, days_back)

            st.subheader(f"Pass Stats for {selected_person.name}")
            st.metric("Passes Given", stats['total_passes_given'])
            st.metric("Passes Received", stats['total_passes_received'])

            if stats['most_passed_to']:
                st.write("**Most often passes to:**")
                for person_id, count in stats['most_passed_to']:
                    person = session.get(People, person_id)
                    st.write(f"• {person.name}: {count} times")

        with attend_person_col:
            # Attendance for selected person
            person_meetings = []
            for meeting in recent_meetings:
                attendance_stmt = select(Attendance).where(
                    Attendance.meeting_id == meeting.meeting_id,
                    Attendance.person_id == selected_person.person_id
                )
                attendance = session.exec(attendance_stmt).first()
                person_meetings.append({
                    'Date': meeting.meeting_date,
                    'Present': attendance.present if attendance else False
                })

            if person_meetings:
                df_person = pd.DataFrame(person_meetings)
                attendance_rate = (df_person['Present'].sum() / len(df_person) * 100)

                st.subheader(f"Attendance for {selected_person.name}")
                st.metric("Attendance Rate", f"{attendance_rate:.1f}%")

                # Show recent attendance
                st.write("**Recent Attendance:**")
                recent_attendance = df_person.tail(10)
                for _, row in recent_attendance.iterrows():
                    status = "✅" if row['Present'] else "❌"
                    st.write(f"{status} {row['Date'].strftime('%Y-%m-%d')}")

    # Export data
    st.header("💾 Export Data")

    exp_attend_col, exp_passes_col, exp_people_col = st.columns(3)

    with exp_attend_col:
        if st.button("Export Attendance"):
            if 'df_attendance' in locals():
                csv = df_attendance.to_csv(index=False)
                st.download_button(
                    label="Download Attendance CSV",
                    data=csv,
                    file_name=f"attendance_{date.today()}.csv",
                    mime="text/csv"
                )

    with exp_passes_col:
        if st.button("Export Passes"):
            if 'df_passes' in locals():
                csv = df_passes.to_csv(index=False)
                st.download_button(
                    label="Download Passes CSV",
                    data=csv,
                    file_name=f"passes_{date.today()}.csv",
                    mime="text/csv"
                )

    with exp_people_col:
        if st.button("Export People"):
            people_data = []
            for person in all_people:
                people_data.append({
                    'Name': person.name,
                    'Team': person.team,
                    'Role': person.role,
                    'Active': person.active
                })
            df_people = pd.DataFrame(people_data)
            csv = df_people.to_csv(index=False)
            st.download_button(
                label="Download People CSV",
                data=csv,
                file_name=f"people_{date.today()}.csv",
                mime="text/csv"
            )