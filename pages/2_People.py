import streamlit as st
from datetime import date
from sqlmodel import Session
from db.engine import engine
from services.people import (
    create_person, get_all_people, update_person,
    deactivate_person, reactivate_person, get_membership_periods
)


st.set_page_config(page_title="People Management", page_icon="👥", layout="wide")

st.title("👥 People Management")

with Session(engine) as session:
    # Add new person section
    st.header("➕ Add New Person")

    col1, col2, col3 = st.columns(3)

    with col1:
        new_name = st.text_input("Name", key="new_person_name")
    with col2:
        new_team = st.text_input("Team (optional)", key="new_person_team")
    with col3:
        new_role = st.text_input("Role (optional)", key="new_person_role")

    if st.button("Add Person", type="primary"):
        if new_name.strip():
            try:
                person = create_person(session, new_name.strip(), new_team.strip() or None, new_role.strip() or None)
                st.success(f"Added {person.name} to the team!")
                st.rerun()
            except Exception as e:
                st.error(f"Error adding person: {str(e)}")
        else:
            st.error("Name is required")

    st.divider()

    # Current roster
    st.header("👫 Current Roster")

    # Filter options
    col1, col2 = st.columns([1, 3])

    with col1:
        show_inactive = st.checkbox("Show inactive members", value=False)

    # Get all people
    all_people = get_all_people(session, active_only=False)

    if show_inactive:
        display_people = all_people
    else:
        display_people = [p for p in all_people if p.active]

    if not display_people:
        st.info("No team members found. Add some people to get started!")
    else:
        # Display people in a table format
        for person in display_people:
            with st.expander(f"{'✅' if person.active else '❌'} {person.name} ({person.team or 'No team'})"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Edit person details
                    st.subheader("Edit Details")

                    new_person_name = st.text_input("Name", value=person.name, key=f"edit_name_{person.person_id}")
                    new_person_team = st.text_input("Team", value=person.team or "", key=f"edit_team_{person.person_id}")
                    new_person_role = st.text_input("Role", value=person.role or "", key=f"edit_role_{person.person_id}")

                    col_save, col_status = st.columns([1, 1])

                    with col_save:
                        if st.button("Save Changes", key=f"save_{person.person_id}"):
                            update_person(
                                session,
                                person.person_id,
                                new_person_name.strip(),
                                new_person_team.strip() or None,
                                new_person_role.strip() or None
                            )
                            st.success("Updated!")
                            st.rerun()

                    with col_status:
                        if person.active:
                            if st.button("Deactivate", key=f"deactivate_{person.person_id}", type="secondary"):
                                deactivation_date = st.date_input("Deactivation date", value=date.today(), key=f"deactivate_date_{person.person_id}")
                                if st.button("Confirm Deactivation", key=f"confirm_deactivate_{person.person_id}"):
                                    deactivate_person(session, person.person_id, deactivation_date)
                                    st.success(f"Deactivated {person.name}")
                                    st.rerun()
                        else:
                            if st.button("Reactivate", key=f"reactivate_{person.person_id}", type="secondary"):
                                reactivation_date = st.date_input("Reactivation date", value=date.today(), key=f"reactivate_date_{person.person_id}")
                                if st.button("Confirm Reactivation", key=f"confirm_reactivate_{person.person_id}"):
                                    reactivate_person(session, person.person_id, reactivation_date)
                                    st.success(f"Reactivated {person.name}")
                                    st.rerun()

                with col2:
                    # Show membership periods
                    st.subheader("Membership History")
                    periods = get_membership_periods(session, person.person_id)

                    if periods:
                        for period in periods:
                            end_str = period.end_date.strftime("%Y-%m-%d") if period.end_date else "Present"
                            st.write(f"📅 {period.start_date.strftime('%Y-%m-%d')} → {end_str}")
                    else:
                        st.info("No membership periods found")

    # Team statistics
    st.header("📊 Team Statistics")

    active_count = len([p for p in all_people if p.active])
    total_count = len(all_people)
    teams = {}

    for person in all_people:
        team = person.team or "No team"
        if team not in teams:
            teams[team] = {"active": 0, "total": 0}
        teams[team]["total"] += 1
        if person.active:
            teams[team]["active"] += 1

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Active Members", active_count)

    with col2:
        st.metric("Total Members", total_count)

    with col3:
        st.metric("Teams", len(teams))

    # Team breakdown
    if teams:
        st.subheader("Team Breakdown")
        for team, counts in teams.items():
            st.write(f"**{team}**: {counts['active']} active / {counts['total']} total")