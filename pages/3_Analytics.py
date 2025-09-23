import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from sqlmodel import Session, select
from db.engine import engine
from models.schema import Attendance, Passes, People
from services.meetings import get_all_meetings
from services.people import get_all_people
from services.predict_pl import get_model_training_info, retrain_model
from services.features import build_feature_matrix, normalize_features


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

            team_bias_col, position_patterns_col = st.columns(2)

            with team_bias_col:
                st.subheader("Team Selection Bias")
                # Calculate team selection patterns
                same_team_passes = df_passes[df_passes['From_Team'] == df_passes['To_Team']]
                cross_team_passes = df_passes[df_passes['From_Team'] != df_passes['To_Team']]

                total_passes = len(df_passes)
                same_team_pct = len(same_team_passes) / total_passes * 100 if total_passes > 0 else 0
                cross_team_pct = len(cross_team_passes) / total_passes * 100 if total_passes > 0 else 0

                # Display as metrics
                st.metric("Same Team Selections", f"{same_team_pct:.1f}%")
                st.metric("Cross Team Selections", f"{cross_team_pct:.1f}%")

                # Team-to-team matrix if multiple teams
                teams = list(set(df_passes['From_Team'].unique()) | set(df_passes['To_Team'].unique()))
                if len(teams) > 1:
                    st.write("**Team → Team Selection Matrix:**")
                    for from_team in teams:
                        team_selections = df_passes[df_passes['From_Team'] == from_team]['To_Team'].value_counts()
                        if not team_selections.empty:
                            top_target = team_selections.index[0]
                            count = team_selections.iloc[0]
                            st.write(f"• {from_team} → {top_target}: {count} times")

            with position_patterns_col:
                st.subheader("Speaking Position Patterns")
                # Calculate speaking positions for each person
                position_data = []

                for meeting in recent_meetings:
                    meeting_passes = df_passes[df_passes['Date'] == meeting.meeting_date]
                    if len(meeting_passes) > 0:
                        # Sort by sequence (assuming chronological order in data)
                        meeting_passes = meeting_passes.reset_index(drop=True)
                        total_speakers = len(meeting_passes)

                        for idx, row in meeting_passes.iterrows():
                            position_pct = idx / (total_speakers - 1) if total_speakers > 1 else 0.5
                            position_data.append({
                                'Person': row['From'],
                                'Position': position_pct,
                                'Category': 'Early' if position_pct < 0.33 else 'Late' if position_pct > 0.67 else 'Middle'
                            })

                if position_data:
                    df_positions = pd.DataFrame(position_data)
                    position_summary = df_positions.groupby(['Person', 'Category']).size().unstack(fill_value=0)

                    # Show top early and late speakers
                    if 'Early' in position_summary.columns:
                        early_speakers = position_summary['Early'].sort_values(ascending=False).head(3)
                        st.write("**Early Speakers:**")
                        for person, count in early_speakers.items():
                            if count > 0:
                                st.write(f"• {person}: {count} times")

                    if 'Late' in position_summary.columns:
                        late_speakers = position_summary['Late'].sort_values(ascending=False).head(3)
                        st.write("**Late Speakers:**")
                        for person, count in late_speakers.items():
                            if count > 0:
                                st.write(f"• {person}: {count} times")

            # Selection consistency and prediction insights
            st.subheader("Selection Consistency & Prediction Insights")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Selection Consistency:**")
                # Calculate how often people pick the same recipients
                consistency_data = []
                for person in df_passes['From'].unique():
                    person_passes = df_passes[df_passes['From'] == person]
                    if len(person_passes) > 1:
                        # Calculate entropy/diversity of their selections
                        recipient_counts = person_passes['To'].value_counts()
                        total_passes = len(person_passes)

                        # Simple consistency metric: max_picks / total_picks
                        max_picks = recipient_counts.iloc[0] if len(recipient_counts) > 0 else 0
                        consistency = max_picks / total_passes

                        most_picked = recipient_counts.index[0] if len(recipient_counts) > 0 else "None"

                        consistency_data.append({
                            'From': person,
                            'Consistency': consistency,
                            'Favorite': most_picked,
                            'Total_Passes': total_passes
                        })

                if consistency_data:
                    df_consistency = pd.DataFrame(consistency_data)
                    df_consistency = df_consistency.sort_values('Consistency', ascending=False)

                    st.write("Most consistent pickers:")
                    for _, row in df_consistency.head(5).iterrows():
                        st.write(f"• **{row['From']}**: {row['Consistency']:.1%} → {row['Favorite']}")

            with col2:
                st.write("**Days Since Last Pick Patterns:**")
                # Analyze how spacing affects selections
                from datetime import datetime

                recent_picks = {}  # person -> {recipient: last_date}
                spacing_analysis = []

                # Sort passes by date to analyze temporal patterns
                df_passes_sorted = df_passes.sort_values('Date')

                for _, row in df_passes_sorted.iterrows():
                    from_person = row['From']
                    to_person = row['To']
                    pass_date = row['Date']

                    if from_person not in recent_picks:
                        recent_picks[from_person] = {}

                    # Check if this person has picked this recipient before
                    if to_person in recent_picks[from_person]:
                        last_pick_date = recent_picks[from_person][to_person]
                        days_between = (pass_date - last_pick_date).days
                        spacing_analysis.append({
                            'From': from_person,
                            'To': to_person,
                            'Days_Between': days_between
                        })

                    recent_picks[from_person][to_person] = pass_date

                if spacing_analysis:
                    df_spacing = pd.DataFrame(spacing_analysis)
                    avg_spacing = df_spacing['Days_Between'].mean()
                    median_spacing = df_spacing['Days_Between'].median()

                    st.metric("Average Days Between Repeat Picks", f"{avg_spacing:.1f}")
                    st.metric("Median Days Between Repeat Picks", f"{median_spacing:.1f}")

                    # Show quick/slow repeat patterns
                    quick_repeats = len(df_spacing[df_spacing['Days_Between'] <= 7])
                    total_repeats = len(df_spacing)
                    quick_pct = quick_repeats / total_repeats * 100 if total_repeats > 0 else 0

                    st.write(f"**Quick repeats** (≤7 days): {quick_pct:.1f}% of repeats")
                else:
                    st.info("Not enough repeat selections to analyze spacing patterns")

    # Model Performance and Features
    st.header("🤖 Plackett-Luce Model Analytics")

    model_info = get_model_training_info(session)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Model Status")
        if model_info['model_exists']:
            st.success("✅ Model trained and ready")
            if model_info['last_modified']:
                st.write(f"**Last updated**: {model_info['last_modified']}")
        else:
            st.warning("⚠️ No model found")

        st.metric("Training Data Points", model_info['training_data_count'])

    with col2:
        st.subheader("Model Management")
        if st.button("Retrain Model", type="secondary"):
            with st.spinner("Retraining model..."):
                retrain_info = retrain_model(session, min_history_days=90)

            if retrain_info['retrain_success']:
                st.success(retrain_info['retrain_message'])
                st.rerun()
            else:
                st.error(retrain_info['retrain_message'])

    with col3:
        st.subheader("Feature Importance")
        if model_info['model_exists']:
            try:
                from services.plackett_luce import PlackettLuceModel, get_model_path
                model = PlackettLuceModel()
                model.load_model(str(get_model_path()))

                importance = model.get_feature_importance()
                if importance:
                    # Create a sorted list of features by importance
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

                    for feature, score in sorted_features:
                        # Display with a simple bar representation
                        bar_length = int(score * 10) if score > 0 else 0
                        bar = "█" * bar_length
                        st.write(f"**{feature}**: {bar} ({score:.3f})")
                else:
                    st.info("Feature importance not available")
            except Exception as e:
                st.warning(f"Could not load feature importance: {str(e)}")
        else:
            st.info("Train model to see feature importance")

    # Model Performance Analysis
    if recent_meetings and model_info['training_data_count'] > 0:
        st.subheader("Model Performance Analysis")

        try:
            from services.predict_pl import collect_training_data
            from services.plackett_luce import PlackettLuceModel, get_model_path

            # Load model and get recent predictions vs reality
            model = PlackettLuceModel()
            model.load_model(str(get_model_path()))

            # Analyze recent training data for accuracy
            recent_training_data = collect_training_data(session, min_history_days=30)

            if len(recent_training_data) >= 5:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Prediction Accuracy Analysis:**")

                    correct_predictions = 0
                    top3_predictions = 0
                    total_predictions = 0

                    for instance in recent_training_data[-10:]:  # Last 10 decisions
                        features = instance['features']
                        actual_choice = instance['chosen_person_id']

                        if len(features) > 1:  # Need at least 2 candidates
                            probabilities = model.predict_proba(features)

                            if len(probabilities) > 0:
                                total_predictions += 1

                                # Get top prediction
                                best_idx = np.argmax(probabilities)
                                predicted_id = features[best_idx]['person_id']

                                if predicted_id == actual_choice:
                                    correct_predictions += 1

                                # Check if actual choice was in top 3
                                sorted_indices = np.argsort(probabilities)[::-1]
                                top3_ids = [features[i]['person_id'] for i in sorted_indices[:3]]
                                if actual_choice in top3_ids:
                                    top3_predictions += 1

                    if total_predictions > 0:
                        accuracy = correct_predictions / total_predictions * 100
                        top3_accuracy = top3_predictions / total_predictions * 100

                        st.metric("Top-1 Accuracy", f"{accuracy:.1f}%")
                        st.metric("Top-3 Accuracy", f"{top3_accuracy:.1f}%")
                        st.metric("Decisions Analyzed", total_predictions)
                    else:
                        st.info("Not enough recent data for accuracy analysis")

                with col2:
                    st.write("**Feature Distribution Analysis:**")

                    # Aggregate all recent features
                    all_features = []
                    for instance in recent_training_data:
                        all_features.extend(instance['features'])

                    if all_features:
                        all_features = normalize_features(all_features)
                        feature_names = [k for k in all_features[0].keys() if k != 'person_id']

                        # Calculate feature statistics
                        for feature_name in feature_names:
                            values = [f[feature_name] for f in all_features if feature_name in f]
                            if values:
                                avg_val = np.mean(values)
                                std_val = np.std(values)
                                st.write(f"• **{feature_name}**: μ={avg_val:.3f}, σ={std_val:.3f}")

            else:
                st.info("Need more historical data (≥5 decisions) for performance analysis")

        except Exception as e:
            st.warning(f"Could not analyze model performance: {str(e)}")

    # Feature Correlation Analysis
    if recent_meetings and model_info['training_data_count'] > 10:
        st.subheader("Feature Correlation Insights")

        try:
            # Get comprehensive feature data
            all_features_data = []
            for meeting in recent_meetings[-5:]:  # Last 5 meetings
                from services.attendance import get_attendees_for_meeting
                attendees = get_attendees_for_meeting(session, meeting.meeting_id)

                if len(attendees) > 2:
                    for speaker in attendees:
                        candidate_ids = [p.person_id for p in attendees if p.person_id != speaker.person_id]
                        features = build_feature_matrix(
                            session=session,
                            meeting_date=meeting.meeting_date,
                            current_speaker_id=speaker.person_id,
                            candidate_ids=candidate_ids[:3]  # Limit for performance
                        )
                        if features:
                            all_features_data.extend(normalize_features(features))

            if len(all_features_data) > 20:
                # Convert to DataFrame for correlation analysis
                feature_df = pd.DataFrame(all_features_data)
                numeric_cols = [col for col in feature_df.columns if col != 'person_id']

                if len(numeric_cols) > 1:
                    correlation_matrix = feature_df[numeric_cols].corr()

                    st.write("**Key Feature Correlations:**")

                    # Find strongest correlations (excluding self-correlation)
                    strong_correlations = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_val = correlation_matrix.iloc[i, j]
                            if abs(corr_val) > 0.3:  # Moderate correlation threshold
                                strong_correlations.append({
                                    'feature1': correlation_matrix.columns[i],
                                    'feature2': correlation_matrix.columns[j],
                                    'correlation': corr_val
                                })

                    # Sort by absolute correlation
                    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

                    for corr in strong_correlations[:5]:  # Top 5 correlations
                        direction = "positively" if corr['correlation'] > 0 else "negatively"
                        st.write(f"• **{corr['feature1']}** & **{corr['feature2']}** are {direction} correlated ({corr['correlation']:.3f})")

                    if not strong_correlations:
                        st.info("No strong feature correlations found (>0.3)")

        except Exception as e:
            st.warning(f"Could not analyze feature correlations: {str(e)}")

    st.divider()

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
            # Enhanced individual statistics
            st.subheader(f"Behavioral Profile for {selected_person.name}")

            # Calculate advanced individual metrics
            person_passes_from = df_passes[df_passes['From'] == selected_person.name] if 'df_passes' in locals() else pd.DataFrame()
            person_passes_to = df_passes[df_passes['To'] == selected_person.name] if 'df_passes' in locals() else pd.DataFrame()

            if not person_passes_from.empty:
                # Team bias metric
                same_team_selections = len(person_passes_from[person_passes_from['From_Team'] == person_passes_from['To_Team']])
                total_selections = len(person_passes_from)
                team_bias_pct = same_team_selections / total_selections * 100 if total_selections > 0 else 0

                st.metric("Team Selection Bias", f"{team_bias_pct:.1f}%",
                         help="Percentage of times they pick someone from their own team")

                # Selection diversity
                unique_recipients = person_passes_from['To'].nunique()
                total_passes = len(person_passes_from)
                diversity_score = unique_recipients / total_passes if total_passes > 0 else 0

                st.metric("Selection Diversity", f"{diversity_score:.2f}",
                         help="How varied their recipient choices are (1.0 = always picks different people)")

                # Most frequent recipient
                if not person_passes_from.empty:
                    most_picked = person_passes_from['To'].value_counts()
                    if len(most_picked) > 0:
                        favorite_recipient = most_picked.index[0]
                        favorite_count = most_picked.iloc[0]
                        favorite_pct = favorite_count / total_passes * 100

                        st.write(f"**Most often passes to:**")
                        st.write(f"• {favorite_recipient}: {favorite_count} times ({favorite_pct:.1f}%)")

                        # Show top 3 if there are more recipients
                        if len(most_picked) > 1:
                            st.write("**Other frequent recipients:**")
                            for recipient, count in most_picked.iloc[1:4].items():
                                pct = count / total_passes * 100
                                st.write(f"• {recipient}: {count} times ({pct:.1f}%)")

            else:
                st.info("No pass data available for this person in the selected time period")

            # Speaking position analysis for this person
            if 'position_data' in locals() and position_data:
                person_positions = [p for p in position_data if p['Person'] == selected_person.name]
                if person_positions:
                    positions = [p['Position'] for p in person_positions]
                    avg_position = sum(positions) / len(positions)

                    # Convert to descriptive text
                    if avg_position < 0.33:
                        position_style = "Early Speaker"
                        position_color = "normal"
                    elif avg_position > 0.67:
                        position_style = "Late Speaker"
                        position_color = "inverse"
                    else:
                        position_style = "Mid Speaker"
                        position_color = "off"

                    st.metric("Speaking Position Style", position_style,
                             help=f"Average position: {avg_position:.2f} (0=always first, 1=always last)")

            # Add "being selected" patterns
            if not person_passes_to.empty:
                st.divider()
                st.write("**Being Selected Patterns:**")

                # Who picks them most often
                most_received_from = person_passes_to['From'].value_counts()
                if len(most_received_from) > 0:
                    top_picker = most_received_from.index[0]
                    pick_count = most_received_from.iloc[0]
                    total_received = len(person_passes_to)
                    pick_pct = pick_count / total_received * 100

                    st.write(f"**Most often selected by:**")
                    st.write(f"• {top_picker}: {pick_count} times ({pick_pct:.1f}%)")

                # Cross-team selection frequency
                cross_team_received = len(person_passes_to[person_passes_to['From_Team'] != person_passes_to['To_Team']])
                total_received = len(person_passes_to)
                cross_team_received_pct = cross_team_received / total_received * 100 if total_received > 0 else 0

                st.write(f"**Selected by other teams:** {cross_team_received_pct:.1f}% of the time")

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