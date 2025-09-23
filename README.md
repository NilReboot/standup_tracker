# 🎙️ Standup Tracker

A local Streamlit web app to track daily "popcorn" standups (≈20 people, one standup per weekday) and suggest the top-3 likely "next person" passes in real time using AI.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   uv add streamlit sqlmodel sqlalchemy pandas numpy alembic pydantic-settings
   ```

2. **Run the application:**
   ```bash
   uv run streamlit run app.py
   ```

3. **Load demo data (optional):**
   ```bash
   uv run python -m db.seed --demo
   ```

4. **Open your browser** to the URL shown (typically http://localhost:8501)

## 🏗️ Architecture

### Technology Stack
- **Database**: SQLite (single file in ./data/standups.db)
- **ORM**: SQLModel (on SQLAlchemy 2.x)
- **UI**: Streamlit
- **Package/venv manager**: uv
- **OS Support**: macOS/Linux/Windows, Python 3.10+

### Dependencies
- **Core**: `streamlit sqlmodel sqlalchemy pandas numpy`
- **Optional**: `alembic` (migrations), `pydantic-settings` (config)

## 📁 Project Structure

```
.
├── app.py                       # Streamlit entrypoint
├── README.md
├── data/
│   └── standups.db              # SQLite database (created on first run)
├── models/
│   ├── __init__.py
│   └── schema.py                # SQLModel models + table definitions
├── db/
│   ├── __init__.py
│   ├── engine.py                # Database engine/session helpers
│   └── seed.py                  # Data seeding script with CLI
├── services/
│   ├── __init__.py
│   ├── attendance.py            # CRUD for attendance tracking
│   ├── meetings.py              # CRUD for meetings
│   ├── turns.py                 # CRUD for speaking turns
│   ├── passes.py                # CRUD for pass tracking
│   ├── people.py                # CRUD for people & membership periods
│   ├── predict_pl.py            # AI prediction using Plackett-Luce model
│   ├── features.py              # Feature engineering for ML model
│   └── plackett_luce.py         # Plackett-Luce model implementation
├── pages/
│   ├── 1_Today.py               # Daily standup capture
│   ├── 2_People.py              # Team roster management
│   ├── 3_Analytics.py           # Analytics & insights dashboard
│   └── 4_Settings.py            # Configuration & data export
└── utils/
    ├── __init__.py
    └── cache.py                 # Streamlit caching utilities
```

## ✨ Features

- **📅 Daily Tracking**: Record attendance and speaking order for daily standups
- **🤖 AI Predictions**: Smart suggestions for next speakers using Markov chain analysis
- **👥 Team Management**: Full roster management with historical membership tracking
- **📊 Analytics**: Comprehensive insights into attendance patterns and speaking trends
- **💾 Data Export**: Export all data as CSV/JSON for external analysis
- **⚙️ Configurable**: Adjust prediction parameters and manage application settings

## Data model (SQLModel)

### People
- `person_id`: int PK
- `name`: str UNIQUE
- `active`: bool (convenience flag for *current* status only)
- `team`: Optional[str]
- `role`: Optional[str]

### MembershipPeriods
- `period_id`: int PK
- `person_id`: FK → People.person_id
- `start_date`: date (inclusive)
- `end_date`: Optional[date] (NULL = still active)
- Constraint: no overlapping periods for a person
- Use this to track historical activity accurately, including re-joins.

### Meetings
- `meeting_id`: int PK
- `meeting_date`: date UNIQUE
- `notes`: Optional[str]

### Attendance
- `meeting_id`: FK → Meetings.meeting_id
- `person_id`: FK → People.person_id
- `present`: bool
- Composite PK: (meeting_id, person_id)

### Turns
- `turn_id`: int PK
- `meeting_id`: FK
- `seq`: int (CHECK seq >= 1), unique per meeting
- `speaker_id`: FK → People.person_id
- `duration_sec`: Optional[int]

### Passes
- `pass_id`: int PK
- `meeting_id`: FK
- `from_person_id`: FK → People.person_id
- `to_person_id`: FK → People.person_id
- `seq`: int (order in meeting)

## Membership logic
- When adding a person: create a MembershipPeriod with `start_date=today`, `end_date=NULL`, set People.active=True.
- When someone leaves: close their latest MembershipPeriod (`end_date=today`), set People.active=False.
- If they return later: insert a new MembershipPeriod row with a new `start_date`, leave old periods intact, set People.active=True.
- Do **not** delete people; history in Attendance/Passes/Turns must remain valid.

## Prediction approach (services/predict_pl.py)
- Uses Plackett-Luce model with feature engineering for more sophisticated predictions
- Features include: days since last picked, team relationships, attendance patterns, speaking history
- Trained on historical pass data with model persistence and automatic retraining
- Candidate set for predictions includes only people:
  - Who are present at today's meeting (`Attendance.present=True`)
  - AND whose membership_period covers today's date (`start_date <= meeting_date <= end_date OR end_date IS NULL`)
  - AND who haven't already spoken in the current meeting

## Streamlit UX
### People page
- CRUD roster.
- Add ability to deactivate/reactivate people with date pickers:
  - On deactivate: close open MembershipPeriod (set end_date), flip People.active=False.
  - On reactivate: add new MembershipPeriod, flip People.active=True.
- Show both current “active” and historical periods.

### Today page
- When selecting attendees, default to people who are active on the meeting_date (based on MembershipPeriods).
- Pass capture and predictions use only attendees who are active on that date.

### Analytics page
- Attendance rates should respect membership periods (e.g., don’t count days before someone joined).
- Optional toggle: include/exclude inactive people in historical charts.

### Settings page
- Parameters for prediction (alpha, lambda_decay, min_history_days).
- Export CSVs of people, membership_periods, meetings, attendance, passes, turns.

## Database engine and session
- Same as before: SQLite engine in ./data/standups.db, `PRAGMA foreign_keys=ON`.

## Seed script
- Populate initial roster and open MembershipPeriods.
- Demo mode: create fake meetings, attendance, passes covering multiple dates, with some people deactivated/reactivated.

## Minimal working code requirements
- Must support:
  - Creating meetings
  - Marking attendance
  - Recording passes
  - Predicting next speaker (respecting membership_periods)
  - Managing people + deactivation/reactivation with history preserved

## 🔧 Additional Commands

### Seeding Data
```bash
# Create initial team members only
uv run python -m db.seed --people-only

# Create comprehensive demo data (30 days of meetings with realistic patterns)
uv run python -m db.seed --demo
```

### Development
```bash
# Run the app in development mode
uv run streamlit run app.py

# Clear Python cache files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

## 🎯 Usage Guide

### Getting Started
1. **Add Team Members**: Use the "People" page to add your team members with their roles and teams
2. **Record Daily Standups**: Use the "Today" page to mark attendance and record speaking passes
3. **View Insights**: Check the "Analytics" page for attendance trends and speaking patterns
4. **Configure Settings**: Adjust AI prediction parameters in the "Settings" page

### AI Predictions
The app uses a Plackett-Luce choice model with:
- **Feature Engineering**: Multi-dimensional features including temporal, social, and behavioral patterns
- **Model Training**: Learns from historical pass data with automatic retraining capabilities
- **Membership Awareness**: Only predicts active team members present at the meeting
- **Context Awareness**: Considers who has already spoken in the current meeting

### Data Management
- **Historical Accuracy**: Team member deactivations preserve all historical data
- **Membership Periods**: Track when people join, leave, and rejoin the team
- **Export Capabilities**: Download data as CSV/JSON for external analysis

## 📊 Quality Features

- ✅ **Strong Typing**: Full type hints throughout the codebase
- ✅ **Data Integrity**: Foreign key constraints and validation
- ✅ **Historical Preservation**: No data loss when deactivating people
- ✅ **Responsive UI**: Clear indication of active vs. inactive members
- ✅ **Cross-Platform**: Works on macOS, Linux, and Windows
- ✅ **Zero Configuration**: SQLite database created automatically

## 🤝 Contributing

This is a local application designed for individual teams. To customize:
1. Modify prediction algorithms in `services/predict_pl.py` or `services/plackett_luce.py`
2. Add new features in `services/features.py`
3. Add new analytics in `pages/3_Analytics.py`
4. Extend the data model in `models/schema.py`
5. Add new UI pages following the existing pattern

