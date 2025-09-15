import argparse
import random
from datetime import date, timedelta
from sqlmodel import Session
from db.engine import engine, create_all_tables
from services.people import create_person
from services.meetings import create_meeting
from services.attendance import mark_attendance
from services.passes import create_pass


def seed_initial_people():
    """Seed initial team members."""
    with Session(engine) as session:
        people_data = [
            ("Alice Johnson", "Frontend", "Senior Developer"),
            ("Bob Smith", "Backend", "Tech Lead"),
            ("Carol Davis", "Frontend", "Developer"),
            ("David Wilson", "Backend", "Developer"),
            ("Eve Brown", "QA", "Test Engineer"),
            ("Frank Miller", "DevOps", "Senior Engineer"),
            ("Grace Lee", "Design", "UX Designer"),
            ("Henry Chen", "Backend", "Developer"),
            ("Iris Zhang", "Frontend", "Junior Developer"),
            ("Jack Taylor", "QA", "Test Engineer"),
        ]

        for name, team, role in people_data:
            try:
                create_person(session, name, team, role)
                print(f"Created person: {name}")
            except Exception as e:
                print(f"Error creating {name}: {e}")


def seed_demo_data():
    """Create comprehensive demo data with meetings, attendance, and passes."""
    # Ensure tables exist
    create_all_tables()

    with Session(engine) as session:
        # First, seed people if they don't exist
        from services.people import get_all_people
        existing_people = get_all_people(session)

        if len(existing_people) < 5:
            print("Seeding initial people...")
            seed_initial_people()
            # Refresh people list
            existing_people = get_all_people(session)

        people = existing_people[:8]  # Use first 8 people
        print(f"Using {len(people)} people for demo data")

        # Create 30 days of demo meetings (weekdays only)
        start_date = date.today() - timedelta(days=45)
        meetings_created = 0

        current_date = start_date
        while current_date <= date.today() and meetings_created < 30:
            # Only create meetings for weekdays (Monday=0 to Friday=4)
            if current_date.weekday() < 5:
                meeting = create_meeting(session, current_date, f"Daily standup for {current_date}")

                # Simulate realistic attendance (80-95% attendance rate)
                attendees = random.sample(people, random.randint(int(len(people) * 0.8), len(people)))

                # Mark attendance
                for person in people:
                    is_present = person in attendees
                    mark_attendance(session, meeting.meeting_id, person.person_id, is_present)

                # Create realistic pass patterns for attendees
                if len(attendees) >= 2:
                    # Simulate 3-8 passes per meeting
                    num_passes = random.randint(3, min(8, len(attendees) * 2))

                    for seq in range(1, num_passes + 1):
                        # Choose from/to with some realistic patterns
                        if seq == 1:
                            # First pass often from team lead or senior person
                            from_person = random.choice([p for p in attendees if "Lead" in (p.role or "") or "Senior" in (p.role or "")])
                            if not from_person:
                                from_person = random.choice(attendees)
                        else:
                            # Subsequent passes can be from anyone
                            from_person = random.choice(attendees)

                        # Choose recipient (not the same as sender)
                        available_recipients = [p for p in attendees if p.person_id != from_person.person_id]
                        to_person = random.choice(available_recipients)

                        create_pass(session, meeting.meeting_id, from_person.person_id, to_person.person_id, seq)

                meetings_created += 1
                print(f"Created meeting for {current_date} with {len(attendees)} attendees and {num_passes if len(attendees) >= 2 else 0} passes")

            current_date += timedelta(days=1)

        print(f"Demo data creation complete!")
        print(f"Created {meetings_created} meetings with realistic attendance and pass patterns")

        # Deactivate one person partway through to demonstrate membership periods
        if len(people) >= 5:
            person_to_deactivate = people[4]  # 5th person
            deactivation_date = date.today() - timedelta(days=15)

            from services.people import deactivate_person, reactivate_person
            deactivate_person(session, person_to_deactivate.person_id, deactivation_date)

            # Reactivate them a week later
            reactivation_date = deactivation_date + timedelta(days=7)
            reactivate_person(session, person_to_deactivate.person_id, reactivation_date)

            print(f"Deactivated and reactivated {person_to_deactivate.name} to demonstrate membership periods")


def main():
    parser = argparse.ArgumentParser(description="Seed database with initial or demo data")
    parser.add_argument("--demo", action="store_true", help="Create comprehensive demo data")
    parser.add_argument("--people-only", action="store_true", help="Create only initial people")

    args = parser.parse_args()

    create_all_tables()

    if args.demo:
        print("Creating demo data...")
        seed_demo_data()
    elif args.people_only:
        print("Creating initial people...")
        seed_initial_people()
    else:
        print("Creating initial people (use --demo for full demo data)...")
        seed_initial_people()

    print("Seeding complete!")


if __name__ == "__main__":
    main()