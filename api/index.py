from flask import Flask, jsonify, request, send_from_directory
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import os

app = Flask(__name__, static_folder='frontend')

# Load the demand data

BASE_DIR = Path(__file__).resolve().parent


DATA_FILE = BASE_DIR / 'data' / 'master_campus_demand_W1_to_W16.csv'
df = pd.read_csv(DATA_FILE)

# ── Academic Calendar Configuration ──
# Add new semesters here. Each entry:
#   id: unique key
#   type: 'Fall' or 'Spring' (affects spring break handling)
#   start_date: first day of Week 1 (YYYY-MM-DD)
#   weeks: number of data weeks (default 16)
#   has_spring_break: True for Spring semesters with a break week
#   spring_break_calendar_week: which calendar week (1-based) is spring break
SEMESTERS = [
    {'id': 'sp2026', 'type': 'Spring', 'start_date': '2026-01-20', 'weeks': 16, 'has_spring_break': True,  'spring_break_calendar_week': 9},
    {'id': 'fa2026', 'type': 'Fall',   'start_date': '2026-08-24', 'weeks': 16, 'has_spring_break': False},
    {'id': 'sp2027', 'type': 'Spring', 'start_date': '2027-01-19', 'weeks': 16, 'has_spring_break': True,  'spring_break_calendar_week': 9},
]


def get_current_academic_time(now=None):
    """Determine current academic semester, week, day, hour from a datetime."""
    if now is None:
        now = datetime.now()

    # Find the most recent (or current) semester
    current_semester = None
    for sem in SEMESTERS:
        start = datetime.strptime(sem['start_date'], '%Y-%m-%d')
        end = start + timedelta(weeks=sem['weeks'])
        if sem.get('has_spring_break'):
            end += timedelta(weeks=1)  # spring break adds a week
        if start <= now < end:
            current_semester = sem
            break

    # If not currently in any semester, return holiday state
    if current_semester is None:
        return {
            'holiday': True,
            'semester_type': 'Holiday',
            'week': 0,
            'day': 'Holiday',
            'hour': now.hour,
            'datetime': now.strftime('%Y-%m-%d %H:%M:%S')
        }

    start = datetime.strptime(current_semester['start_date'], '%Y-%m-%d')
    days_since_start = (now - start).days

    # Calendar week (1-based) within the semester
    calendar_week = (days_since_start // 7) + 1
    calendar_week = max(1, min(calendar_week, current_semester['weeks'] + (1 if current_semester.get('has_spring_break') else 0)))

    # Convert calendar week to data week (accounting for spring break)
    data_week = calendar_week
    if current_semester.get('has_spring_break'):
        sbw = current_semester.get('spring_break_calendar_week', 9)
        if calendar_week >= sbw:
            data_week = calendar_week - 1  # Skip the break week in data
        # If we landed exactly on spring break week, move to next week
        if calendar_week == sbw:
            data_week = sbw  # Map to the week after spring break (data week 9)

    data_week = max(1, min(data_week, current_semester['weeks']))

    # Day of week (Monday=0 ... Sunday=6)
    day_of_week = now.weekday()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name = days[day_of_week]

    hour = now.hour

    return {
        'semester_id': current_semester['id'],
        'semester_type': current_semester['type'],
        'semester_start': current_semester['start_date'],
        'calendar_week': calendar_week,
        'week': data_week,
        'day': day_name,
        'hour': hour,
        'datetime': now.strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/demand', methods=['GET'])
def get_demand():
    week = request.args.get('week', default=1, type=int)
    day = request.args.get('day', default='Monday', type=str)
    hour = request.args.get('hour', default=0, type=int)

    filtered_df = df[(df['Week'] == week) & (df['Day'] == day) & (df['Hour'] == hour)]
    results = filtered_df[['Building', 'Demand_Prediction']].to_dict('records')
    
    return jsonify({
        'week': week, 'day': day, 'hour': hour, 'predictions': results
    })

@app.route('/api/buildings', methods=['GET'])
def get_buildings():
    buildings = df['Building'].unique().tolist()
    return jsonify({'buildings': buildings})

@app.route('/api/all_days', methods=['GET'])
def get_all_days():
    """Returns demand for all buildings across all days of a week at a given hour."""
    week = request.args.get('week', default=1, type=int)
    hour = request.args.get('hour', default=12, type=int)

    filtered_df = df[(df['Week'] == week) & (df['Hour'] == hour)]
    results = filtered_df[['Building', 'Day', 'Demand_Prediction']].to_dict('records')

    return jsonify({
        'week': week,
        'hour': hour,
        'predictions': results
    })

@app.route('/api/calendar/<building>', methods=['GET'])
def get_calendar(building):
    # Returns demand for a building across a specific week
    week = request.args.get('week', default=1, type=int)
    filtered_df = df[(df['Building'] == building) & (df['Week'] == week)]

    # Group by day and hour
    data = filtered_df.groupby(['Day', 'Hour'])['Demand_Prediction'].mean().unstack().to_dict('index')

    return jsonify({
        'building': building,
        'week': week,
        'calendar': data
    })

@app.route('/api/semesters', methods=['GET'])
def get_semesters():
    return jsonify({'semesters': SEMESTERS})

@app.route('/api/current_demand', methods=['GET'])
def get_current_demand():
    """Returns demand predictions for the current academic moment.

    Optionally accepts query params for testing: date, week, day, hour.
    """
    now = datetime.now()

    # Allow override via query params for testing
    date_str = request.args.get('date')
    if date_str:
        try:
            now = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    acad = get_current_academic_time(now)
    if acad is None:
        return jsonify({'error': 'No semester configured for this date.'}), 400

    # Holiday state — return zero demand for all buildings
    if acad.get('holiday'):
        buildings = df['Building'].unique().tolist()
        holiday_predictions = [{'Building': b, 'Demand_Prediction': 0} for b in buildings]
        return jsonify({
            'academic': acad,
            'week': 0,
            'day': 'Holiday',
            'hour': now.hour,
            'predictions': holiday_predictions
        })

    # Allow week/day/hour override
    week = request.args.get('week', default=acad['week'], type=int)
    day = request.args.get('day', default=acad['day'], type=str)
    hour = request.args.get('hour', default=acad['hour'], type=int)

    filtered_df = df[(df['Week'] == week) & (df['Day'] == day) & (df['Hour'] == hour)]
    results = filtered_df[['Building', 'Demand_Prediction']].to_dict('records')

    return jsonify({
        'academic': acad,
        'week': week,
        'day': day,
        'hour': hour,
        'predictions': results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)