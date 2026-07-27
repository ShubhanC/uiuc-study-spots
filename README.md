# UIUC Study Spots

A web application that predicts how busy study spots are on the University of Illinois at Urbana-Champaign campus.

## Problem

Students often struggle to find available study spaces on campus, especially during peak times like midterms and finals. Existing tools like Google Popular Times provide general foot traffic data but don't capture the specific dynamics of academic spaces (e.g., sudden emptying after a lecture ends). This project aims to provide a more accurate, real-time prediction of study spot occupancy by combining:

- Historical foot traffic data (Google Popular Times)
- Class schedules and exam schedules
- Historical occupancy patterns
- User-reported occupancy (optional)

The goal is to help students make informed decisions about where to study, reducing wasted time searching for available spaces.

## Architecture

The application consists of a Flask backend serving a vanilla JavaScript SPA (Single Page Application). There is no build step or frontend framework.

### Backend (`api/index.py`)
- **Framework**: Flask (Python)
- **Data Source**: Pre-processed CSV file (`data/master_campus_demand_W1_to_W16.csv`) containing weekly/daily/hourly demand predictions for 8 campus buildings.
- **Academic Calendar Logic**: Encodes semester start dates, breaks (spring break, Thanksgiving), and week mapping to ensure accurate date-to-week conversion.
- **API Endpoints**:
  - `GET /` - Serves the frontend `index.html`
  - `GET /api/demand?week=&day=&hour=` - Returns demand predictions for all buildings at a specific time
  - `GET /api/buildings` - Returns list of all building names
  - `GET /api/all_days?week=&hour=` - Returns demand for all buildings across all days of a week at a given hour (used for monthly calendar view)
  - `GET /api/calendar/<building>?week=` - Returns a 24x7 grid of demand for a specific building and week (used for daily detail view)
  - `GET /api/semesters` - Returns configured academic periods
  - `GET /api/current_demand?date=` - Returns demand for the current moment (or a specified date for testing)

### Frontend (`public/`)
- **Language**: Vanilla JavaScript (ES6)
- **Styling**: Custom CSS (no framework)
- **Views** (managed by toggling CSS classes):
  1. **Home (`view-home`)** - Landing page with hero image and call-to-action
  2. **Hub (`view-hub`)** - Navigation to Map, Calendar, and Quick Search
  3. **Map (`view-map`)** - Interactive campus map with color-coded building overlays showing real-time demand
  4. **Monthly Calendar (`view-calendar-out`)** - Month-view calendar where each day is colored by average demand (at noon)
  5. **Daily Detail (`view-calendar-in`)** - Hourly bar chart for a selected building and day, plus a ranked list of all buildings for that hour

### Key Features
- **Real-time Map View**: Building overlays update every 5 minutes with live demand data.
- **Monthly Planning**: Color-coded calendar helps identify generally busy/quiet days.
- **Daily Detail View**: See hour-by-hour demand for a specific building and compare across all buildings.
- **Academic Calendar Awareness**: Correctly handles semester breaks (spring break, Thanksgiving) and adjusts week numbering accordingly.
- **Responsive Design**: Works on desktop and mobile browsers.

### Data
The prediction model (not included in this repository) generates the `master_campus_demand_W1_to_W16.csv` file, which contains:
- `Week`: Semester week (1-16)
- `Day`: Monday-Sunday
- `Hour`: 0-23
- `Building`: One of 8 campus buildings
- `Demand_Prediction`: Float from -1.0 (closed) to 1.0 (fully occupied)
- `Pressure_Multiplier`: Internal model factor (not used in UI)
- `Timestep`: Sequential index

## Results
Due to the nature of the prediction model (trained on historical data), the tool provides reasonable estimates of building occupancy. During validation:
- The map view correctly reflects known busy periods (e.g., midterm weeks show higher demand in libraries).
- The Thanksgiving break logic correctly shows reduced demand during the holiday week.
- The semester end date correctly falls on a Friday, preventing erroneous "in-session" labels on weekends.

Screenshots would typically show:
1. The home screen with the project introduction.
2. The hub with navigation cards.
3. The map view with colored building overlays.
4. The monthly calendar with day-specific coloring.
5. The daily detail view with an hourly bar chart and building list.

## How to Run
1. **Prerequisites**:
   - Python 3.8+
   - Conda (optional but recommended) or virtualenv
   - Required Python packages: Flask, pandas

2. **Setup**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd uiuc-study-spots

   # (Optional) Create and activate a conda environment
   conda create -n uiuc-spots python=3.9
   conda activate uiuc-spots

   # Install dependencies
   pip install flask pandas
   ```

3. **Run the Server**:
   ```bash
   # From the project root
   python api/index.py
   ```
   The server will start on `http://localhost:5000`.

4. **Usage**:
   - Open your browser to `http://localhost:5000`.
   - Navigate via the hub or directly to views.
   - In the calendar view, click a day to see the hourly demand breakdown for that day.

## Architecture Diagram (Textual)
```
+---------------------+       +----------------------+
|   Web Browser       |<----->|   Flask Server       |
| (SPA Vanilla JS)   |  HTTP | (api/index.py)       |
+---------------------+       +----------------------+
        ^                             ^
        |                             |
        |          Serves static      | Serves JSON API
        |          files (HTML, CSS,  | endpoints:
        |          JS) from /public   |  - /api/demand
        |                             |  - /api/buildings
        |                             |  - /api/all_days
        |                             |  - /api/calendar/<building>
        |                             |  - /api/semesters
        |                             |  - /api/current_demand
        v                             v
+---------------------+       +----------------------+
|   Static Assets     |       |   CSV Data File      |
| (HTML, CSS, JS, SVG)|       | master_campus_demand_|
|   served by Flask   |       |   W1_to_W16.csv      |
+---------------------+       +----------------------+
```

## Notes
- The prediction model itself is not included in this repository; only its output CSV is used.
- To update predictions, replace `data/master_campus_demand_W1_to_W16.csv` with a newly generated file (same format).
- The academic calendar is hardcoded for Spring 2026, Fall 2026, and Spring 2027. To add new semesters, update the `SEMESTERS` list in both `api/index.py` and `public/script.js`.

## Future Improvements
- Integrate real-time user feedback to adjust predictions.
- Add push notifications for when a favorite building becomes available.
- Implement predictive caching to reduce API calls.
- Expand to more buildings and campuses.
- Add user accounts and saved preferences.

## License
This project is provided as-is for educational purposes.

---
*Last updated: July 2026*