/**
 * Illini Spots - Frontend Controller
 */

let selectedWeek = 1;
let selectedDay = 'Monday';
let selectedHour = 12;
let selectedBuilding = 'Grainger Library'; // internal name (CSV)
let allBuildings = ['Grainger Library', 'Funk Library', 'Main Library', 'Illini Union', 'Siebel Center for CS', 'CIF', 'BIF', 'Siebel Center for Design'];

// Calendar state
let calendarDate = new Date(); // tracks which month/year is shown
let selectedDateStr = ''; // 'YYYY-MM-DD' of the clicked day

// Display names used on map overlay labels (key = internal CSV name, value = label shown on map)
const MAP_LABELS = {
    'Grainger Library': 'Grainger Engineering Library',
    'Funk Library': 'ACES Library',
};

// ── Academic Calendar (mirrors backend) ──
const SEMESTERS = [
    { id: 'sp2026', type: 'Spring', start_date: '2026-01-20', weeks: 16, has_spring_break: true,  spring_break_calendar_week: 9 },
    { id: 'fa2026', type: 'Fall',   start_date: '2026-08-24', weeks: 16, has_spring_break: false },
    { id: 'sp2027', type: 'Spring', start_date: '2027-01-19', weeks: 16, has_spring_break: true,  spring_break_calendar_week: 9 },
];

function getCurrentAcademicTime(now) {
    now = now || new Date();
    const ts = now.getTime();

    let current = null;
    for (const sem of SEMESTERS) {
        const start = new Date(sem.start_date + 'T00:00:00');
        let end = new Date(start);
        end.setDate(end.getDate() + sem.weeks * 7);
        if (sem.has_spring_break) end.setDate(end.getDate() + 7);
        if (ts >= start.getTime() && ts < end.getTime()) { current = sem; break; }
    }
    if (!current) {
        // Outside any semester — return holiday state
        return {
            holiday: true,
            semester_type: 'Holiday',
            week: 0,
            day: 'Holiday',
            hour: now.getHours(),
            datetime: now.toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })
        };
    }

    const start = new Date(current.start_date + 'T00:00:00');
    const daysSince = Math.floor((ts - start.getTime()) / 86400000);
    let calWeek = Math.max(1, Math.floor(daysSince / 7) + 1);
    const maxCal = current.weeks + (current.has_spring_break ? 1 : 0);
    calWeek = Math.min(calWeek, maxCal);

    let dataWeek = calWeek;
    if (current.has_spring_break) {
        const sbw = current.spring_break_calendar_week || 9;
        if (calWeek >= sbw) dataWeek = Math.max(1, calWeek - 1);
    }
    dataWeek = Math.max(1, Math.min(dataWeek, current.weeks));

    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const dayName = days[now.getDay()];
    const hour = now.getHours();

    return {
        semester_id: current.id,
        semester_type: current.type,
        semester_start: current.start_date,
        calendar_week: calWeek,
        week: dataWeek,
        day: dayName,
        hour: hour,
        datetime: now.toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    };
}

/* ── Building map positions (% positions on the map image, 1008×1043 px) ── */
const BUILDING_POSITIONS = {
    'Siebel Center for CS':          { x: 59.2, y: 14.7 },
    'Grainger Library':              { x: 48.0, y: 33 },  // label: Grainger Engineering Library
    'CIF':                           { x: 41, y: 38.5 },
    'Illini Union':                  { x: 47.0, y: 52 },
    'Main Library':                  { x: 43, y: 75 },
    'BIF':                           { x: 33.0, y: 84 },
    'Siebel Center for Design':      { x: 40, y: 93.4 },
    'Funk Library':                  { x: 59, y: 91.8 },  // label: ACES Library
};

function demandColor(demand) {
    if (demand < 0) return '#888';
    if (demand >= 0.8) return '#e53935';  // red
    if (demand >= 0.6) return '#FF5F05';  // orange
    if (demand >= 0.3) return '#FDD835';  // yellow
    return '#4CAF50';                      // green
}

function demandStatus(demand) {
    if (demand < 0) return 'Closed';
    if (demand >= 0.8) return 'Crowded';
    if (demand >= 0.6) return 'Busy';
    if (demand >= 0.3) return 'Moderate';
    return 'Available';
}

document.addEventListener('DOMContentLoaded', async () => {
    // Fetch live building list
    try {
        const res = await fetch('/api/buildings');
        const data = await res.json();
        if (data.buildings && data.buildings.length) allBuildings = data.buildings;
    } catch (_) {}
    populateBuildingDropdown();
    renderMapMarkers();
    renderMonthlyCalendar();
    startMapAutoRefresh();
});

function navigateTo(viewId) {
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    const targetView = document.getElementById(viewId);
    if (targetView) targetView.classList.add('active');

    if (viewId === 'view-map') {
        renderMapMarkers();
    }
    if (viewId === 'view-calendar-out') {
        renderMonthlyCalendar();
    }
    if (viewId === 'view-calendar-in') {
        updateDayDetail();
    }
}

/* ── Monthly Calendar ── */

function dateToWeekInfo(dateStr) {
    const date = new Date(dateStr + 'T12:00:00');
    const ts = date.getTime();

    for (const sem of SEMESTERS) {
        const start = new Date(sem.start_date + 'T00:00:00');
        let end = new Date(start);
        end.setDate(end.getDate() + sem.weeks * 7);
        if (sem.has_spring_break) end.setDate(end.getDate() + 7);
        if (ts >= start.getTime() && ts < end.getTime()) {
            const daysSince = Math.floor((ts - start.getTime()) / 86400000);
            let calWeek = Math.max(1, Math.floor(daysSince / 7) + 1);
            const maxCal = sem.weeks + (sem.has_spring_break ? 1 : 0);
            calWeek = Math.min(calWeek, maxCal);

            let dataWeek = calWeek;
            if (sem.has_spring_break) {
                const sbw = sem.spring_break_calendar_week || 9;
                if (calWeek >= sbw) dataWeek = Math.max(1, calWeek - 1);
            }
            dataWeek = Math.max(1, Math.min(dataWeek, sem.weeks));

            const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
            const dayName = days[date.getDay()];

            return { semester: sem.id, week: dataWeek, day: dayName, inSemester: true };
        }
    }
    return { inSemester: false };
}

async function renderMonthlyCalendar() {
    const grid = document.getElementById('month-grid');
    if (!grid) return;

    const year = calendarDate.getFullYear();
    const month = calendarDate.getMonth();

    document.getElementById('month-title').textContent =
        new Date(year, month).toLocaleString('en-US', { month: 'long', year: 'numeric' });

    const today = new Date();
    const todayStr = today.toISOString().slice(0, 10);

    // First day of month and number of days
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startDayOfWeek = firstDay.getDay(); // 0=Sun

    // Days from prev month to fill first row
    const prevMonthLastDay = new Date(year, month, 0).getDate();

    grid.innerHTML = '';

    // Day-of-week headers
    const dayHeaders = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    dayHeaders.forEach(d => {
        const header = document.createElement('div');
        header.className = 'day-header';
        header.textContent = d;
        grid.appendChild(header);
    });

    // Collect all days in this month that are in semester to fetch predictions
    const semesterDays = [];
    for (let d = 1; d <= daysInMonth; d++) {
        const dateObj = new Date(year, month, d);
        const dateStr = dateObj.toISOString().slice(0, 10);
        const info = dateToWeekInfo(dateStr);
        if (info.inSemester) {
            semesterDays.push({ date: dateStr, week: info.week, day: info.day });
        }
    }

    // Fetch demand data for semester days at hour 12 (noon — representative)
    let demandMap = {};
    if (semesterDays.length > 0) {
        try {
            // Fetch predictions for each distinct (week, day) combo
            const uniqueCombos = [...new Map(semesterDays.map(d => [`${d.week}-${d.day}`, d])).values()];
            for (const combo of uniqueCombos) {
                const res = await fetch(`/api/demand?week=${combo.week}&day=${combo.day}&hour=12`);
                const data = await res.json();
                if (data.predictions) {
                    const avgDemand = data.predictions.reduce((s, p) => s + (p.Demand_Prediction >= 0 ? p.Demand_Prediction : 0), 0)
                        / data.predictions.filter(p => p.Demand_Prediction >= 0).length || 0;
                    demandMap[`${combo.week}-${combo.day}`] = avgDemand;
                }
            }
        } catch (_) {}
    }

    // Empty cells for days before month starts
    for (let i = 0; i < startDayOfWeek; i++) {
        const empty = document.createElement('div');
        empty.className = 'day-cell other-month';
        const dayNum = prevMonthLastDay - startDayOfWeek + 1 + i;
        empty.innerHTML = `<div class="day-number">${dayNum}</div>`;
        grid.appendChild(empty);
    }

    // Day cells
    for (let d = 1; d <= daysInMonth; d++) {
        const dateObj = new Date(year, month, d);
        const dateStr = dateObj.toISOString().slice(0, 10);
        const cell = document.createElement('div');
        cell.className = 'day-cell';
        if (dateStr === todayStr) cell.classList.add('today');

        const info = dateToWeekInfo(dateStr);
        let demandLevel = null;

        if (info.inSemester) {
            const key = `${info.week}-${info.day}`;
            if (demandMap[key] !== undefined) demandLevel = demandMap[key];
        }

        let previewHtml = '';
        if (demandLevel !== null && demandLevel >= 0) {
            const color = demandColor(demandLevel);
            const statusText = demandStatus(demandLevel);
            previewHtml = `<div class="day-demand-preview"><span class="day-demand-dot" style="background:${color}"></span>${statusText}</div>`;
        } else {
            previewHtml = `<div class="day-demand-preview" style="color:#ccc;">—</div>`;
        }

        cell.innerHTML = `<div class="day-number">${d}</div>${previewHtml}`;
        cell.onclick = () => {
            selectedDateStr = dateStr;
            selectedDay = info.inSemester ? info.day : dateObj.toLocaleString('en-US', { weekday: 'long' });
            selectedWeek = info.inSemester ? info.week : 0;
            // Set hour to current Central Time
            const now = new Date();
            const centralOffset = -6; // CST = UTC-6, CDT = UTC-5
            const localOffset = -now.getTimezoneOffset() / 60;
            const centralHour = (now.getUTCHours() + centralOffset + 24) % 24;
            selectedHour = centralHour;
            document.getElementById('hour-slider-in').value = selectedHour;
            document.getElementById('hour-display-in').textContent = `${selectedHour}:00`;
            navigateTo('view-calendar-in');
        };

        grid.appendChild(cell);
    }

    // Fill remaining cells
    const totalCells = startDayOfWeek + daysInMonth;
    const remaining = (7 - (totalCells % 7)) % 7;
    for (let i = 1; i <= remaining; i++) {
        const empty = document.createElement('div');
        empty.className = 'day-cell other-month';
        empty.innerHTML = `<div class="day-number">${i}</div>`;
        grid.appendChild(empty);
    }
}

function navigateMonth(delta) {
    calendarDate.setMonth(calendarDate.getMonth() + delta);
    renderMonthlyCalendar();
}

/* ── Calendar In: Day Detail View ── */

async function updateDayDetail() {
    const slider = document.getElementById('hour-slider-in');
    selectedHour = parseInt(slider ? slider.value : selectedHour);
    const display = document.getElementById('hour-display-in');
    if (display) display.textContent = `${selectedHour}:00`;

    // Format the date nicely
    if (selectedDateStr) {
        const d = new Date(selectedDateStr + 'T12:00:00');
        const formatted = d.toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
        const el = document.getElementById('detail-full-date');
        if (el) el.textContent = formatted;
    }

    // Check if this date is in a semester
    const info = dateToWeekInfo(selectedDateStr);
    const inSemester = info && info.inSemester;

    if (!inSemester) {
        // Holiday — show all buildings with zero demand
        const chart = document.getElementById('hourly-chart');
        if (chart) {
            chart.innerHTML = '';
            for (let h = 0; h < 24; h++) {
                const container = document.createElement('div');
                container.className = 'hour-bar-container';
                const bar = document.createElement('div');
                bar.className = 'hour-bar';
                bar.style.height = '0%';
                bar.style.background = '#eee';
                container.appendChild(bar);
                const label = document.createElement('span');
                label.className = 'hour-label';
                label.textContent = `${h}`;
                container.appendChild(label);
                chart.appendChild(container);
            }
        }
        const list = document.getElementById('demand-list');
        if (list) {
            list.innerHTML = allBuildings.map(b => {
                const displayName = MAP_LABELS[b] || b;
                return `
                    <div class="building-demand-card" style="border-left-color:#888;">
                        <div class="building-info">
                            <h4>${displayName}</h4>
                            <p>Holiday</p>
                        </div>
                        <div class="demand-bar-bg">
                            <div class="demand-bar-fill" style="width:0%;background:#888;"></div>
                        </div>
                        <span class="demand-value">0%</span>
                    </div>
                `;
            }).join('');
        }
        return;
    }

    selectedWeek = info.week;
    selectedDay = info.day;

    // Build hourly chart
    await renderHourlyBars();

    // Build demand list
    await renderDemandList();
}

async function renderHourlyBars() {
    const chart = document.getElementById('hourly-chart');
    if (!chart) return;

    chart.innerHTML = '<p style="color:#888;padding:20px;">Loading chart...</p>';

    try {
        const res = await fetch(`/api/calendar/${encodeURIComponent(selectedBuilding)}?week=${selectedWeek}`);
        const data = await res.json();
        const calendar = data.calendar || {};

        const dayData = calendar[selectedDay];
        chart.innerHTML = '';

        for (let h = 0; h < 24; h++) {
            const demand = dayData && dayData[h] != null ? dayData[h] : -1;
            const percent = demand < 0 ? 0 : Math.round(demand * 100);
            const heightPercent = demand < 0 ? 0 : Math.max(2, percent);
            const color = demand < 0 ? '#eee' : demandColor(demand);
            const isActive = h === selectedHour;

            const container = document.createElement('div');
            container.className = 'hour-bar-container';
            container.style.cursor = 'pointer';

            const bar = document.createElement('div');
            bar.className = 'hour-bar';
            bar.style.height = `${heightPercent}%`;
            bar.style.background = isActive ? color : (demand < 0 ? '#eee' : '#ddd');
            bar.style.opacity = isActive ? '1' : '0.5';
            bar.title = `${h}:00 - ${percent}%`;
            container.appendChild(bar);

            const label = document.createElement('span');
            label.className = 'hour-label';
            label.textContent = `${h}`;
            container.appendChild(label);

            container.onclick = () => {
                document.getElementById('hour-slider-in').value = h;
                updateDayDetail();
            };

            chart.appendChild(container);
        }
    } catch (e) {
        chart.innerHTML = '<p style="color:#888;padding:20px;">Error loading chart data.</p>';
    }
}

async function renderDemandList() {
    const list = document.getElementById('demand-list');
    if (!list) return;

    list.innerHTML = '<p style="padding:10px 0;color:#888;">Loading...</p>';

    try {
        const res = await fetch(`/api/demand?week=${selectedWeek}&day=${selectedDay}&hour=${selectedHour}`);
        const data = await res.json();
        const items = data.predictions || [];

        list.innerHTML = '';

        items.forEach(item => {
            const demand = item.Demand_Prediction;
            const percent = demand < 0 ? 0 : Math.round(demand * 100);
            const color = demandColor(demand);
            const statusText = demand < 0 ? 'Closed' : demandStatus(demand);

            const card = document.createElement('div');
            card.className = 'building-demand-card';
            card.style.borderLeftColor = color;
            card.onclick = () => {
                selectedBuilding = item.Building;
                renderHourlyBars();
            };
            card.innerHTML = `
                <div class="building-info">
                    <h4>${item.Building}${item.Building === selectedBuilding ? ' <span style="font-size:12px;color:#FF5F05;">(selected)</span>' : ''}</h4>
                    <p>${statusText}</p>
                </div>
                <div class="demand-bar-bg">
                    <div class="demand-bar-fill" style="width:${percent}%;background:${color};"></div>
                </div>
                <span class="demand-value">${percent}%</span>
            `;
            list.appendChild(card);
        });
    } catch (e) {
        list.innerHTML = '<p style="padding:10px 0;color:#888;">Error loading data.</p>';
    }
}

function populateBuildingDropdown() {
    const select = document.getElementById('building-dropdown');
    if (!select) return;
    select.innerHTML = '<option value="">Select a building...</option>';
    allBuildings.forEach(b => {
        const opt = document.createElement('option');
        opt.value = b;
        opt.textContent = b;
        select.appendChild(opt);
    });
}

/* ── Hub ── */

function handleBuildingChange() {
    const select = document.getElementById('building-dropdown');
    if (!select || !select.value) return;
    selectedBuilding = select.value;
    if (!selectedDateStr) {
        const now = new Date();
        selectedDateStr = now.toISOString().slice(0, 10);
    }
    selectedHour = 12;
    navigateTo('view-calendar-in');
}

/* ── Map ── */

async function renderMapMarkers() {
    const container = document.getElementById('map-markers');
    if (!container) return;
    container.innerHTML = '';

    // Update date/time display
    const acad = getCurrentAcademicTime();
    const dtEl = document.getElementById('map-datetime');
    const semEl = document.getElementById('map-semester');
    const isHoliday = acad && acad.holiday;

    if (dtEl && acad) dtEl.textContent = acad.datetime;
    if (semEl && acad) {
        if (isHoliday) {
            semEl.textContent = 'Holiday — No classes in session';
            semEl.style.color = '#888';
        } else {
            semEl.textContent = `${acad.semester_type} ${acad.semester_id.slice(2,6)} · Week ${acad.week} · ${acad.day}`;
            semEl.style.color = '';
        }
    }

    // Fetch current demand (skip if holiday — all demand = 0)
    let predictions = [];
    if (!isHoliday) {
        try {
            const res = await fetch('/api/current_demand');
            const data = await res.json();
            predictions = data.predictions || [];
        } catch (_) {}
    }

    // Build a lookup: building -> demand
    const demandMap = {};
    predictions.forEach(p => { demandMap[p.Building] = p.Demand_Prediction; });

    // Update allBuildings from predictions
    if (predictions.length) {
        allBuildings = predictions.map(p => p.Building);
    }

    allBuildings.forEach(building => {
        const pos = BUILDING_POSITIONS[building];
        if (!pos) return;

        let demand, percent, color;

        if (isHoliday) {
            demand = 0;
            percent = 0;
            color = '#888';
        } else {
            demand = demandMap[building] !== undefined ? demandMap[building] : -1;
            percent = demand < 0 ? 0 : Math.round(demand * 100);
            color = demandColor(demand);
        }

        // — Semi-transparent colored overlay acting as button —
        const overlay = document.createElement('div');
        overlay.className = 'map-building-overlay';
        overlay.style.left = `${pos.x}%`;
        overlay.style.top = `${pos.y}%`;
        overlay.style.background = `${color}CC`; // hex with alpha ~80%
        overlay.title = `${MAP_LABELS[building] || building}\n${percent}% · ${demand < 0 ? 'Closed' : demandStatus(demand)}`;

        overlay.onclick = () => {
            selectedBuilding = building;
            const now = new Date();
            selectedDateStr = now.toISOString().slice(0, 10);
            navigateTo('view-calendar-in');
        };

        // Label inside the overlay
        const labelEl = document.createElement('span');
        labelEl.className = 'overlay-label';
        labelEl.textContent = MAP_LABELS[building] || building;
        overlay.appendChild(labelEl);

        container.appendChild(overlay);
    });

    // Populate the detail panel
    populateMapDetailPanel(predictions, isHoliday);
}

function populateMapDetailPanel(predictions, isHoliday) {
    const body = document.getElementById('map-detail-body');
    if (!body) return;

    const acad = getCurrentAcademicTime();

    // Holiday state
    if (isHoliday) {
        let html = '<div class="map-detail-section-title">Building Demand</div>';
        allBuildings.forEach(building => {
            const displayName = MAP_LABELS[building] || building;
            html += `
                <div class="map-building-row">
                    <div class="map-building-dot" style="background:#888;"></div>
                    <span class="map-building-name">${displayName}</span>
                    <span class="map-building-status" style="color:#888;">Holiday</span>
                    <span class="map-building-percent">0%</span>
                </div>
            `;
        });
        body.innerHTML = html;
        return;
    }

    if (!predictions || !predictions.length) {
        body.innerHTML = '<div class="map-panel-placeholder">No demand data available for this time.</div>';
        return;
    }

    // Sort by demand descending (closed buildings last)
    const sorted = [...predictions].sort((a, b) => {
        const da = a.Demand_Prediction < 0 ? -1 : a.Demand_Prediction;
        const db = b.Demand_Prediction < 0 ? -1 : b.Demand_Prediction;
        return db - da;
    });

    let html = '<div class="map-detail-section-title">Building Demand</div>';
    sorted.forEach(item => {
        const demand = item.Demand_Prediction;
        const percent = demand < 0 ? 0 : Math.round(demand * 100);
        const color = demandColor(demand);
        const statusText = demand < 0 ? 'Closed' : demandStatus(demand);
        const displayName = MAP_LABELS[item.Building] || item.Building;

        html += `
            <div class="map-building-row" onclick="selectMapBuilding('${item.Building.replace(/'/g, "\\'")}')">
                <div class="map-building-dot" style="background:${color};"></div>
                <span class="map-building-name">${displayName}</span>
                <span class="map-building-status" style="color:${color};">${statusText}</span>
                <span class="map-building-percent">${percent}%</span>
            </div>
        `;
    });
    body.innerHTML = html;
}

function selectMapBuilding(building) {
    selectedBuilding = building;
    const now = new Date();
    selectedDateStr = now.toISOString().slice(0, 10);
    navigateTo('view-calendar-in');
}

// Auto-refresh map every 5 minutes
let mapRefreshInterval = null;
function startMapAutoRefresh() {
    if (mapRefreshInterval) clearInterval(mapRefreshInterval);
    mapRefreshInterval = setInterval(() => {
        const mapView = document.getElementById('view-map');
        if (mapView && mapView.classList.contains('active')) {
            renderMapMarkers();
        }
    }, 300000); // 5 minutes
}