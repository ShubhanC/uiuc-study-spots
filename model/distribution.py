# Tuning parameters for each location's peaks (on a normal day) and distributions:
grainger_max = 0.8
funk_max = 0.75
union_max_lunch = 0.75
union_max_dinner = 0.65
main_lib_max = 0.7

import pandas as pd
import numpy as np

# --- 1. Helper to generate a smoothed hour-by-hour curve ---
def generate_daily_curve(peak_hour, width=3.0, max_val=1.0, is_weekend=False):
    """Creates a bell curve of occupancy based on the peak hour."""
    hours = np.arange(24)
    # Gaussian distribution equation
    curve = np.exp(-0.5 * ((hours - peak_hour) / width)**2)
    
    # Scale down maximum occupancy on weekends
    if is_weekend:
        max_val *= 0.6
        
    return curve * max_val

# --- 2. Build the Base Google Curves for the Week ---
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
base_data = []

for day in days:
    is_wkd = day in ['Saturday', 'Sunday']
    
    # Grainger: Peak 3 PM (15), busiest Wed-Fri
    g_max = grainger_max if day in ['Wednesday', 'Thursday', 'Friday'] else 0.85*grainger_max
    grainger = generate_daily_curve(15, width=3.5, max_val=g_max, is_weekend=is_wkd)
    
    # Funk: Variable peaks
    if day in ['Monday', 'Wednesday']: f_peak = 13
    elif day in ['Tuesday', "Thursday"]: f_peak = 14
    elif day == 'Friday': f_peak = 12
    else: f_peak = 14
    f_max = funk_max if not is_wkd else 0.5
    funk = generate_daily_curve(f_peak, width=3.0, max_val=f_max, is_weekend=is_wkd)
    
    # Main Library: Peak 2 PM (14)
    m_max = main_lib_max if not is_wkd else 0.5*main_lib_max 
    main_lib = generate_daily_curve(14, width=3.5, max_val=m_max, is_weekend=is_wkd)
    
    # Illini Union: Bi-modal (Lunch and Dinner peaks)
    u_lunch = generate_daily_curve(12, width=2.0, max_val=union_max_lunch, is_weekend=is_wkd)
    u_dinner = generate_daily_curve(18, width=3.0, max_val=union_max_dinner, is_weekend=is_wkd)
    union = np.maximum(u_lunch, u_dinner) # Takes the highest point of both curves
    
    for h in range(24):
        base_data.append({
            'Day': day, 'Hour': h, 
            'Grainger Library': grainger[h], 
            'Funk Library': funk[h], 
            'Main Library': main_lib[h], 
            'Illini Union': union[h]
        })

df_base = pd.DataFrame(base_data)

# --- 3. The Sister Mapping & Survey Scaling ---
# Multiply by the survey ratios (e.g. CIF had 14 votes, Grainger had 15)
df_base['CIF'] = df_base['Grainger Library'] * (14/15)
df_base['BIF'] = df_base['Main Library'] * (7/10)
df_base['Siebel Center for CS'] = df_base['Grainger Library'] * (7/15)
df_base['Siebel Center for Design'] = df_base['Funk Library'] * (7/10)

# Reshape data for ML
df_melted = df_base.melt(id_vars=['Day', 'Hour'], var_name='Building', value_name='Base_Popularity')

# --- 4. Advanced Building Hours Mask ---

# We map each building to a dictionary of Days, containing a list of integer hours (0-23) it is open.
# Note: 8:30 AM is rounded to 8 (8:00 AM - 8:59 AM hour block).
# Closes at 11 PM (23:00) means the 22nd hour (10:00 PM - 10:59 PM) is the last fully open block. 
# Midnight (12 AM) is effectively the end of the 23rd hour.

schedule = {
    'Grainger Library': {
        'Monday': list(range(0, 24)), 
        'Tuesday': list(range(0, 24)), 
        'Wednesday': list(range(0, 24)), 
        'Thursday': list(range(0, 24)), 
        'Friday': list(range(0, 24)), 
        'Saturday': list(range(10, 24)), 
        'Sunday': list(range(10, 24))
    },
    'Funk Library': {
        # 1A and 2A spillovers handled at 0 and 1 of the NEXT day
        'Monday': [0, 1] + list(range(8, 24)),      # Sun night spillover + 8:30A-Midnight
        'Tuesday': [0, 1] + list(range(8, 24)),     # Mon night spillover + 8:30A-Midnight
        'Wednesday': [0, 1] + list(range(8, 24)),   # Tue night spillover + 8:30A-Midnight
        'Thursday': [0, 1] + list(range(8, 24)),    # Wed night spillover + 8:30A-Midnight
        'Friday': [0, 1] + list(range(8, 18)),      # Thu night spillover + 8:30A-6P
        'Saturday': list(range(10, 21)),            # 10A-9P
        'Sunday': list(range(13, 24))               # 1P-Midnight (spills into Monday)
    },
    'Main Library': {
        'Monday': list(range(8, 23)),               # 8:30A-11P
        'Tuesday': list(range(8, 23)),
        'Wednesday': list(range(8, 23)),
        'Thursday': list(range(8, 23)),
        'Friday': list(range(8, 18)),               # 8:30A-6P
        'Saturday': list(range(13, 17)),            # 1P-5P
        'Sunday': list(range(13, 23))               # 1P-11P
    },
    'Illini Union': {
        day: list(range(7, 23)) for day in days     # 7A-11P Every Day
    },
    'CIF': {
        'Monday': list(range(7, 23)),
        'Tuesday': list(range(7, 23)),
        'Wednesday': list(range(7, 23)),
        'Thursday': list(range(7, 23)),
        'Friday': list(range(7, 21)),               # 7A-9P
        'Saturday': list(range(9, 21)),             # 9A-9P
        'Sunday': list(range(12, 23))               # 12P-11P
    },
    'BIF': {
        'Monday': list(range(6, 24)),               # 6A-12A
        'Tuesday': list(range(6, 24)),
        'Wednesday': list(range(6, 24)),
        'Thursday': list(range(6, 24)),
        'Friday': list(range(6, 20)),               # 6A-8P
        'Saturday': list(range(6, 20)),             # 6A-8P
        'Sunday': list(range(10, 24))               # Assuming 10A-12A (Midnight)
    },
    'Siebel Center for CS': {
        day: list(range(7, 22)) for day in days     # 7A-9:30P (Using 21 as last full hour)
    },
    'Siebel Center for Design': {
        day: list(range(7, 24)) for day in days     # 7A-12A Every Day
    }
}

def apply_hours_mask(row):
    building = row['Building']
    day = row['Day']
    hour = row['Hour']
    base_pop = row['Base_Popularity']
    
    open_hours = schedule[building][day]
    
    if hour in open_hours:
        return base_pop
    else:
        # --- Handle After-Hours Edge Cases ---
        
        # Siebel CS Swipe Access: Instead of dropping to 0, it drops to 15% of its normal 
        # popularity curve (simulating majors swiping in to study late).
        if building == 'Siebel Center for CS':
            return base_pop * 0.15 
            
        # All other buildings are strictly locked
        return 0.0

df_melted['Base_Popularity'] = df_melted.apply(apply_hours_mask, axis=1)

print(df_melted.head(10))

df_melted.to_csv('./data/base_popularity_distributions.csv', index=False)