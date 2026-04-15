import math
import numpy as np
import pandas as pd

## distribution is df_melted form distribution.py

class CampusDemandModel:
    def __init__(self, baseline_df):
        # baseline_df is the dataframe we generated in the previous step 
        # (contains Day, Hour, Building, Base_Popularity)
        self.baseline = baseline_df
        
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Survey votes for active distribution (excluding Dorms)
        self.building_votes = {
            'Grainger Library': 15, 'CIF': 14, 'Illini Union': 13, 
            'Main Library': 10, 'Funk Library': 10, 'BIF': 7, 
            'Siebel Center for Design': 7, 'Siebel Center for CS': 5
        }
        self.total_votes = sum(self.building_votes.values())
        
        # Structured Exam Schedule (Exams mapped to the day they end)
        # Format: (Week, Day) : Number of exams
        self.exam_schedule = {
            # Week 3
            (3, 'Tuesday'): 1,   # Chem 104 CBTF
            (3, 'Thursday'): 1,  # CS 173 CBTF
            (3, 'Friday'): 1,    # Ochem 1
            # Week 4
            (4, 'Wednesday'): 2, # Math 257 CBTF, Chem 101
            (4, 'Thursday'): 1,  # CS 173 CBTF
            (4, 'Friday'): 1,    # Math 241 A/C CBTF
            (4, 'Saturday'): 1,  # MCB 150 CBTF
            # Week 5
            (5, 'Tuesday'): 2,   # Math 220, Chem 104 CBTF
            (5, 'Wednesday'): 1, # Chem 102
            (5, 'Thursday'): 2,  # Math 241 B, CS 173 CBTF
            (5, 'Friday'): 1,    # Phys 211
            (5, 'Sunday'): 1,    # Phys 212 CBTF
            # Week 6
            (6, 'Thursday'): 1,  # CS 173 CBTF
            (6, 'Friday'): 1,    # CS 173 C
            # Week 7
            (7, 'Tuesday'): 1,   # Chem 104 CBTF
            (7, 'Wednesday'): 1, # Chem 101
            (7, 'Thursday'): 1,  # CS 173 CBTF
            (7, 'Friday'): 2,    # Math 241 A/C CBTF, Ochem 1
            # Week 8
            (8, 'Wednesday'): 1, # Math 257 CBTF
            (8, 'Thursday'): 1,  # CS 173 CBTF
            (8, 'Friday'): 1,    # MCB 150 CBTF
            # Week 9
            (9, 'Tuesday'): 2,   # Math 220, Chem 104 CBTF
            (9, 'Thursday'): 2,  # Math 241 B, CS 173 CBTF
            (9, 'Friday'): 2,    # Math 241 A/C CBTF, Phys 211
            # Week 10
            (10, 'Wednesday'): 1, # Chem 102
            (10, 'Thursday'): 1,  # CS 173 CBTF
            (10, 'Friday'): 1,    # CS 173 C
            (10, 'Sunday'): 1,    # Phys 212 CBTF
            # Week 11
            (11, 'Friday'): 1, # Ochem 1
            # Week 12
            (12, 'Tuesday') : 1,
            (12, 'Wednesday') : 2,
            (12, 'Friday') : 1,
            (12, 'Saturday') : 1,
            # Week 13
            (13, 'Thursday') : 1,
            (13, 'Friday') : 1,
            # Week 14
            (14, 'Friday') : 1,
            (14, 'Sunday') : 1,
            # Week 15
            (15, 'Wednesday') : 1,
            (15, 'Thursday') : 1,
            # Finals
            (15, 'Friday') : 4,
            (16, 'Monday') : 4,
            (16, 'Tuesday') : 5,
            (16, 'Wednesday') : 2,
            (16, 'Thursday') : 3

        }

    def _get_future_day(self, current_week, current_day, days_ahead):
        """Helper to find the (week, day) for X days from now."""
        idx = self.days_of_week.index(current_day) + days_ahead
        new_week = current_week + (idx // 7)
        new_day = self.days_of_week[idx % 7]
        return new_week, new_day

    def _calculate_pressure(self, week, day):
        """Looks ahead 3 days, fetches exams, and returns the multiplier."""
        exams_list = []
        
        # Look at Today (0), Tomorrow (1), In 2 days (2), In 3 days (3)
        for days_ahead in range(4):
            target_week, target_day = self._get_future_day(week, day, days_ahead)
            exam_count = self.exam_schedule.get((target_week, target_day), 0)
            
            # Add an entry to the list for EVERY exam found that day
            exams_list.extend([days_ahead] * exam_count)
            
        # The squashing logic we built earlier
        raw_score = sum([(4 - d) for d in exams_list])
        THRESHOLD, MAX_BOOST, GROWTH_RATE = 4.0, 0.60, 0.25
        
        if raw_score <= THRESHOLD:
            return 1.0
        return 1.0 + (MAX_BOOST * (1 - math.exp(-GROWTH_RATE * (raw_score - THRESHOLD))))

    def predict_demand(self, week, day, hour):
        """The main prediction pipeline."""
        # 0. Correct week
        if week > 16: 
            week = 16
        elif week < 1:
            week = 1;
        
        # 1. Get the Pressure Multiplier
        pressure_multiplier = self._calculate_pressure(week, day)
        
        # 2. Fetch Baseline for this Day/Hour
        mask = (self.baseline['Day'] == day) & (self.baseline['Hour'] == hour)
        current_hour_df = self.baseline[mask].copy()
        
        if current_hour_df.empty:
            return "Invalid time parameters."

        # 3. Apply Multiplier & Calculate Raw Demand
        # Note: If base popularity is 0 (closed), demand stays 0
        current_hour_df['Raw_Demand'] = current_hour_df['Base_Popularity'] * pressure_multiplier
        
        # 4. Calculate Spillover
        raw_spillover = 0.0
        for idx, row in current_hour_df.iterrows():
            if row['Raw_Demand'] > 1.0:
                raw_spillover += (row['Raw_Demand'] - 1.0)
                current_hour_df.at[idx, 'Raw_Demand'] = 1.0 # Cap the building at 100%
        
        # 5. Route the Spillover (Reduced by 70% for people giving up/going to dorms)
        retained_spillover = raw_spillover * 0.30
        
        if retained_spillover > 0:
            for idx, row in current_hour_df.iterrows():
                b_name = row['Building']
                # Only distribute to buildings that are OPEN (Base > 0) and not full
                if row['Base_Popularity'] > 0 and row['Raw_Demand'] < 1.0:
                    weight = self.building_votes[b_name] / self.total_votes
                    share_of_spillover = retained_spillover * weight
                    
                    # Add spillover and enforce the 1.0 cap one last time
                    new_demand = min(1.0, row['Raw_Demand'] + share_of_spillover)
                    current_hour_df.at[idx, 'Raw_Demand'] = new_demand

        # Return results neatly
        results = current_hour_df[['Building', 'Raw_Demand']].set_index('Building').to_dict()['Raw_Demand']
        return {
            'Pressure_Multiplier': round(pressure_multiplier, 3),
            'Location_Demands': {k: round(v, 3) for k, v in results.items()}
        }

# ==========================================
# Example Usage
# ==========================================
# (Assuming df_melted is available from the previous step)
df_melted = pd.read_csv('./data/base_popularity_distributions.csv')

model = CampusDemandModel(baseline_df=df_melted)
print(model.predict_demand(week=16, day='Saturday', hour=14))