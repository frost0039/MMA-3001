import pandas as pd

# Load the datasets
env_data = pd.read_csv('data/5EnvSensor_MayToDec2024_180kRows.csv')
occ_data = pd.read_csv('data/5occupancySensor_MayToDec2024_9MRows.csv')

xls = pd.ExcelFile('data/Sensor ID and Locations.xlsx')
env_map_raw = pd.read_excel(xls, sheet_name='ENV')
occ_map_raw = pd.read_excel(xls, sheet_name='OCC (Incorrect Data - SpaceFloo')

# Clean and standardize the matching columns
env_data['sensorid'] = env_data['sensorid'].astype(str).str.strip()
env_map_raw['ID'] = env_map_raw['ID'].astype(str).str.strip()

occ_data['floorspaceid'] = occ_data['floorspaceid'].astype(str).str.strip().str.lower()
occ_map_raw['SpaceId'] = occ_map_raw['SpaceId'].astype(str).str.strip().str.lower()

# Map the physical Floor/Room to both datasets
env_map = env_map_raw.rename(columns={'ID': 'sensorid'})
occ_map = occ_map_raw.rename(columns={'SpaceId': 'floorspaceid'})

env_data = pd.merge(env_data, env_map[['sensorid', 'Floor/Room']], on='sensorid', how='inner')
occ_data = pd.merge(occ_data, occ_map[['floorspaceid', 'Floor/Room']], on='floorspaceid', how='inner')

print(f"Rows after Env Mapping: {len(env_data)}")
print(f"Rows after Occ Mapping: {len(occ_data)}")

# Clean the Room text to eliminate invisible Excel spaces or capitalization differences
env_data['Floor/Room'] = env_data['Floor/Room'].astype(str).str.strip().str.upper()
occ_data['Floor/Room'] = occ_data['Floor/Room'].astype(str).str.strip().str.upper()

# Standardize to UTC (Do not round—keep the exact millisecond for the proximity merge)
env_data['time'] = pd.to_datetime(env_data['createdate'], utc=True)
occ_data['time'] = pd.to_datetime(occ_data['occupancystatuschangedate'], utc=True)

# Sort both datasets chronologically (Mandatory for merge_asof)
env_data = env_data.sort_values('time')
occ_data = occ_data.sort_values('time')

# Isolate one occupancy room to create a clean target variable for the ML model
occ_data_g20 = occ_data[occ_data['Floor/Room'] == 'G.20'].copy()

# Sort both datasets chronologically (Mandatory for merge_asof)
env_data = env_data.sort_values('time')
occ_data_g20 = occ_data_g20.sort_values('time')

# Merge without the strict 'by' rule, relying purely on time proximity
final_dataset = pd.merge_asof(
    env_data, 
    occ_data_g20, 
    on='time', 
    direction='nearest', 
    tolerance=pd.Timedelta('1h')
)

# Drop any environmental readings that had no occupancy data nearby
final_dataset = final_dataset.dropna(subset=['occupancystatus'])

print("\nFinal Dataset Shape:", final_dataset.shape)
print(final_dataset[['Floor/Room_x', 'Floor/Room_y', 'time', 'occupancystatus']].head())

final_dataset.to_csv('data/clean_merged_dataset.csv', index=False)