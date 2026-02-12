import pybaseball as pb
import numpy as np

data = pb.fg_pitching_data(start_season=2020, end_season=2020, qual=50)

minnesota_twins_fg_data = data[data['Team'] == 'MIN']

print(minnesota_twins_fg_data[['Name', 'ERA', 'IP']].head())

player_id = pb.playerid_lookup(last='Judge', first='Aaron')
print(player_id['key_mlbam'].values[0])
#statcast_pitcher_data = pb.statcast_pitcher(start_dt="2020-01-01", end_dt="2020-12-31", player_id = )
