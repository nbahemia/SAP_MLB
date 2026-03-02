import pybaseball as pb
import pandas as pd

players = [
    ("Valdez", "Framber"),
    ("McCutchen", "Andrew"),
    ("Tauchman", "Mike"),
    ("Early", "Connelly"),
    ("Díaz", "Yandy"),
    ("Beeks", "Jalen"),
    ("Leclerc", "José"),
    ("Bradley", "Taj"),
    ("Correa", "Carlos")
]

id_list = []

for last, first in players:
    try:
        info = pb.playerid_lookup(last=last, first=first)
        if info.empty:
            print(f"⚠️ Could not find MLBAM ID for {first} {last}")
            continue
        mlbam_id = info['key_mlbam'].values[0]
        full_name = f"{first} {last}"
        id_list.append({"Player": full_name, "MLBAM_ID": mlbam_id})
        print(f"{full_name} → {mlbam_id}")
    except Exception as e:
        print(f"⚠️ Error looking up {first} {last}: {e}")

# Save all found IDs to CSV
df_ids = pd.DataFrame(id_list)
df_ids.to_csv("player_ids.csv", index=False)
print("\n✅ MLBAM IDs saved to player_ids.csv")