import pandas as pd
import json

# Read the JSON data
with open('../data/IDS.json', 'r') as file:
    data = json.load(file)

# Create DataFrames for X and Y
df_x = pd.DataFrame(data['X'], columns=['Season', 'Episode', 'Chunk'])
df_y = pd.DataFrame(data['Y'], columns=['Ex', 'Ey', 'Px', 'Py'])

# Combine X and Y DataFrames
df_combined = pd.concat([df_x, df_y], axis=1)

# Write the combined DataFrame to a CSV file
df_combined.to_csv('../data/IDS.csv', index=False)
