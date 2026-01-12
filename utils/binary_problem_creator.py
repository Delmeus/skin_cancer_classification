import pandas as pd

input_file = '../dataset/GroundTruth.csv'
output_file = '../dataset/labels.csv'

df = pd.read_csv(input_file)

# Conditions considered "sick"
sick_conditions = ['MEL', 'BCC', 'AKIEC']

# Initialize as not sick
df['sick'] = 0

# Mark as sick if ANY sick condition is present
df.loc[df[sick_conditions].any(axis=1), 'sick'] = 1

# Keep only required columns
df = df[['image', 'sick']]

df.to_csv(output_file, index=False)

print(f"New CSV file saved as {output_file}")
