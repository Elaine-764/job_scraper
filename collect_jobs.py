import pandas as pd
import glob
import json
from datetime import date

# Collect all JSON entries from files under roles/
json_files = glob.glob('roles/*.json')
data = []
for jf in json_files:
	try:
		with open(jf, 'r', encoding='utf-8') as fh:
			entries = json.load(fh)
			if isinstance(entries, dict):
				entries = [entries]
			data.extend(entries)
	except Exception as e:
		print(f"Failed to read {jf}: {e}")

# Normalize so each listing becomes one row and include the company
if data:
	df = pd.json_normalize(data, record_path='listings', meta=['company'], errors='ignore')
else:
	df = pd.DataFrame(columns=['company'])

print(df.head())
today = date.today()
output_path = f"all_roles_{today.isoformat().replace('-', '')}.csv"
df.to_csv(output_path, index=False)