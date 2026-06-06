import io
import urllib.request

import pandas as pd

url = "https://raw.githubusercontent.com/karen-pal/borges/master/datasets/full_corpus.csv"
print(f"Downloading {url}...")
try:
	response = urllib.request.urlopen(url)
	data = response.read().decode('utf-8')
	df = pd.read_csv(io.StringIO(data))
	print("Columns:", df.columns.tolist())
	print("Shape:", df.shape)
	print("\nSample row:")
	print(df.iloc[0].to_dict())
except Exception as e:
	print("Error:", e)
