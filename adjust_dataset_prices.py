import os
import sys
import pandas as pd


def main():
	if len(sys.argv) > 1:
		try:
			scale = float(sys.argv[1])
		except Exception:
			scale = float(os.environ.get("PRICE_SCALE", 1.5))
	else:
		scale = float(os.environ.get("PRICE_SCALE", 1.5))

	source_path = "latestnewdataset.csv"
	dest_path = "latestnewdataset_adjusted.csv"

	df = pd.read_csv(source_path)
	if "Price (Lakhs)" not in df.columns:
		raise ValueError("Missing 'Price (Lakhs)' column in dataset")

	before_min = df["Price (Lakhs)"].min()
	before_mean = df["Price (Lakhs)"].mean()
	before_max = df["Price (Lakhs)"].max()

	df["Price (Lakhs)"] = df["Price (Lakhs)"] * scale

	after_min = df["Price (Lakhs)"].min()
	after_mean = df["Price (Lakhs)"].mean()
	after_max = df["Price (Lakhs)"].max()

	df.to_csv(dest_path, index=False)

	print(f"Scaled dataset saved to {dest_path}")
	print(f"Scale factor: {scale}")
	print(f"Price (Lakhs) before -> min: {before_min:.2f}, mean: {before_mean:.2f}, max: {before_max:.2f}")
	print(f"Price (Lakhs) after  -> min: {after_min:.2f},  mean: {after_mean:.2f},  max: {after_max:.2f}")


if __name__ == "__main__":
	main()


