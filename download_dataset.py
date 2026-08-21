from datasets import load_dataset

print("===================================")
print("       GreenAI Dataset Loader")
print("===================================")

print("\nLoading EuroSAT Multispectral dataset...")
print("Please wait...")

dataset = load_dataset("giswqs/EuroSAT_MS")

print("\nDataset loaded successfully!")
print(dataset)

print("\nAvailable splits:")

for split in dataset:
    print(split, ":", len(dataset[split]), "samples")