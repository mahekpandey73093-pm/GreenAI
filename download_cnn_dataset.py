from datasets import load_dataset
from pathlib import Path
from PIL import Image
import random

# =====================================================
# GreenAI - EuroSAT RGB CNN Dataset
# Green vs Urban
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
CNN_DIR = BASE_DIR / "cnn"

TRAIN_DIR = CNN_DIR / "train"
TEST_DIR = CNN_DIR / "test"

GREEN_TRAIN = TRAIN_DIR / "green"
URBAN_TRAIN = TRAIN_DIR / "urban"
GREEN_TEST = TEST_DIR / "green"
URBAN_TEST = TEST_DIR / "urban"

for folder in [
    GREEN_TRAIN,
    URBAN_TRAIN,
    GREEN_TEST,
    URBAN_TEST
]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("       GreenAI - EuroSAT RGB Dataset")
print("=" * 55)

print("\nDownloading EuroSAT RGB dataset...")
print("This is RGB only, NOT multispectral.")
print("Please wait...\n")

dataset = load_dataset(
    "giswqs/EuroSAT_RGB",
    split="train"
)

print("Dataset downloaded successfully!")
print("Total images:", len(dataset))

# -----------------------------------------------------
# Class mapping
# -----------------------------------------------------

GREEN_CLASSES = {
    "Forest",
    "HerbaceousVegetation",
    "Pasture",
    "PermanentCrop"
}

URBAN_CLASSES = {
    "Highway",
    "Industrial",
    "Residential"
}

print("\nGreen classes:")
print(GREEN_CLASSES)

print("\nUrban classes:")
print(URBAN_CLASSES)

# -----------------------------------------------------
# Collect relevant samples
# -----------------------------------------------------

green_samples = []
urban_samples = []

for i, sample in enumerate(dataset):

    label = sample["label"]

    # label can be integer or class name depending on dataset version
    if isinstance(label, int):
        class_name = dataset.features["label"].names[label]
    else:
        class_name = str(label)

    if class_name in GREEN_CLASSES:
        green_samples.append((i, class_name))

    elif class_name in URBAN_CLASSES:
        urban_samples.append((i, class_name))

print("\nRelevant images found:")
print("Green :", len(green_samples))
print("Urban :", len(urban_samples))

# -----------------------------------------------------
# Balance classes
# -----------------------------------------------------

random.seed(42)

# We don't need all 27,000 images.
# Keep a manageable balanced dataset.

MAX_PER_CLASS = 1000

random.shuffle(green_samples)
random.shuffle(urban_samples)

green_samples = green_samples[:MAX_PER_CLASS]
urban_samples = urban_samples[:MAX_PER_CLASS]

print("\nSelected:")
print("Green :", len(green_samples))
print("Urban :", len(urban_samples))

# -----------------------------------------------------
# Save images
# -----------------------------------------------------

def save_samples(samples, train_folder, test_folder, prefix):

    random.shuffle(samples)

    split_index = int(len(samples) * 0.8)

    train_samples = samples[:split_index]
    test_samples = samples[split_index:]

    print(f"\n{prefix}")
    print("Train:", len(train_samples))
    print("Test :", len(test_samples))

    # Train
    for count, (idx, class_name) in enumerate(train_samples):

        image = dataset[idx]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        output_path = train_folder / f"{prefix.lower()}_{count:04d}.jpg"
        image.save(output_path, quality=95)

    # Test
    for count, (idx, class_name) in enumerate(test_samples):

        image = dataset[idx]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        output_path = test_folder / f"{prefix.lower()}_{count:04d}.jpg"
        image.save(output_path, quality=95)


save_samples(
    green_samples,
    GREEN_TRAIN,
    GREEN_TEST,
    "GREEN"
)

save_samples(
    urban_samples,
    URBAN_TRAIN,
    URBAN_TEST,
    "URBAN"
)

# -----------------------------------------------------
# Final summary
# -----------------------------------------------------

print("\n" + "=" * 55)
print("        DATASET PREPARATION COMPLETE")
print("=" * 55)

print("\nDataset location:")
print(CNN_DIR)

print("\nFinal structure:")
print("""
dataset/
└── cnn/
    ├── train/
    │   ├── green/
    │   └── urban/
    └── test/
        ├── green/
        └── urban/
""")

print("Green training images :", len(list(GREEN_TRAIN.glob("*.jpg"))))
print("Urban training images :", len(list(URBAN_TRAIN.glob("*.jpg"))))
print("Green testing images  :", len(list(GREEN_TEST.glob("*.jpg"))))
print("Urban testing images  :", len(list(URBAN_TEST.glob("*.jpg"))))

print("\nCNN dataset is ready!")