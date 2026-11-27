
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd

# Dataset paths
DATA_DIR = Path(r"E:\Project\Brain Tumor\Epic and CSCR hospital Dataset")
TRAIN_DIR = DATA_DIR / "Train"
VAL_DIR = DATA_DIR / "Val"
TEST_DIR = DATA_DIR / "Test"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def create_validation_split():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

    # Prevent accidental overwriting of an existing validation set
    if VAL_DIR.exists() and any(VAL_DIR.rglob("*")):
        raise FileExistsError(
            f"{VAL_DIR} already contains files. "
            "Rename or back up this folder before creating a new split."
        )

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for class_name in CLASSES:
        class_dir = TRAIN_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) < 2:
            raise ValueError(
                f"Not enough images in {class_dir} to create a split."
            )

        records.extend(
            {"path": str(p), "label": class_name} for p in images
        )

    df = pd.DataFrame(records)

    # Allocate 15% of the original training images to validation
    train_records, val_records = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["label"]
    )

    # Create class folders
    for class_name in CLASSES:
        (VAL_DIR / class_name).mkdir(parents=True, exist_ok=True)

    # Move validation images from Train to Val
    for _, row in val_records.iterrows():
        source = Path(row["path"])
        destination = VAL_DIR / row["label"] / source.name
        shutil.move(str(source), str(destination))

    print("\nValidation split created successfully.")
    print(f"Training folder:   {TRAIN_DIR}")
    print(f"Validation folder: {VAL_DIR}")
    print(f"Test folder:       {TEST_DIR}")
    print("\nValidation images per class:")

    for class_name in CLASSES:
        count = len(list((VAL_DIR / class_name).glob("*")))
        print(f"{class_name}: {count}")

if __name__ == "__main__":
    create_validation_split()
