"""Set up COD test datasets with symlinks for PRNet evaluation."""
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = Path(os.path.expanduser("~/Documents/Python/datasets"))
TEST_DATASET_DIR = PROJECT_DIR / "TestDataset"


def setup_camo():
    """CAMO: 250 test images + 250 GT masks (filtered from 1250)."""
    camo_dir = DATASETS_DIR / "CAMO-V.1.0-CVIU2019"
    out_dir = TEST_DATASET_DIR / "CAMO"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Symlink test images
    img_target = camo_dir / "Images" / "Test"
    (out_dir / "Imgs").symlink_to(img_target, target_is_directory=True)

    # GT: filter only the 250 test image IDs
    gt_out = out_dir / "GT"
    gt_out.mkdir(parents=True, exist_ok=True)
    gt_src = camo_dir / "GT"

    test_stems = set()
    for f in os.listdir(img_target):
        if f.endswith(".jpg"):
            test_stems.add(f.replace(".jpg", ""))

    linked = 0
    for f in sorted(os.listdir(gt_src)):
        stem = f.replace(".png", "")
        if stem in test_stems:
            src = gt_src / f
            dst = gt_out / f
            if not dst.exists():
                dst.symlink_to(src)
                linked += 1

    print(f"CAMO: {linked} GT symlinks created for {len(test_stems)} test images")


def setup_cod10k():
    """COD10K: 4000 test images + 4000 GT masks."""
    cod_dir = DATASETS_DIR / "COD10K-v3"
    out_dir = TEST_DATASET_DIR / "COD10K"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "Imgs").symlink_to(cod_dir / "Test" / "Image", target_is_directory=True)
    (out_dir / "GT").symlink_to(cod_dir / "Test" / "GT_Object", target_is_directory=True)
    print("COD10K: symlinks created")


def setup_nc4k():
    """NC4K: extracted from HuggingFace parquet -> test split."""
    nc4k_dir = DATASETS_DIR / "NC4K_extracted" / "test"
    out_dir = TEST_DATASET_DIR / "NC4K"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "Imgs").symlink_to(nc4k_dir / "Imgs", target_is_directory=True)
    (out_dir / "GT").symlink_to(nc4k_dir / "GT", target_is_directory=True)
    print("NC4K: symlinks created")


def main():
    TEST_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    setup_camo()
    setup_cod10k()
    setup_nc4k()

    print(f"\nTestDataset directory: {TEST_DATASET_DIR}")
    for d in sorted(TEST_DATASET_DIR.iterdir()):
        if d.is_dir():
            imgs = d / "Imgs"
            gts = d / "GT"
            img_count = len(os.listdir(imgs)) if imgs.exists() else 0
            gt_count = len(os.listdir(gts)) if gts.exists() else 0
            print(f"  {d.name}: {img_count} images, {gt_count} GT masks")


if __name__ == "__main__":
    main()
