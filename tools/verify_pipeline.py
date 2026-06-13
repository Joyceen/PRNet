"""Verify data pipeline and model forward pass without checkpoint."""
import sys
sys.path.insert(0, '/home/chen/Documents/Python/COD/PRNet')
sys.path.append('/home/chen/Documents/Python/COD/PRNet/models')

import torch
from models.PRNet import PRNet
from data_cod import test_dataset

TEST_PATH = '/home/chen/Documents/Python/COD/PRNet/TestDataset/'
datasets = ['CAMO', 'COD10K', 'NC4K']

print("=" * 60)
print("PRNet Pipeline Verification")
print("=" * 60)

# 1. Verify datasets
print("\n[1] Checking datasets...")
for ds in datasets:
    try:
        loader = test_dataset(f'{TEST_PATH}{ds}/Imgs/', f'{TEST_PATH}{ds}/GT/', 384)
        print(f"  {ds}: {loader.size} samples OK")
        img, gt, name, _ = loader.load_data()
        print(f"    Image shape: {img.shape}, GT shape: {gt.shape}, name: {name}")
    except Exception as e:
        print(f"  {ds}: ERROR - {e}")

# 2. Verify model initialization
print("\n[2] Initializing PRNet model...")
try:
    model = PRNet()
    print("  PRNet initialized successfully")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# 3. Verify forward pass
print("\n[3] Testing forward pass...")
try:
    model.cuda()
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 384).cuda()
    with torch.no_grad():
        out = model(dummy_input)
    print(f"  Output shapes: s1={out[0].shape}, s2={out[1].shape}, s3={out[2].shape}, s4={out[3].shape}")
    print("  Forward pass OK")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Verify end-to-end on first image of each dataset
print("\n[4] Testing end-to-end inference...")
for ds in datasets:
    try:
        loader = test_dataset(f'{TEST_PATH}{ds}/Imgs/', f'{TEST_PATH}{ds}/GT/', 384)
        img, gt, name, _ = loader.load_data()
        img_tensor = img.cuda()
        with torch.no_grad():
            s1, s2, s3, s4 = model(img_tensor)
        print(f"  {ds}: {name} -> prediction shape {s1.shape}")
    except Exception as e:
        print(f"  {ds}: ERROR - {e}")

print("\n" + "=" * 60)
print("Pipeline verification complete!")
print("=" * 60)
