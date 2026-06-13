"""Setup training data for PRNet by combining COD10K + CAMO."""
import os
import shutil
import glob

BASE = '/home/chen/Documents/Python/datasets'
TARGET = '/home/chen/Documents/Python/COD/PRNet/TrainDataset'

os.makedirs(f'{TARGET}/Imgs', exist_ok=True)
os.makedirs(f'{TARGET}/GT', exist_ok=True)

# 1. COD10K Train (6000 images)
cod_img_dir = f'{BASE}/COD10K-v3/Train/Image'
cod_gt_dir = f'{BASE}/COD10K-v3/Train/GT_Object'

for f in os.listdir(cod_img_dir):
    src = os.path.join(cod_img_dir, f)
    dst = os.path.join(TARGET, 'Imgs', f)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    # GT: same stem but .png
    stem = os.path.splitext(f)[0]
    gt_src = os.path.join(cod_gt_dir, stem + '.png')
    gt_dst = os.path.join(TARGET, 'GT', stem + '.png')
    if os.path.exists(gt_src) and not os.path.exists(gt_dst):
        os.symlink(gt_src, gt_dst)

# 2. CAMO Train (1000 images)
camo_img_dir = f'{BASE}/CAMO-V.1.0-CVIU2019/Images/Train'
camo_gt_dir = f'{BASE}/CAMO-V.1.0-CVIU2019/GT'

for f in os.listdir(camo_img_dir):
    if not f.lower().endswith(('.jpg', '.png')):
        continue
    src = os.path.join(camo_img_dir, f)
    stem = os.path.splitext(f)[0]
    dst_name = f'CAMO-{stem}.jpg'
    dst = os.path.join(TARGET, 'Imgs', dst_name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
    for ext in ['.png', '.jpg']:
        gt_src = os.path.join(camo_gt_dir, stem + ext)
        if os.path.exists(gt_src):
            gt_dst = os.path.join(TARGET, 'GT', f'CAMO-{stem}.png')
            if not os.path.exists(gt_dst):
                os.symlink(gt_src, gt_dst)
            break

img_count = len(os.listdir(f'{TARGET}/Imgs'))
gt_count = len(os.listdir(f'{TARGET}/GT'))
print(f'Training data ready: {img_count} images, {gt_count} GT')
