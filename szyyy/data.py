# -*- coding: utf-8 -*-
"""
Convert Hyperspectral MAT dataset to nnU-Net v2 format
原始图像: xxx.mat -> 标签: xxxlabels.mat
图像: (1024,1280,60)，标签: (1024,1280)，取值 {0,1}
输出: nnUNet_raw/Dataset501_DGA/
"""
import os
import json
import shutil
import numpy as np
import nibabel as nib
import scipy.io as sio
from sklearn.model_selection import train_test_split

def convert_mat_to_nii(mat_path, nii_path, is_label=False, target_depth=None):
    """把 .mat 文件转换成 .nii.gz"""
    data = sio.loadmat(mat_path)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    arr = data[key]

    if not is_label:
        # [H, W, C] → [C, H, W]，nnUNet 默认通道在前
        arr = np.transpose(arr, (2, 0, 1))
    else:
        arr = arr.astype(np.uint8)
        if target_depth is not None and arr.ndim == 2:
            # 扩展到目标深度
            arr = np.repeat(arr[np.newaxis, :, :], target_depth, axis=0)

    img = nib.Nifti1Image(arr, np.eye(4))
    nib.save(img, nii_path)
    return arr.shape

def main():
    # ---------------- 参数配置 ----------------
    image_dir = "/data/CXY/gwj/fanyan/data/DGA/IMAGE_UNIFIED"
    label_dir = "/home/ubuntu/dataset_Med/DGA/DGA_label"
    out_root = "/data/CXY/g/szy/data/nnUNet_raw/Dataset518_XIAN"

    split_ratio = 0.15  # 测试集比例
    os.makedirs(out_root, exist_ok=True)

    imagesTr, imagesTs = os.path.join(out_root, "imagesTr"), os.path.join(out_root, "imagesTs")
    labelsTr, labelsTs = os.path.join(out_root, "labelsTr"), os.path.join(out_root, "labelsTs")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(imagesTs, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    os.makedirs(labelsTs, exist_ok=True)

    # ---------------- 匹配图像和标签 ----------------
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".mat")])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".mat")])
    assert len(image_files) == len(label_files), "图像和标签数量不一致！"
    pairs = [(os.path.join(image_dir, i), os.path.join(label_dir, j))
             for i, j in zip(image_files, label_files)]

    # ---------------- 划分训练/测试 ----------------
    train_pairs, test_pairs = train_test_split(pairs, test_size=split_ratio, random_state=42)

    # ---------------- 转换并保存 ----------------
    for img_path, lbl_path in train_pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_out = os.path.join(imagesTr, f"{base}_0000.nii.gz")
        lbl_out = os.path.join(labelsTr, f"{base}.nii.gz")

        img_shape = convert_mat_to_nii(img_path, img_out, is_label=False)
        # label 扩展到图像深度
        convert_mat_to_nii(lbl_path, lbl_out, is_label=True, target_depth=img_shape[0])

        print(f"Saved {img_out} | shape: {img_shape}")
        print(f"Saved {lbl_out} | shape: {(img_shape[0],) + img_shape[1:]}")

    for img_path, lbl_path in test_pairs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_out = os.path.join(imagesTs, f"{base}_0000.nii.gz")
        lbl_out = os.path.join(labelsTs, f"{base}.nii.gz")

        img_shape = convert_mat_to_nii(img_path, img_out, is_label=False)
        convert_mat_to_nii(lbl_path, lbl_out, is_label=True, target_depth=img_shape[0])

        print(f"Saved {img_out} | shape: {img_shape}")
        print(f"Saved {lbl_out} | shape: {(img_shape[0],) + img_shape[1:]}")

    # ---------------- 生成 dataset.json ----------------
    dataset = {
        "channel_names": {
            "0": "HSI"
        },
        "labels": {
            "background": 0,
            "target": 1
        },
        "numTraining": len(train_pairs),
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(out_root, "dataset.json"), "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"✅ 数据准备完成！结果保存在 {out_root}")

if __name__ == "__main__":
    main()
