"""Preprocess the BraTS 2020 training data for HGG-vs-LGG slice classification.

Per patient:
  1. Find the four modalities (T1, T1ce, T2, FLAIR) and the segmentation mask.
  2. Z-score each modality over the non-zero (brain) voxels of the volume.
  3. Keep the axial slices with the most tumour (at most --max-slices, each with >= --min-area tumour pixels).
  4. Crop a square box around the tumour in each slice and resize it to --size x --size.
Then assign every PATIENT to the held-out test set or to one of --folds cross-validation folds, stratified
by grade, so slices from one patient never end up on both sides of a split.

Output (in --out):
  slices.npz    X (N, 4, S, S) float16, y (N,) with 1 = HGG and 0 = LGG, patient (N,), z (N,), fold (N,)
                fold: -1 = held-out test patient, 0..K-1 = cross-validation fold
  patients.csv  patient id, grade, number of slices, fold
  preview.png   a few example crops, as a sanity check

Usage
  python preprocess_brats.py --root /path/to/BraTS2020 --out data_64

--root is the folder that contains BraTS2020_TrainingData. Grades are read from the name_mapping.csv
that ships with the training data. The crops use the ground-truth tumour mask.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

MODALITIES = ["t1", "t1ce", "t2", "flair"]  # channel order in the output
TEST_FOLD = -1


# ----------------------------------------------------------------------------- finding the files
def strip_nifti_ext(name):
    """'x_t1.nii.gz' -> 'x_t1'; returns None for files that are not NIfTI."""
    lower = name.lower()
    for ext in (".nii.gz", ".nii"):
        if lower.endswith(ext):
            return lower[: -len(ext)]
    return None


def find_cases(root):
    """Return {case id: {modality or 'seg': path}} for every folder holding all four modalities and a mask."""
    cases = {}
    for dirpath, _, files in os.walk(root):
        found = {}
        for filename in files:
            stem = strip_nifti_ext(filename)
            if stem is None:
                continue
            path = Path(dirpath) / filename
            # "seg" also matches the oddly named mask of one patient ("..._Segm.nii").
            if "seg" in stem:
                found["seg"] = path
            elif stem.endswith("_t1ce"):
                found["t1ce"] = path
            elif stem.endswith("_flair"):
                found["flair"] = path
            elif stem.endswith("_t2"):
                found["t2"] = path
            elif stem.endswith("_t1"):
                found["t1"] = path
        if all(key in found for key in MODALITIES + ["seg"]):
            cases[Path(dirpath).name] = found
    return cases


def load_grades(root):
    """Read {patient id: 'HGG' | 'LGG'} from name_mapping.csv; returns (grades, csv path) or (None, None)."""
    csv_files = list(Path(root).rglob("name_mapping.csv"))
    if not csv_files:
        return None, None
    with open(csv_files[0], newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    columns = list(rows[0].keys())
    id_col = next((c for c in columns if "2020" in c and "id" in c.lower()), None)
    grade_col = next((c for c in columns if c.strip().lower() == "grade"), None)
    if id_col is None or grade_col is None:
        raise SystemExit(f"Unexpected columns in {csv_files[0]}: {columns}")
    grades = {r[id_col].strip(): r[grade_col].strip().upper() for r in rows if r[id_col].strip()}
    return grades, csv_files[0]


# ----------------------------------------------------------------------------- image processing
def zscore(volume):
    """Normalise to zero mean and unit variance over the non-zero voxels; background stays 0."""
    out = np.zeros(volume.shape, dtype=np.float32)
    brain = volume != 0
    if brain.sum() == 0:
        return out
    mean, std = volume[brain].mean(), volume[brain].std()
    out[brain] = (volume[brain] - mean) / (std + 1e-6)
    return out


def resize_hw(patch, size):
    """Bilinear resize of a (C, h, w) array to (C, size, size)."""
    _, h, w = patch.shape
    out = zoom(patch, (1, size / h, size / w), order=1)[:, :size, :size]  # slice: guards float rounding
    if out.shape[1] < size or out.shape[2] < size:
        out = np.pad(out, ((0, 0), (0, size - out.shape[1]), (0, size - out.shape[2])))
    return out


def process_case(files, size, max_slices, min_area, margin):
    """Return the cropped slices (4, size, size) of one patient and their axial slice indices."""
    tumour = np.asanyarray(nib.load(str(files["seg"])).dataobj) > 0
    volume = np.stack([zscore(np.asanyarray(nib.load(str(files[m])).dataobj).astype(np.float32))
                       for m in MODALITIES])
    height, width = tumour.shape[:2]
    areas = tumour.sum(axis=(0, 1))
    chosen = [int(z) for z in np.argsort(-areas) if areas[z] >= min_area][:max_slices]

    crops, slice_ids = [], []
    for z in sorted(chosen):
        rows, cols = np.where(tumour[:, :, z])
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        side = int(max(y1 - y0, x1 - x0) + 2 * margin)
        side = min(max(side, 16), height, width)
        top = int(np.clip((y0 + y1) // 2 - side // 2, 0, height - side))
        left = int(np.clip((x0 + x1) // 2 - side // 2, 0, width - side))
        patch = volume[:, top:top + side, left:left + side, z]
        crops.append(resize_hw(patch, size).astype(np.float16))
        slice_ids.append(z)
    return crops, slice_ids


# ----------------------------------------------------------------------------- splitting and reporting
def assign_folds(patients, grades, n_folds, test_frac, seed):
    """Stratified patient-level split: {patient id: fold}, with TEST_FOLD for the held-out test patients."""
    ids = sorted(set(patients))
    is_hgg = np.array([grades[p] == "HGG" for p in ids]).astype(int)
    train_val, test = train_test_split(ids, test_size=test_frac, stratify=is_hgg, random_state=seed)
    train_val_hgg = np.array([grades[p] == "HGG" for p in train_val]).astype(int)

    fold_of = {p: TEST_FOLD for p in test}
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(train_val, train_val_hgg)):
        for i in val_idx:
            fold_of[train_val[i]] = k
    return fold_of


def print_summary(X, y, patient, fold, n_folds):
    print("\n=== Summary ===")
    print(f"X shape: {X.shape}  (N, channels[{','.join(MODALITIES)}], H, W)")
    n_hgg_patients = len({p for p, label in zip(patient, y) if label == 1})
    print(f"Patients: HGG={n_hgg_patients}  LGG={len(set(patient.tolist())) - n_hgg_patients}")
    print(f"Slices:   HGG={int((y == 1).sum())}  LGG={int((y == 0).sum())}")
    for f in [TEST_FOLD] + list(range(n_folds)):
        mask = fold == f
        name = "test " if f == TEST_FOLD else f"fold{f}"
        print(f"  {name}: {len(set(patient[mask].tolist())):3d} patients, {int(mask.sum()):5d} slices, "
              f"HGG slices={int((y[mask] == 1).sum())}, LGG slices={int((y[mask] == 0).sum())}")


def save_preview(path, X, y, patient, z, seed):
    """Optional sanity-check figure: four HGG and four LGG FLAIR crops."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: preview skipped")
        return
    rng = np.random.default_rng(seed)
    picks = list(rng.choice(np.where(y == 1)[0], 4, replace=False)) + \
            list(rng.choice(np.where(y == 0)[0], 4, replace=False))
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(X[i, MODALITIES.index("flair")].astype(np.float32), cmap="gray")
        ax.set_title(f"{'HGG' if y[i] else 'LGG'} {patient[i][-3:]} z={z[i]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    print(f"Saved preview to {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder containing BraTS2020_TrainingData")
    ap.add_argument("--out", default="data_64")
    ap.add_argument("--size", type=int, default=64, help="side length of the output crops")
    ap.add_argument("--max-slices", type=int, default=16, help="max tumour slices kept per patient")
    ap.add_argument("--min-area", type=int, default=100, help="min tumour pixels for a slice to count")
    ap.add_argument("--margin", type=int, default=6, help="pixels of context around the tumour box")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.15, help="fraction of patients held out for testing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    train_root = root / "BraTS2020_TrainingData"
    if not train_root.exists():
        train_root = root
    print(f"Searching for cases under: {train_root}")

    cases = find_cases(train_root)
    print(f"Found {len(cases)} complete cases (4 modalities + segmentation)")
    if not cases:
        sys.exit("No complete cases found: expected folders with T1, T1ce, T2, FLAIR and segmentation NIfTI files.")

    grades, grades_path = load_grades(train_root)
    if grades is None:
        sys.exit("name_mapping.csv not found under the training folder (needed for the HGG/LGG grades).")
    print(f"Grades read from: {grades_path}")

    usable = [c for c in sorted(cases) if grades.get(c) in ("HGG", "LGG")]
    skipped = [c for c in sorted(cases) if c not in usable]
    if skipped:
        print(f"Skipping {len(skipped)} cases without an HGG/LGG label, e.g. {skipped[:3]}")
    if not usable:
        sys.exit("No case ID matched the grade table. Check the ID column of name_mapping.csv.")

    X, y, patient, z = [], [], [], []
    for case_id in tqdm(usable, desc="patients"):
        crops, slice_ids = process_case(cases[case_id], args.size, args.max_slices, args.min_area, args.margin)
        for crop, slice_id in zip(crops, slice_ids):
            X.append(crop)
            y.append(1 if grades[case_id] == "HGG" else 0)
            patient.append(case_id)
            z.append(slice_id)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    patient = np.array(patient)
    z = np.array(z, dtype=np.int16)

    fold_of = assign_folds(patient.tolist(), grades, args.folds, args.test_frac, args.seed)
    fold = np.array([fold_of[p] for p in patient], dtype=np.int8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "slices.npz", X=X, y=y, patient=patient, z=z, fold=fold)
    with open(out / "patients.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient", "grade", "n_slices", "fold"])
        for p in sorted(set(patient.tolist())):
            writer.writerow([p, grades[p], int((patient == p).sum()), fold_of[p]])

    print_summary(X, y, patient, fold, args.folds)
    save_preview(out / "preview.png", X, y, patient, z, args.seed)
    print(f"\nDone. Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
