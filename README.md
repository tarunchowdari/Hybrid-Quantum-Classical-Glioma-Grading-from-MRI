# Hybrid Quantum–Classical Glioma Grading from MRI

A reimplementation of a published hybrid quantum-classical MRI classifier [1], plus the check I felt the paper was missing:
does the quantum circuit help at all compared with a classical layer of the same size?

**Short version:** I built a working hybrid model that reaches about 88-89% patient-level accuracy (AUC about 0.93) on BraTS 2020.
That is below the 94.12% reported in [1]. With a matched classical layer in place of the circuit, I saw no evidence that the
circuit adds anything. Everything is run on a simulator; nothing here says anything about real quantum hardware.

I'm a student and built this to understand how these models actually work, so the write-up is deliberately plain about what did
and didn't work.

## The task

Classify a glioma as high-grade (HGG) or low-grade (LGG) from four MRI modalities (T1, T1ce, T2, FLAIR).

## Data

- **BraTS 2020 training set** [2-5]: 369 patients (293 HGG, 76 LGG). Grades come from the `name_mapping.csv` that ships with the
  training data. The data is not included in this repo; it can be obtained through the BraTS challenge [5]. The BraTS organisers
  ask that [2], [3] and [4] be cited when using it.
- **Why not BraTS 2021, which the paper names?** As far as I can tell, that release ships tumour segmentation masks but not
  HGG/LGG grades, and I couldn't find where the paper says its grade labels come from. BraTS 2020 has the same four modalities and
  does include grades, so I used it. My numbers are therefore not a like-for-like replication of [1].
- **What "HGG" and "LGG" mean here:** in the BraTS releases, HGG is, as far as I know, essentially glioblastoma and LGG is the
  lower grades grouped together; one analysis of the dataset makes this point [10]. So this is not the clinical WHO low-vs-high split.

## Pipeline (`preprocess_brats.py`)

1. Z-score each modality over the brain (non-zero) voxels of the volume.
2. Per patient, keep up to 16 axial slices with the most tumour (at least 100 tumour pixels).
3. Crop a square box around the tumour (6 px margin), resize to 64x64, and stack the 4 modalities as channels.
4. Result: 5,904 slices (4,688 HGG / 1,216 LGG), shape `(N, 4, 64, 64)`.
5. Split **by patient**: 15% of patients (56) are held out as a test set; the rest go into 5 stratified folds (seed 42). No patient's
   slices appear on both sides of any split.

The crops use the ground-truth tumour mask, as [1] also does. A real system would need a separate tumour detector first.

## How the hybrid model works

1. An **encoder** turns the image into 2^n numbers (n = number of qubits). Either fixed PCA (what [1] describes) or a small learned CNN (my addition).
2. **Amplitude encoding** [9]: the 2^n numbers are normalised to unit length and used as the amplitudes of an n-qubit state.
3. A **circuit** applies L layers of Ry rotations (trainable angles) on every qubit, each followed by a ring of CZ gates.
4. We **read out** the expectation value of Pauli-Z on each qubit, giving n numbers between -1 and 1.
5. A small classical **head** turns those n numbers into an HGG/LGG probability.

The circuit is a statevector simulator I wrote in PyTorch [7] (`quantum_hybrid.py`). Ry and CZ only ever produce real amplitudes,
so it is ordinary real tensor algebra that trains with normal autograd on the GPU. `python quantum_hybrid.py --check` compares it
against a brute-force dense-matrix simulation and against PennyLane [6]; the maximum difference is about 1e-16.

To test whether the circuit matters, the **classical core** option swaps the circuit for a `Linear(2^n -> n) + tanh` layer with
a similar parameter count, same input and same readout size.

Training: 30 epochs, AdamW + OneCycle, batch 64, class-weighted BCE. The circuit's rotation angles get their own learning rate (0.05).
Slice probabilities are averaged per patient, and **patient-level** numbers are the ones reported below. Classical tooling:
scikit-learn [8].

## Results

### Cross-validation, patient level (5 folds x 3 seeds, mean +- std)

| Model | Accuracy | Balanced acc. | AUC |
|---|---|---|---|
| CNN alone (23.9K params) | 0.883 +- 0.030 | 0.840 +- 0.040 | 0.929 +- 0.038 |
| CNN -> quantum circuit (28.0K) | 0.881 +- 0.025 | 0.831 +- 0.037 | 0.933 +- 0.041 |
| CNN -> classical layer (28.4K) | 0.891 +- 0.022 | 0.847 +- 0.046 | 0.936 +- 0.037 |

Compared fold by fold (same folds and seeds), quantum minus classical was about -1.1 points in accuracy (the quantum model won
2 of 15 pairs), -1.6 in balanced accuracy (3 of 15) and -0.3 in AUC (7 of 15). Those 15 pairs share folds, so I wouldn't treat the
small gap as proven. The fair summary is a tie, if anything slightly in the classical model's favour.

### PCA input (the setup described in [1]), one seed each

| Model | Params | Accuracy | Balanced acc. | AUC |
|---|---|---|---|---|
| PCA(64) + logistic regression | | 0.818 | 0.762 | 0.842 |
| PCA -> quantum, 6 qubits, 3 layers | 25 | 0.674 | 0.667 | 0.659 |
| PCA -> quantum, 6 qubits, 3 layers + 16-unit head | 147 | 0.683 | 0.661 | 0.707 |
| PCA -> quantum, 8 qubits, 6 layers + 16-unit head | 209 | 0.639 | 0.525 | 0.546 |
| PCA -> classical, 6-qubit-equivalent | 397 | 0.834 | 0.772 | 0.841 |
| PCA -> classical, 6-qubit-equivalent + 16-unit head | 519 | 0.831 | 0.742 | 0.800 |
| PCA -> classical, 8-qubit-equivalent + 16-unit head | 2,217 | 0.834 | 0.738 | 0.816 |

With PCA input the circuit was clearly worse than a classical layer, and making it bigger did not help. The quantum models'
training loss stayed high (about 0.19-0.24) while the classical ones fit the training set almost perfectly, so the circuit is
under-fitting, not over-fitting. One fix along the way: my first attempt started the rotation angles near zero, which throws away
most of the input information. The numbers above use wide initialisation and a separate circuit learning rate; the earlier runs are
kept in `results/superseded/`. I did not diagnose the 8-qubit result beyond this.

### Held-out test set (56 patients, about 12 of them LGG; scored once, after fixing the models)

| Model | Accuracy | Balanced acc. | AUC |
|---|---|---|---|
| CNN alone | 0.893 | 0.841 | 0.907 |
| CNN -> quantum circuit | 0.875 | 0.860 | 0.953 |
| CNN -> classical layer | 0.857 | 0.758 | 0.907 |

This is a very small test set: the three accuracies are 50, 49 and 48 patients correct out of 56, and one LGG patient moves balanced
accuracy by about 4 points. The ordering even differs from cross-validation. I read it as "all three are consistent with each
other", not as the quantum model winning. These values are transcribed from console output into `results/heldout_test.json`.

## What I take from this

- A learned CNN encoder does the heavy lifting: going from PCA input to a CNN encoder moved AUC from about 0.82-0.84 to about 0.93.
- Under patient-level evaluation, the hybrid model is not better than a plain CNN, and the quantum circuit is not better than a matched classical layer.
- I did not reproduce the 94.12% accuracy in [1]. I can't say why: the data differs (see above) and I didn't try to match every
  detail of their training setup.

## Things in the paper I couldn't reconcile

These are from reading the paper text [1]; I may be missing context.

- It describes a flattened 4,096-dim input (64x64) with amplitude encoding, which needs 12 qubits, but also says "4-8 qubits".
  PCA is described as optional and the number of components is never given.
- It reports "~4,000 ROI slices (balanced across classes)", while BraTS-style grade data is strongly imbalanced at patient level.
- Table 9 lists the proposed model at about 14K trainable parameters and 38.2 ms/sample; Table 12 lists 3.2M parameters and 6.4 ms/image.
- Accuracy SD is +-1.08 in Table 6 but +-0.87 in Table 10. Precision and recall in Table 6 (92.75 / 93.86) look swapped relative
  to Table 10 (93.85 / 92.78). The best classical baseline is a CNN at 90.34% in one table and 91.80% in another.
- The "entirely classical" ablation is a small dense network on the same compressed input, not a tuned CNN.

The paper does state that it used patient-aware 5-fold cross-validation, so I'm not claiming leakage.

## Limitations

- Crops use ground-truth masks; there is no tumour-detection step.
- One public dataset, 2D slices only, and one held-out test set of 56 patients.
- The PCA-input runs and the test-set runs are single-seed.
- Simulation only: no hardware, no noise model.
- Tuning of the quantum path was limited to the attempts listed above.

## Reproduce

```
pip install -r requirements.txt

python preprocess_brats.py --root "<path to BraTS2020>" --out data_64
python baselines.py --model pca_lr --pca 64
python baselines.py --model cnn --epochs 30 --seeds 3
python quantum_hybrid.py --check
python quantum_hybrid.py --encoder cnn --core quantum   --qubits 6 --layers 3 --seeds 3
python quantum_hybrid.py --encoder cnn --core classical --qubits 6 --seeds 3
# held-out test set, once, at the end: add --eval-test
```

`--root` should contain the `BraTS2020_TrainingData` folder. `python baselines.py --model pca_lr` does not need PyTorch. Each run writes a JSON to `results/`. The result files stored in this
repo are in `results/cv/` (final cross-validation runs), `results/superseded/` (earlier runs, kept for transparency) and
`results/heldout_test.json`. Result files without a `qlr` field were run before the circuit-initialisation fix; that fix does not
affect the classical models. The stored results were produced by an earlier version of these scripts, before I merged
duplicated code into `common.py` and `torch_utils.py`; I checked that the current scripts give identical results to that
version on the same data and seeds.

## Repo contents

- `preprocess_brats.py` - cropping, normalisation, patient-level splits
- `baselines.py` - PCA + logistic regression and the CNN baseline
- `quantum_hybrid.py` - circuit simulator, hybrid models, ablations, simulator self-test (`--check`)
- `common.py` - data loading, patient-level metrics, cross-validation driver (no PyTorch needed)
- `torch_utils.py` - shared CNN backbone, augmentation and training loop
- `results/` - per-run JSON outputs

## References

1. A. Harshavardhan, V. C. S. Rao, Y. M. Reddy, S. R. Polamuri, B. Jamalpur, V. L. Reddy. "Hybrid quantum-classical learning for MRI-based brain tumour diagnosis." *Discover Computing* 29, 269 (2026). https://doi.org/10.1007/s10791-026-10165-2
2. B. H. Menze et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." *IEEE Transactions on Medical Imaging* 34(10), 1993-2024 (2015). https://doi.org/10.1109/TMI.2014.2377694
3. S. Bakas et al. "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features." *Scientific Data* 4, 170117 (2017). https://doi.org/10.1038/sdata.2017.117
4. S. Bakas et al. "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge." arXiv:1811.02629 (2018). https://arxiv.org/abs/1811.02629
5. Multimodal Brain Tumor Segmentation Challenge 2020: Data. https://www.med.upenn.edu/cbica/brats2020/data.html
6. V. Bergholm et al. "PennyLane: Automatic differentiation of hybrid quantum-classical computations." arXiv:1811.04968 (2018). https://arxiv.org/abs/1811.04968
7. A. Paszke et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *Advances in Neural Information Processing Systems* 32 (2019). https://arxiv.org/abs/1912.01703
8. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research* 12, 2825-2830 (2011).
9. M. Schuld and F. Petruccione. *Supervised Learning with Quantum Computers.* Springer (2018). https://doi.org/10.1007/978-3-319-96424-9
10. "Exploring Radiologic Criteria for Glioma Grade Classification on the BraTS Dataset" (2021). https://www.em-consulte.com/article/1443114
