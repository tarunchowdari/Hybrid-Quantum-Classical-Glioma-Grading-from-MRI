import argparse

import numpy as np

from common import run_experiment


def make_pca_lr(args):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_train, y_train, X_val, seed):
        A = X_train.reshape(len(X_train), -1).astype(np.float32)
        B = X_val.reshape(len(X_val), -1).astype(np.float32)
        pca = PCA(args.pca, svd_solver="randomized", random_state=seed).fit(A)
        A, B = pca.transform(A), pca.transform(B)
        scaler = StandardScaler().fit(A)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=args.C)
        clf.fit(scaler.transform(A), y_train)
        return clf.predict_proba(scaler.transform(B))[:, 1], None

    return fit_predict


def make_cnn(args):
    import torch.nn as nn
    from torch_utils import conv_stack, count_params, fit_and_predict, get_device, seed_everything

    class SmallCNN(nn.Module):
        def __init__(self, width, dropout=0.3):
            super().__init__()
            self.f, channels = conv_stack(width)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 1))

        def forward(self, x):
            return self.head(self.f(x).mean((2, 3))).squeeze(1)

    device = get_device()

    def fit_predict(X_train, y_train, X_val, seed):
        seed_everything(seed)
        model = SmallCNN(args.width).to(device)
        probs = fit_and_predict(model, X_train, y_train, X_val, epochs=args.epochs, batch_size=args.bs,
                                lr=args.lr, device=device, augment_inputs=True, log_every=5)
        return probs, count_params(model)

    return fit_predict


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_64/slices.npz")
    ap.add_argument("--model", choices=["pca_lr", "cnn"], required=True)
    ap.add_argument("--pca", type=int, default=64, help="pca_lr: number of principal components")
    ap.add_argument("--C", type=float, default=0.1, help="pca_lr: inverse regularisation strength")
    ap.add_argument("--epochs", type=int, default=30, help="cnn only")
    ap.add_argument("--bs", type=int, default=64, help="cnn only: batch size")
    ap.add_argument("--lr", type=float, default=2e-3, help="cnn only: peak learning rate")
    ap.add_argument("--width", type=int, default=16, help="cnn only: channels in the first conv block")
    ap.add_argument("--seeds", type=int, default=1, help="repeat the whole CV with this many seeds")
    ap.add_argument("--eval-test", action="store_true",
                    help="train on all CV patients and score the held-out test patients (use ONCE)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.model == "pca_lr":
        fit_predict, tag = make_pca_lr(args), f"pca_lr_pca{args.pca}"
    else:
        fit_predict, tag = make_cnn(args), f"cnn_w{args.width}"
    run_experiment(fit_predict, args, tag)


if __name__ == "__main__":
    main()
