"""Hybrid quantum-classical HGG-vs-LGG classifier, with the quantum circuit simulated directly in PyTorch.

Pipeline
  image (4 x 64 x 64)
    -> ENCODER -> vector of 2^n numbers (n = number of qubits)
         pca : fixed PCA fitted on the training fold only (the setup described in the reference paper)
         cnn : small learnable CNN followed by a linear layer
    -> L2-normalise -> used as the amplitudes of an n-qubit state                (amplitude encoding)
    -> L layers of [Ry(theta) on every qubit, then a ring of CZ gates]           (trainable circuit)
    -> read out <Z_i> on every qubit (n numbers in [-1, 1])
    -> small classical head -> logit for HGG

Why a hand-written simulator: Ry and CZ only ever produce real amplitudes, so the whole circuit is real
tensor algebra. It runs batched on the GPU and trains with ordinary autograd. `--check` compares it with a
brute-force dense-matrix simulation and, if installed, with PennyLane.

Ablation (--core)
  quantum   : the circuit above
  classical : a Linear(2^n -> n) + tanh layer instead of the circuit (same input, same readout size)
  none      : encoder output goes straight into the head

Usage
  python quantum_hybrid.py --encoder pca --core quantum --qubits 6 --layers 3
  python quantum_hybrid.py --encoder cnn --core quantum --qubits 6 --layers 3 --seeds 3
  python quantum_hybrid.py --encoder cnn --core classical --qubits 6
  python quantum_hybrid.py --check
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from common import run_experiment
from torch_utils import conv_stack, count_params, fit_and_predict, get_device, seed_everything


# ============================================================================ simulator
class RyCZCircuit(nn.Module):
    """n-qubit real statevector simulator: `layers` x [Ry on every qubit, CZ ring], then measure <Z_i>.

    Qubit 0 is the most significant bit of the basis index: index = sum_i b_i * 2^(n-1-i).
    """

    def __init__(self, n, layers):
        super().__init__()
        self.n, self.layers = n, layers
        # Wide initial angles: with tiny angles the circuit is close to the identity and the readout
        # only sees the squared amplitudes, i.e. it throws away the sign information in the input.
        self.theta = nn.Parameter((torch.rand(layers, n) * 2 - 1) * np.pi)

        index = torch.arange(2 ** n)
        bits = torch.stack([(index >> (n - 1 - q)) & 1 for q in range(n)], dim=1)  # (2^n, n)
        ring = [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
        cz = torch.ones(2 ** n)
        for a, b in ring:
            cz = cz * (1 - 2 * (bits[:, a] & bits[:, b]).float())  # -1 on basis states where both qubits are 1
        self.register_buffer("cz", cz)
        self.register_buffer("z_sign", 1 - 2 * bits.float())  # (2^n, n): +1 where the qubit is 0, -1 where it is 1

    def forward(self, amplitudes):
        batch, n = amplitudes.shape[0], self.n
        psi = amplitudes
        for layer in range(self.layers):
            for q in range(n):
                c, s = torch.cos(self.theta[layer, q] / 2), torch.sin(self.theta[layer, q] / 2)
                psi = psi.reshape(batch, 2 ** q, 2, 2 ** (n - q - 1))
                zero, one = psi[:, :, 0], psi[:, :, 1]
                psi = torch.stack([c * zero - s * one, s * zero + c * one], dim=2)  # Ry = [[c, -s], [s, c]]
            psi = psi.reshape(batch, 2 ** n) * self.cz
        return (psi ** 2) @ self.z_sign  # (batch, n) expectation values of Pauli-Z


# ============================================================================ model
class CNNEncoder(nn.Module):
    """Image -> 2^n numbers."""

    def __init__(self, out_dim, width=16, dropout=0.3):
        super().__init__()
        self.f, channels = conv_stack(width)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(channels, out_dim)

    def forward(self, x):
        return self.proj(self.drop(self.f(x).mean((2, 3))))


class Hybrid(nn.Module):
    def __init__(self, encoder, core, n_qubits, layers, hidden):
        super().__init__()
        self.core_kind = core
        dim = 2 ** n_qubits
        self.enc = CNNEncoder(dim) if encoder == "cnn" else nn.Identity()  # "pca" features arrive precomputed
        if core == "quantum":
            self.core = RyCZCircuit(n_qubits, layers)
            features = n_qubits
        elif core == "classical":  # same input size, same readout size, similar parameter count
            self.core = nn.Sequential(nn.Linear(dim, n_qubits), nn.Tanh())
            features = n_qubits
        else:
            self.core = nn.Identity()
            features = dim
        if hidden:
            self.head = nn.Sequential(nn.Linear(features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        else:
            self.head = nn.Linear(features, 1)

    def forward(self, x):
        v = self.enc(x)
        if self.core_kind == "quantum":
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)  # amplitude encoding: unit-length state vector
        return self.head(self.core(v)).squeeze(1)


# ============================================================================ training
def pca_features(X_train, X_val, n_components, seed):
    """PCA fitted on the training fold, each component scaled to unit variance."""
    from sklearn.decomposition import PCA

    A = X_train.reshape(len(X_train), -1).astype(np.float32)
    B = X_val.reshape(len(X_val), -1).astype(np.float32)
    pca = PCA(n_components, svd_solver="randomized", random_state=seed).fit(A)
    A, B = pca.transform(A), pca.transform(B)
    scale = A.std(0, keepdims=True) + 1e-6
    return (A / scale).astype(np.float32), (B / scale).astype(np.float32)


def make_fit_predict(args, device):
    def fit_predict(X_train, y_train, X_val, seed):
        seed_everything(seed)
        if args.encoder == "pca":
            X_train, X_val = pca_features(X_train, X_val, 2 ** args.qubits, seed)
        model = Hybrid(args.encoder, args.core, args.qubits, args.layers, args.hidden).to(device)
        probs = fit_and_predict(model, X_train, y_train, X_val, epochs=args.epochs, batch_size=args.bs,
                                lr=args.lr, device=device, circuit_lr=args.qlr,
                                augment_inputs=(args.encoder == "cnn"), log_every=10)
        return probs, count_params(model)

    return fit_predict


# ============================================================================ simulator check
def check():
    """Compare the simulator with a brute-force dense-matrix simulation and with PennyLane (if installed)."""
    torch.manual_seed(0)
    n, layers = 4, 2
    circuit = RyCZCircuit(n, layers).double()
    circuit.theta.data = torch.randn(layers, n, dtype=torch.double)
    amplitudes = torch.randn(3, 2 ** n, dtype=torch.double)
    amplitudes = amplitudes / amplitudes.norm(dim=1, keepdim=True)
    with torch.no_grad():
        got = circuit(amplitudes).numpy()

    # Brute force with explicit 2^n x 2^n matrices.
    I, Z = np.eye(2), np.diag([1.0, -1.0])

    def kron_all(matrices):
        out = np.array([[1.0]])
        for m in matrices:
            out = np.kron(out, m)
        return out

    def ry(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]])

    theta = circuit.theta.detach().numpy()
    cz = np.diag(circuit.cz.detach().numpy().astype(float))
    dense = []
    for b in range(amplitudes.shape[0]):
        psi = amplitudes[b].numpy().copy()
        for layer in range(layers):
            psi = cz @ (kron_all([ry(theta[layer, q]) for q in range(n)]) @ psi)
        dense.append([psi @ kron_all([Z if k == q else I for k in range(n)]) @ psi for q in range(n)])
    error = np.abs(got - np.array(dense)).max()
    print(f"max |simulator - dense matrices| = {error:.2e}")
    assert error < 1e-9, "simulator disagrees with the dense-matrix reference"

    try:
        import pennylane as qml
    except ImportError:
        print("PennyLane not installed: skipped (pip install pennylane to cross-check)")
    else:
        device = qml.device("default.qubit", wires=n)

        @qml.qnode(device)
        def circuit_pl(state, angles):
            qml.StatePrep(state, wires=range(n))
            for layer in range(layers):
                for q in range(n):
                    qml.RY(angles[layer, q], wires=q)
                for q in range(n):
                    qml.CZ(wires=[q, (q + 1) % n])
            return [qml.expval(qml.PauliZ(q)) for q in range(n)]

        reference = np.array([np.array(circuit_pl(amplitudes[b].numpy(), theta)) for b in range(amplitudes.shape[0])])
        print(f"max |simulator - PennyLane|      = {np.abs(got - reference).max():.2e}")
    print("simulator OK")


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_64/slices.npz")
    ap.add_argument("--encoder", choices=["pca", "cnn"], default="pca")
    ap.add_argument("--core", choices=["quantum", "classical", "none"], default="quantum")
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--layers", type=int, default=3, help="circuit depth (quantum core only)")
    ap.add_argument("--hidden", type=int, default=0, help="hidden units in the head (0 = a single linear layer)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64, help="batch size")
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate for network weights")
    ap.add_argument("--qlr", type=float, default=5e-2, help="peak learning rate for the circuit's rotation angles")
    ap.add_argument("--seeds", type=int, default=1, help="repeat the whole CV with this many seeds")
    ap.add_argument("--eval-test", action="store_true",
                    help="train on all CV patients and score the held-out test patients (use ONCE)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--check", action="store_true", help="validate the simulator and exit")
    args = ap.parse_args()

    if args.check:
        return check()

    device = get_device()
    print("device:", device)
    suffix = f"_L{args.layers}_h{args.hidden}" if args.core == "quantum" else f"_h{args.hidden}"
    tag = f"hybrid_{args.encoder}_{args.core}_q{args.qubits}{suffix}"
    run_experiment(make_fit_predict(args, device), args, tag)


if __name__ == "__main__":
    main()
