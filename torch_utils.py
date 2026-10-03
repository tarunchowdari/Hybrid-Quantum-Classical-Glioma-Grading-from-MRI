import numpy as np
import torch
import torch.nn as nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def conv_stack(width, in_channels=4):
    """Three conv blocks (conv -> batch norm -> ReLU -> 2x2 max-pool).

    Returns the blocks and the number of output channels. Apply a global average over (H, W) afterwards.
    """
    layers, ch = [], in_channels
    for out_ch in (width, 2 * width, 4 * width):
        layers += [nn.Conv2d(ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
                   nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        ch = out_ch
    return nn.Sequential(*layers), ch


def augment(x):
    """Random horizontal / vertical flips per image, plus one random 90-degree rotation per batch."""
    flip_w = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(flip_w[:, None, None, None], x.flip(3), x)
    flip_h = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(flip_h[:, None, None, None], x.flip(2), x)
    return torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))


def fit_and_predict(model, X_train, y_train, X_val, *, epochs, batch_size, lr, device,
                    circuit_lr=None, augment_inputs=False, log_every=5):
    """Train `model` with class-weighted BCE and AdamW + OneCycle, then return P(HGG) for `X_val`.

    Parameters named `theta` (the circuit's rotation angles) get their own learning rate `circuit_lr`
    and no weight decay, because angles live on a different scale than network weights.
    """
    Xt = torch.from_numpy(X_train).to(device)  # stays float16 on the device, cast per batch
    yt = torch.from_numpy(y_train).float().to(device)
    Xv = torch.from_numpy(X_val).to(device)

    pos_weight = torch.tensor((y_train == 0).sum() / (y_train == 1).sum(), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    angles = [p for name, p in model.named_parameters() if name.endswith("theta")]
    others = [p for name, p in model.named_parameters() if not name.endswith("theta")]
    groups = [{"params": others, "lr": lr}]
    if angles:
        groups.append({"params": angles, "lr": circuit_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(Xt) + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups], total_steps=steps)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xt), device=device)
        total_loss = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = order[i:i + batch_size]
            if len(idx) < 2:  # batch norm needs at least two samples
                continue
            xb = Xt[idx].float()
            if augment_inputs:
                xb = augment(xb)
            loss = loss_fn(model(xb), yt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}/{epochs}  train loss {total_loss / len(Xt):.4f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xv), 256):
            probs.append(torch.sigmoid(model(Xv[i:i + 256].float())).cpu().numpy())
    return np.concatenate(probs)
