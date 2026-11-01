"""A small binary CNN trained from scratch on CPU. No pretrained weights."""

import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from camera_quality.inspection import metrics, preprocess


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(24*4*4, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, images):
        return self.network(images).squeeze(1)


def tensors(images: list[np.ndarray]) -> torch.Tensor:
    channels = []
    for image in images:
        smooth, residual = preprocess(image)
        channels.append(np.stack([smooth.astype(np.float32)/255.0, residual.astype(np.float32)/64.0]))
    return torch.from_numpy(np.stack(channels))


def train(train_images, train_labels, validation_images, validation_labels, epochs=10, seed=20261101):
    if epochs < 1:
        raise ValueError("At least one epoch required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    model = TinyCNN().cpu()
    inputs = tensors(train_images)
    targets = torch.tensor(train_labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_x = tensors(validation_images)
    val_y = torch.tensor(validation_labels, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_state, history, best_epoch = float("inf"), None, [], None
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(y)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(val_x), val_y))
        history.append(dict(epoch=epoch, train_loss=total/len(targets), validation_loss=validation_loss))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(val_x)).tolist()
    # Same validation-F1 selection objective as the classical method.
    threshold = max([i/20 for i in range(2, 19)], key=lambda t: (
        metrics(validation_labels, [int(p >= t) for p in probabilities])["f1"], -abs(t-.5)))
    return model, threshold, history, best_epoch


def predict(model: TinyCNN, images: list[np.ndarray]) -> list[float]:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(tensors(images))).tolist()
