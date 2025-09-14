import sys
import random
import torch
from torch import nn
from torchvision import datasets, transforms

# Dispositivo (CPU/GPU)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device for inference.")

# Trasformazioni per test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13066047430038452,), (0.30810782313346863,)),
    transforms.Pad(2, fill=0, padding_mode='constant'),
])

# LeNet-5 definizione
class LeNet5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.le_stack = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1), self.tanh,
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1), self.tanh,
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, 5, 1), self.tanh
        )
        self.fc_stack = nn.Sequential(
            nn.Linear(120, 84), self.tanh,
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.le_stack(x)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x

def main():
    if len(sys.argv) != 3:
        print("Usage: python inference.py <model.pth> <num_images>")
        return

    model_path = sys.argv[1]
    try:
        n = int(sys.argv[2])
        if n <= 0:
            raise ValueError()
    except ValueError:
        print("Error: <num_images> must be a positive integer.")
        return

    # Carica dataset MNIST test
    test_data = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    if n > len(test_data):
        print(f"Warning: requested {n} images, but dataset contains only {len(test_data)}. Using {len(test_data)} instead.")
        n = len(test_data)

    # Seleziona n immagini casuali
    indices = random.sample(range(len(test_data)), n)
    samples = [test_data[i] for i in indices]

    # Separiamo immagini e labels e carichiamoli tutti in GPU come batch
    imgs = torch.stack([img for img, _ in samples]).to(device)
    labels = torch.tensor([label for _, label in samples], device=device)

    model = LeNet5Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        for i in range(n):
            print(f"[{i}] Predicted: {preds[i].item()} — Actual: {labels[i].item()}")

if __name__ == "__main__":
    main()

