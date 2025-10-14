from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchinfo import summary


def test_model_robustness_accuracy(model, test_loader, device):
    """Test model accuracy on validation set"""
    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Process images
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"\nFinal Results on {total} images:")
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    max_samples = 1000
    print(f"Using device: {device}")

    # Try ImageNet validation set, fallback to a smaller dataset
    print("Loading CIFAR-100")
    # )
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True
    ).to(device)

    summary(model, input=(224,224))

    mean = [0.5070, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2761]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    cifar100 = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    # Create dataloader
    train_loader = DataLoader(
        cifar100,
        batch_size=32,
        shuffle=False,
    )

    test_loader = DataLoader(
        cifar100,
        batch_size=32,
        shuffle=False,
    )

    # Test accuracy
    print("\nTesting model accuracy...")
    test_model_accuracy(model, test_loader, device)


if __name__ == "__main__":
    main()
