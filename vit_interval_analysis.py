from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import ViTForImageClassification, AutoImageProcessor

def collate_fn(batch):
    """Custom collate function to handle PIL images"""
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels

def test_model_accuracy(model, image_processor, test_loader, device):
    """Test model accuracy on validation set"""
    total = 0
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, labels = batch
            
            # Process images with the image processor
            # Images are PIL images from the dataset
            inputs = image_processor(images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            labels = labels.to(device)
            
            # Get predictions
            outputs = model(pixel_values)
            predicted = outputs.logits.argmax(dim=1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    print(f"\nFinal Results on {total} images:")
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    max_samples = 1000
    print(f"Using device: {device}")
    
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    
    # Move model to device and ensure all parameters are on the same device
    model = model.to(device)
    model.eval()
    
    summary(model, input_size=(1, 3, 224, 224))
    
    # Don't apply any transforms - let the image processor handle it
    # Imagenette returns PIL images by default which is what we want
    imagenet = datasets.Imagenette(
        root="./data", 
        split="val", 
        download=True,
        transform=None  # No transform, we'll process in the loop
    )
    
    test_loader = DataLoader(
        imagenet,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn  # IMPORTANT: Use custom collate function
    )
    
    # Test accuracy
    print("\nTesting model accuracy...")
    test_model_accuracy(model, image_processor, test_loader, device)

if __name__ == "__main__":
    main()
