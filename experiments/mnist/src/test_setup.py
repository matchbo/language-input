import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def test_setup():
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for MPS (Metal) availability on Mac
    print(f"MPS (Metal) device available: {torch.backends.mps.is_available()}")
    
    # Create a test tensor
    x = torch.randn(3, 3)
    print(f"\nTest tensor:\n{x}")
    
    # Test MNIST dataset access
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    print(f"\nMNIST dataset size: {len(train_dataset)}")
    
    # Display a sample image
    image, label = train_dataset[0]
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

    # Original tensor shape
    print(f"Original shape: {image.shape}")

    # After squeeze
    print(f"After squeeze: {image.squeeze().shape}")

    # Try different colormaps
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Gray')

    plt.subplot(1, 3, 2)
    plt.imshow(image.squeeze(), cmap='viridis')
    plt.title('Viridis')

    plt.subplot(1, 3, 3)
    plt.imshow(image.squeeze(), cmap='hot')
    plt.title('Hot')

    plt.show()    

if __name__ == "__main__":
    test_setup()
