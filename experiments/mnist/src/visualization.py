import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# 1. Display multiple digits in a grid
def show_digit_grid(dataset, rows=3, cols=6):
    plt.figure(figsize=(15, 8))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        image, label = dataset[i]
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')  # Hide axes for cleaner look
    plt.tight_layout()
    plt.show()

# 2. Analyze pixel values of a single digit
def analyze_digit(dataset, index=0):
    image, label = dataset[index]
    
    # Original image
    plt.figure(figsize=(15, 5))
    
    # Show original
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Original Digit: {label}')
    plt.axis('off')
    
    # Show pixel values as heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(image.squeeze(), cmap='hot')
    plt.title('Pixel Intensity Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    # Show pixel values as numbers
    plt.subplot(1, 3, 3)
    pixel_values = image.squeeze().numpy()
    plt.imshow(pixel_values, cmap='gray')
    # Only show some pixel values for clarity (every 4th pixel)
    for i in range(0, 28, 4):
        for j in range(0, 28, 4):
            plt.text(j, i, f'{pixel_values[i, j]:.2f}', 
                    color='red', fontsize=8, ha='center', va='center')
    plt.title('Selected Pixel Values')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nDigit Statistics:")
    print(f"Shape: {image.shape}")
    print(f"Min pixel value: {image.min():.3f}")
    print(f"Max pixel value: {image.max():.3f}")
    print(f"Mean pixel value: {image.mean():.3f}")

# Run both visualizations
print("Showing grid of digits...")
show_digit_grid(train_dataset)

print("\nAnalyzing a single digit...")
analyze_digit(train_dataset, index=0)

def find_digit_indices(dataset, digits, samples_per_digit=6):
    indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label in digits and len(indices[label]) < samples_per_digit:
            indices[label].append(idx)
        if all(len(indices[d]) >= samples_per_digit for d in digits):
            break
    return indices

def show_digit_variants(dataset, digits=[2, 3, 4], samples_per_digit=6):
    indices = find_digit_indices(dataset, digits, samples_per_digit)
    
    rows = len(digits)
    cols = samples_per_digit
    plt.figure(figsize=(15, 8))
    
    for row, digit in enumerate(digits):
        for col, idx in enumerate(indices[digit]):
            image, _ = dataset[idx]
            plt.subplot(rows, cols, row * cols + col + 1)
            plt.imshow(image.squeeze(), cmap='gray')
            if col == 0:
                plt.ylabel(f'Digit {digit}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    return indices

def analyze_pixel_distributions(dataset, digit_indices):
    plt.figure(figsize=(15, 10))
    
    # Collect pixel values for each digit
    all_pixels = []
    digit_pixels = defaultdict(list)
    
    for digit, indices in digit_indices.items():
        for idx in indices:
            image, _ = dataset[idx]
            pixels = image.numpy().flatten()
            all_pixels.extend(pixels)
            digit_pixels[digit].extend(pixels)
    
    # Overall distribution
    plt.subplot(2, 2, 1)
    plt.hist(all_pixels, bins=50, density=True, alpha=0.7)
    plt.title('Overall Pixel Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    
    # Individual digit distributions
    for i, digit in enumerate(digit_indices.keys(), 2):
        plt.subplot(2, 2, i)
        plt.hist(digit_pixels[digit], bins=50, density=True, alpha=0.7)
        plt.title(f'Pixel Distribution for Digit {digit}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print("\nPixel Value Statistics:")
    print(f"Overall mean: {np.mean(all_pixels):.3f}")
    print(f"Overall std: {np.std(all_pixels):.3f}")
    for digit in digit_indices.keys():
        print(f"\nDigit {digit}:")
        print(f"Mean: {np.mean(digit_pixels[digit]):.3f}")
        print(f"Std: {np.std(digit_pixels[digit]):.3f}")

# Run the analysis
indices = show_digit_variants(train_dataset)
analyze_pixel_distributions(train_dataset, indices)

# Demonstrate normalization effects
def show_normalization_effects():
    # Create a sample image with different normalizations
    plt.figure(figsize=(15, 5))
    
    # Get a sample digit
    image, label = train_dataset[0]
    
    # Original normalized [0,1]
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title('Original (normalized to [0,1])')
    plt.axis('off')
    
    # Unnormalized [0,255]
    plt.subplot(1, 3, 2)
    plt.imshow(image.squeeze() * 255, cmap='gray')
    plt.title('Unnormalized (scaled to [0,255])')
    plt.axis('off')
    
    # Different normalization [-1,1]
    plt.subplot(1, 3, 3)
    plt.imshow(image.squeeze() * 2 - 1, cmap='gray')
    plt.title('Normalized to [-1,1]')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

show_normalization_effects()