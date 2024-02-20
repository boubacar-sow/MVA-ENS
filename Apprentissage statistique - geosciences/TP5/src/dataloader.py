import os
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='Train'):
        self.root_dir = root_dir
        self.split = split

        if split == 'Train' or split == 'Val':
            self.images_dir = os.path.join(root_dir, split, 'images')
            self.labels_dir = os.path.join(root_dir, split, 'labels')
            self.file_names = os.listdir(self.images_dir)
        elif split == 'Test':
            self.images_dir = os.path.join(root_dir, split, 'images')
            self.file_names = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_path = os.path.join(self.images_dir, file_name)
        
        if self.split == 'Train' or self.split == 'Val':
            label_path = os.path.join(self.labels_dir, file_name)
            label = np.load(label_path)
        else:
            label = None

        image = np.load(image_path)

        return image, label

if __name__ == "__main__":
    # Test the dataloader
    root_dir = './data'
    train_dataset = SegmentationDataset(root_dir, split='Train')
    val_dataset = SegmentationDataset(root_dir, split='Val')
    test_dataset = SegmentationDataset(root_dir, split='Test')

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Access a sample from the datasets
    train_sample = train_dataset[0]
    val_sample = val_dataset[0]
    test_sample = test_dataset[0]

    # Print sample shapes
    print(f"Train sample shape: {train_sample[0].shape}, Label shape: {train_sample[1].shape}")
    print(f"Val sample shape: {val_sample[0].shape}, Label shape: {val_sample[1].shape}")
    print(f"Test sample shape: {test_sample[0].shape}")

    # print one label
    plt.imshow(train_sample[1])

    # Display the first two images
    plt.subplot(1, 2, 1)
    plt.imshow(train_sample[0])
    plt.title('Train Image')
    plt.subplot(1, 2, 2)
    plt.imshow(val_sample[0])
    plt.title('Val Image')
    plt.show()