import os
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, RandomAffine, ColorJitter
from sklearn.model_selection import train_test_split
from PIL import Image
class AnimalDataSet(Dataset):
    def __init__(self, root, transform=None, train=True, test_size=0.1, random_state=42):
        self.root = root
        self.transform = transform
        self.train = train
        
        # Define the 10 animal categories
        self.categories = [
            'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
            'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
        ]

        # Collect all image paths and their labels
        all_image_paths = []
        all_labels = []
        
        for idx, category in enumerate(self.categories):
            category_path = os.path.join(root, category)
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        all_image_paths.append(os.path.join(category_path, filename))
                        all_labels.append(idx)
        
        # Split the dataset into train and test sets
        # Use stratified split to maintain class distribution
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_image_paths, 
            all_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=all_labels
        )
        
        # Assign the appropriate split based on train parameter
        if train:
            self.image_paths = train_paths
            self.labels = train_labels
        else:
            self.image_paths = test_paths
            self.labels = test_labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_category_name(self, label_idx):
        """Get category name from label index"""
        return self.categories[label_idx]
    
    def get_num_classes(self):
        """Get number of classes"""
        return len(self.categories)

if __name__ == "__main__":
    # Define transforms
    train_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create train and test datasets
    train_dataset = AnimalDataSet(root="dataset", transform=train_transform, train=True, test_size=0.1)
    test_dataset = AnimalDataSet(root="dataset", transform=test_transform, train=False, test_size=0.1)
    print(len(test_dataset))
