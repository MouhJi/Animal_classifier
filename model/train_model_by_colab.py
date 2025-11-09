from ultralytics import YOLOv5
from model import model
import os
import shutil
import time
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomAffine, ColorJitter
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from Transfer_byResNet import Transfer_ResNet


# Ẩn cảnh báo TensorFlow / XLA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ===========================================
# Mount Google Drive
# ===========================================
drive_path = "/content/drive"
if not os.path.exists(drive_path):
    from google.colab import drive
    drive.mount(drive_path)
else:
    print("Drive đã được mount.")

# ===========================================
# Copy dataset về local runtime
# ===========================================
if not os.path.exists("/content/dataset"):
    print("Copying dataset from Drive to local runtime...")
    shutil.copytree(
        "/content/drive/MyDrive/Colab Notebooks/dataset",
        "/content/dataset",
        dirs_exist_ok=True
    )
else:
    print("Dataset already exists in local runtime.")

DATA_ROOT = "/content/dataset"

# ===========================================
# Dataset class
# ===========================================
class AnimalDataSet(Dataset):
    def __init__(self, root, transform=None, train=True, test_size=0.1, random_state=42, use_cache=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.use_cache = use_cache

        self.categories = [
            'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
            'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
        ]

        all_image_paths, all_labels = [], []
        for idx, cat in enumerate(self.categories):
            cat_dir = os.path.join(root, cat)
            for fn in os.listdir(cat_dir):
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_image_paths.append(os.path.join(cat_dir, fn))
                    all_labels.append(idx)

        train_paths, test_paths, train_labels, test_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=test_size, random_state=random_state, stratify=all_labels
        )

        if train:
            self.image_paths, self.labels = train_paths, train_labels
        else:
            self.image_paths, self.labels = test_paths, test_labels

        self.cache = {} if (use_cache and len(self.image_paths) < 5000) else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        if self.cache is not None and img_path in self.cache:
            image = self.cache[img_path]
        else:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot read {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            if self.cache is not None:
                self.cache[img_path] = image

        if self.transform:
            image = self.transform(image)
        return image, label

# ===========================================
# Hàm vẽ confusion matrix
# ===========================================
def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_Matrix', figure, epoch)

# ===========================================
# Parser
# ===========================================
def parser():
    parser = ArgumentParser(description="Train Animal Classifier")
    parser.add_argument("--batch_size", "-b", default=32, type=int)
    parser.add_argument("--epochs", "-e", default=30, type=int)
    parser.add_argument("--size_image", "-s", default=224, type=int)
    parser.add_argument("--check_point", "-c", type=str, default=None)
    parser.add_argument("--logging", "-l", type=str, default="Tensorboard")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_model")
    parser.add_argument("--drive_folder", "-d", type=str, default="/content/drive/MyDrive/Colab Models")
    return parser.parse_args()

# ===========================================
# Model import
# ===========================================

# ===========================================
# Main
# ===========================================
if __name__ == '__main__':
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.trained_model, exist_ok=True)
    os.makedirs(args.drive_folder, exist_ok=True)
    writer = SummaryWriter(args.logging)

    # Transform
    train_transform = Compose([
        Resize((args.size_image, args.size_image)),
        RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(-5, 5)),
        ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.05),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = Compose([
        Resize((args.size_image, args.size_image)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset + Loader
    data_train = AnimalDataSet(root=DATA_ROOT, train=True, transform=train_transform)
    data_test = AnimalDataSet(root=DATA_ROOT, train=False, transform=test_transform)
    dataLoader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dataLoader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model setup
    model = model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_epoch, best_acc = 0, 0
    if args.check_point and os.path.exists(args.check_point):
        ckpt = torch.load(args.check_point)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt.get("best_acc", 0)

    # ===========================================
    # TRAINING LOOP + AUTO-SAVE TO DRIVE
    # ===========================================
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        progress = tqdm(dataLoader_train, colour="green")

        for i, (images, labels) in enumerate(progress):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_description(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss.item():.3f}")
            writer.add_scalar("train/loss", loss.item(), epoch * len(progress) + i)

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in dataLoader_test:
                images, labels = images.to(device), labels.to(device)
                preds = torch.argmax(model(images), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(writer, cm, data_test.categories, epoch)
        writer.add_scalar("test/accuracy", acc, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} | Accuracy: {acc:.4f} | Time: {(time.time()-start_time)/60:.2f} min")

        # ---- Save last model ----
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
        }
        local_path = f"{args.trained_model}/last_state_model.pt"
        torch.save(ckpt, local_path)

        # ---- Copy lên Drive ----
        drive_path = os.path.join(args.drive_folder, f"epoch_{epoch+1:03d}.pt")
        try:
            shutil.copy(local_path, drive_path)
            print(f"Saved checkpoint to Drive: {drive_path}")
        except Exception as e:
            print(f"Drive save failed: {e}")

        # ---- Save best model (fixed version) ----
        if acc > best_acc:
            best_acc = acc
            ckpt_best = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc
            }
            best_path = os.path.join(args.drive_folder, "best_model.pt")
            torch.save(ckpt_best, best_path)
            print(f"New best model saved to Drive! acc={acc:.4f}")

        # Dọn bộ nhớ
        gc.collect()
        torch.cuda.empty_cache()

    print("Training finished successfully.")
