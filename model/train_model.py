import os.path
import shutil
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize, Compose,ColorJitter,RandomAffine, ToTensor, Normalize
from dataset import AnimalDataSet
from model import model
import torch.nn as nn
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from Transfer_byResNet import Transfer_ResNet


# show confusion matrix in tensorboard
def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

# argument parser for run by terminal
def parser():
    parser = ArgumentParser(description="train data")
    parser.add_argument("--batch_size", "-b", default=8, type=int, help="batch of epoch")
    parser.add_argument("--epochs", "-e", default=50, type=int, help="epoch for training")
    parser.add_argument("--size_image", "-s", default=224, type=int, help="size of the image")
    parser.add_argument("--check_point", "-c", type=str, default=None)
    parser.add_argument("--root", "-r", type=str, default="./dataset", help="root of dataset")
    parser.add_argument("--logging", "-l", type=str, default="Tensorboard", help="Tennsorboard")
    parser.add_argument("--trained_model", "-t", type=str, default="trained_model")
    return parser.parse_args()

if __name__ == '__main__':
    args = parser()

    # define the divice you have
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform for train dataset
    train_transform = Compose([
        Resize((args.size_image, args.size_image)), #resize for each image to equal
        ToTensor(), #convert dimension of image to C, H, W. Scale value in range [0,1] and type is Tensor
        RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.9, 1.1), shear=(-5, 5)), # adjust position of image
        ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.05), # adjust color of image
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    # normalize value to image_net's value
    ])
    # transform for test dataset
    test_transform = Compose([
        Resize((args.size_image, args.size_image)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # define dataloader for train set
    data_train = AnimalDataSet(root="./dataset", train=True, transform=train_transform)
    dataLoader_train = DataLoader(dataset=data_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # define dataloader for test set
    data_test = AnimalDataSet(root = "./dataset", train = False, transform=test_transform)
    dataLoader_test = DataLoader(dataset=data_test, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)
    # remove file tensorboard from previous phase train
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)

    # make folder trained model to save model by best accuracy and last epoch
    if not os.path.isdir(args.trained_model):
        os.mkdir(args.trained_model)

    writer = SummaryWriter(args.logging)
    model = model(num_classes=10).to(device) #define model and transform to gpu
    criterion = nn.CrossEntropyLoss()   #define loss function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)   #define optimizer algorithm

    # load check point from last train or best accuracy
    if args.check_point:
        check_point = torch.load(args.check_point)
        start_epoch = check_point["epoch"]
        best_acc = check_point["best_acc"]
        model.load_state_dict(check_point["model"])
        optimizer.load_state_dict(check_point["optimizer"])
    #if the train is begin then initalize epoch and best accuracy = 0
    else:
        start_epoch = 0
        best_acc = 0
    num_inter = len(dataLoader_train) #the number of loop when finished the epoch
    for epoch in range(start_epoch, args.epochs):
        #forward
        model.train()
        progress_bar = tqdm(dataLoader_train, colour="green")
        for index,(image, label) in enumerate(progress_bar):
            #convert to gpu
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss_value = criterion(output, label)
        # backward
            optimizer.zero_grad() #don't evaluate gradient in buffer
            loss_value.backward() #backward by chain rule
            optimizer.step()
        progress_bar.set_description("Epoch {}/{}   interation{}/{}     loss:{:.3f}".format(epoch+1, args.epochs, index, num_inter, loss_value))
        writer.add_scalar("train/loss",loss_value, epoch * num_inter + index) #show out tensorboard
        #Inference
        model.eval()
        all_predictions = []
        all_label = []
        for index,(image, label) in enumerate(dataLoader_test):
            all_label.append(label)
            image = image.to(device)
            label = label.to(device)

            with torch.no_grad():
                prediction = model(image)
                indice = torch.argmax(prediction,dim=1).cpu()
                all_predictions.extend(indice)
                loss_value = criterion(prediction, label)
        all_label = [label.item() for label in all_label]
        all_predictions = [prediction.item() for prediction in all_predictions]
        plot_confusion_matrix(writer, confusion_matrix(all_label,all_predictions), class_names=data_test.categories,epoch=epoch)
        accuracy = accuracy_score(all_label, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
        writer.add_scalar("Inference/Accuracy", accuracy, epoch)

        checkPoint = {
            "epoch" : epoch+1,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        torch.save(checkPoint, "{}/last_state_model.pt".format(args.trained_model))

        if accuracy > best_acc:
            checkPoint = {
                "epoch" : epoch + 1,
                "model" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_acc" : accuracy
            }
            torch.save(checkPoint, "{}/best_state_model.pt".format(args.trained_model))
