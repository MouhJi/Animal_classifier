# 🐾 Animal Classification using Custom CNN

A PyTorch-based deep learning project for classifying 10 types of animals using a custom convolutional neural network (CNN).  
The project also includes training visualization through TensorBoard and inference on single images.

---

## 📘 Project Structure
```
├──  model/
|      ├── Transfer_byResNet.py # fine-tune layer from resnet model
|      ├── dataset.py # Dataset definition and preprocessing
|      ├── model.py # Custom CNN model    
|      ├── test_model.py # Inference script for single image prediction
|      ├── train_model.py # Training loop with TensorBoard logging   
|      └── train_model_by_colab # setting train optimize to colab
├──  Dockerfile # (Optional) Containerized environment setup
└──  requirement.txt
```


---

## 🧠 Model Architecture

The model is a 5-layer CNN built from scratch with convolutional, batch normalization, and LeakyReLU activations.  
It ends with a fully connected block of 3 linear layers.

```python
Conv2D → BatchNorm → LeakyReLU → MaxPool × 5  
Flatten → Linear(8192→512) → ReLU → Linear(512→256) → ReLU → Linear(256→num_classes)
```

Each convolutional block extracts hierarchical features from animal images for robust classification.

---

🐶 Dataset

link to dataset

(https://www.kaggle.com/datasets/alessiocorrado99/animals10/data)

You can replace the label to italian or modify code
This project uses a dataset of 10 animal categories:
```
cane, cavallo, elefante, farfalla, gallina,
gatto, mucca, pecora, ragno, scoiattolo
```
Each folder contains images for one class.
The dataset is automatically split into train/test (90/10) with stratification.

You can structure your dataset like:
```
dataset/
├── cane/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── cavallo/
├── elefante/
└── ...
```

---

⚙️ Installation
```
git clone https://github.com/MouhJi/Animal_classifier.git
cd Animal_classifier
pip install -r requirements.txt
```

---

🚀 Training

To start training your model:
```
python train_model.py --batch_size 8 --epochs 50 --root ./dataset
```

Optional arguments:
| Argument          | Description                     | Default       |
| ----------------- | ------------------------------- | ------------- |
| `--batch_size`    | Batch size for training         | 8             |
| `--epochs`        | Number of epochs                | 50            |
| `--size_image`    | Resize dimension of input image | 224           |
| `--check_point`   | Resume training from checkpoint | None          |
| `--root`          | Path to dataset                 | ./dataset     |
| `--logging`       | TensorBoard log directory       | Tensorboard   |
| `--trained_model` | Output folder for weights       | trained_model |
Training progress and metrics are logged in TensorBoard.

To visualize them:
```
tensorboard --logdir Tensorboard
```

---

📊 TensorBoard Visualization

Below is an example of the training accuracy and confusion matrix logged via TensorBoard:

<img width="446" height="323" alt="test_accuraccy" src="https://github.com/user-attachments/assets/f67379ae-91e3-4e82-b786-3090f5b8e7f5" />

<img width="433" height="325" alt="train_loss" src="https://github.com/user-attachments/assets/dd655b54-b645-4ec5-900a-df7c52c49545" />

<img width="925" height="880" alt="confusion_matrix" src="https://github.com/user-attachments/assets/abe55736-83ac-4f1d-97c1-36f4478ef76d" />

---

🧪 Inference

Once you’ve trained your model, test it on any image:
```
python test_model.py --check_point trained_model/best_state_model.pt --image_path ./samples/cat.jpg
```
The predicted label and confidence score will be shown on the image.

<img width="504" height="319" alt="Screenshot 2025-11-09 164034" src="https://github.com/user-attachments/assets/13a46a85-baf2-4834-a473-e6447f3763ce" />

<img width="460" height="343" alt="Screenshot 2025-11-09 164119" src="https://github.com/user-attachments/assets/e402548f-78cd-4bba-8743-27f2e95358d5" />

<img width="339" height="374" alt="Screenshot 2025-11-09 161224" src="https://github.com/user-attachments/assets/fedf5980-44e8-4bd0-98ec-bdfa20be889b" />

---

🐋 Docker Support (Optional)

If you want to run the entire project in Docker:
```
docker build -t animal_cnn .
docker run -it --gpus all -v "$(pwd)":/workspace animal_cnn
```
---
📈 Results

| Metric   | Value           |
| -------- | --------------- |
| Accuracy | ~0.9767         |
| Loss     | ~0.012          |

---
🧑‍💻 Author

Mouh Ji
GitHub: @MouhJi

---

🪪 License

This project is released under the MIT License.
