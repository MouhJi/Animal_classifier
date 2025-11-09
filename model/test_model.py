import cv2
import torch
from argparse import ArgumentParser
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from model import model
from Transfer_byResNet import Transfer_ResNet
import torch.nn as nn
from PIL import Image
def parser():
    parser = ArgumentParser(description="Test model on single image")
    parser.add_argument("--size_image", "-s", default=224, type=int, help="Image size")
    parser.add_argument("--check_point", "-c", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="Path to image")
    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = Compose([
        Resize((args.size_image, args.size_image)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    categories = [
        'cane', 'cavallo', 'elefante', 'farfalla', 'gallina',
        'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo'
    ]

    # Load model
    net = Transfer_ResNet(num_class=len(categories)).to(device)

    checkpoint = torch.load(args.check_point, map_location=device)
    net.load_state_dict(checkpoint["model"])
    net.eval()

    image_original = cv2.imread(args.image_path)
    image_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)  #  Chuyển sang PIL
    image_tensor = test_transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(image_tensor)
        prob = nn.Softmax(dim=1)(output)
        max_idx = torch.argmax(prob, dim=1).item()
        confidence = prob[0][max_idx].item()

    label = f"{categories[max_idx]}: {confidence:.2f}"
    cv2.putText(image_original, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", image_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
