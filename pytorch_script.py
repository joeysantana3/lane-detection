# C:\Users\joeys\source\virtualenvs\pytorchenv\Scripts\activate.bat

import cv2
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2


CHECKPOINT_PATH = r"C:\Users\joeys\source\lane-detection\my_checkpoint.pth.tar"
VIDEO_FILE = r"C:\Users\joeys\OneDrive\Desktop\production ID_4608285.mp4"
SENSITIVITY = 230 # between 0-255, decides which values to count for mask
SCALE_PERCENTAGE = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 320  # 1280 originally
IMAGE_WIDTH = 480  # 1918 originally




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

    def test():
        x = torch.randn((3, 1, 161, 161))
        model = UNET(in_channels=1, out_channels=1)
        preds = model(x)
        assert preds.shape == x.shape


model = UNET(in_channels=3, out_channels=1).to(DEVICE)
model_file = torch.load(CHECKPOINT_PATH)
model.load_state_dict(model_file['state_dict'])
model.eval()

train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )



cap = cv2.VideoCapture(VIDEO_FILE)
  
if (cap.isOpened()== False):
    print("Error opening video file")
  
while(cap.isOpened()):
      
    ret, frame = cap.read()
    if ret == True:
        width = int(frame.shape[1] * SCALE_PERCENTAGE / 100)
        height = int(frame.shape[0] * SCALE_PERCENTAGE / 100)
        dsize = (width, height)
        frame = cv2.resize(frame, dsize)
        augmentations = train_transform(image=frame)
        image = augmentations["image"]
        image = image.to('cuda')
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            pred = torch.sigmoid(model(image))
            pred = (pred > 0.5).float()

        transform = T.ToPILImage()
        pred = torch.squeeze(pred)
        predicted_mask = transform(pred)
        predicted_mask = np.array(predicted_mask)
        image = frame
        predicted_mask = cv2.resize(predicted_mask, (image.shape[1], image.shape[0]))
        color_mask = np.zeros_like(image)
        color_mask[predicted_mask >= SENSITIVITY] = (0, 0, 255) # green color
        masked_color = cv2.bitwise_and(image, color_mask)
        output = cv2.addWeighted(image, 0.9, masked_color, 0.9, 1)



        cv2.imshow('Frame', output)
          
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
  
    else:
        break
  
cap.release()
cv2.destroyAllWindows()