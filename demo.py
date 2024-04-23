import cv2
import torch
import argparse
import numpy as np 
from PIL import Image


def get_mean_stdinv(img):
    """
    Compute mean and standard deviation
    """
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]

    mean_img = np.zeros((img.shape))
    mean_img[:,:,0] = mean[0]
    mean_img[:,:,1] = mean[1]
    mean_img[:,:,2] = mean[2]
    mean_img = np.float32(mean_img)

    std_img = np.zeros((img.shape))
    std_img[:,:,0] = std[0]
    std_img[:,:,1] = std[1]
    std_img[:,:,2] = std[2]
    stdinv_img = 1 / np.float32(std_img)
    return mean_img, stdinv_img


def numpy2tensor(img):
    """
    Convert numpy to tensor
    """
    return torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()


def preprocess(img, size, device):
    """
    Apply image transforms and convert image to normalized tensor
    """
    # Resize image
    img = cv2.imread(img)
    height, width = img.shape[:2]
    if height < width:
        scale = size / height
    else:
        scale = size / width
    new_height = int(height * scale)
    new_width = int(width * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Center crop
    start_h = (new_height - size) // 2
    start_w = (new_width - size) // 2
    end_w = start_w + size
    end_h = start_h + size
    img = img[start_h:end_h, start_w:end_w]

    # Normalize tensor
    mean_img, stdinv_img = get_mean_stdinv(img)
    img_tensor = numpy2tensor(img).to(device)
    mean_img_tensor = numpy2tensor(mean_img).to(device)
    stdinv_img_tensor = numpy2tensor(stdinv_img).to(device)
    img_tensor = img_tensor - mean_img_tensor
    img_tensor = img_tensor * stdinv_img_tensor
    return img_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run demo')
    parser.add_argument('--img', type=str, default='./images/real/6040.jpg')
    parser.add_argument('--checkpoint', type=str, default='./deployment/efficientformer/end2end.pt')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    img_tensor = preprocess(args.img, args.size, args.device)
    model = torch.jit.load(args.checkpoint).to(args.device)
    output = model(img_tensor)

    idx = torch.argmax(output)
    classes = ['DALL-E 2', 'DALL-E 3', 'Kandinsky 2.1', 'LCM (2 steps)', 'LCM (4 steps)', 'Midjourney 6', 'MidJourney 5.2', 'SD 1.5', 'SD 2.0', 'SDXL', 'SDXL Turbo', 'Stable Cascade', 'Real']
    pred = classes[idx]
    print("Prediction: ", pred)
