import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import os
from PIL import Image
import numpy as np

from datasets import KidneyDataset
from models import UNet
import segmentation_models_pytorch as smp

def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} device.')

    transform = transforms.Compose([
        transforms.Resize(192),
        transforms.ToTensor(),
    ])

    test_set = KidneyDataset('test', transform=transform)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, num_workers=0)

    net = smp.Unet(
        encoder_name="resnet152",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
    ).to(device)

    if args.model_path and os.path.exists(args.model_path):
        # Load model weights.
        net.load_state_dict(torch.load(args.model_path, map_location=device))

    idx = 0
    net.eval()
    for index, (images, masks, image_name) in enumerate(test_loader, 1):
        images = images.to(device)

        with torch.no_grad():
            outputs = net(images)

        for x in range(len(images)):

            Image.fromarray(((images[x].cpu().numpy().transpose(1, 2, 0))*255).astype(np.uint8)).save(f'{args.figure_path}/{os.path.basename(image_name[x]).replace(".png", "")}_image_{idx}.png')
            Image.fromarray(((masks[x].cpu().numpy().transpose(1, 2, 0)[..., 0]) * 255).astype(np.uint8)).save(f'{args.figure_path}/{os.path.basename(image_name[x]).replace(".png", "")}_mask_{idx}.png')
            plt.imsave(f'{args.figure_path}/{os.path.basename(image_name[x]).replace(".png", "")}_predicted_{idx}.png', outputs[x].cpu().numpy().transpose(1, 2, 0)[..., 0])

            idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--figure_path', default='figure', type=str)
    args = parser.parse_args()
    print(vars(args))
    
    os.makedirs(args.figure_path, exist_ok=True)

    main(args)
