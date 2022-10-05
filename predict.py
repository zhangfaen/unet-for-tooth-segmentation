import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    print("image shape: ", img.shape)

    with torch.no_grad():
        output = net(img)
        print("output shape:", output.shape)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        print("probs shape:", probs.shape)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    print("net.n_classes:", net.n_classes)
    ret = None
    if net.n_classes == 1:
        ret = (full_mask > out_threshold).numpy()
    else:
        ret = F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    return ret


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    args = get_args()
    logging.info(args)
    in_files = args.input
    out_files = get_output_filenames(args)
    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    # from ptflops import get_model_complexity_info
    # from fvcore.nn import FlopCountAnalysis 

   

    # flops = FlopCountAnalysis(net, torch.randn(1, 3, 572, 572))
    # print("total flops: ", flops.total())
    # print("flops.by_module_and_operator():", flops.by_module_and_operator())

    # macs, params = get_model_complexity_info(net, (3, 572, 572), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # EMED-UNet crosses the accuracy of U-Net with significantly reduced parameters from 31.043M to 6.72M, FLOPs from 386 G to 114 G, and model size from 386 MB to 80 MB. 
    # output: 
    # Computational complexity:       199.22 GMac (Giga(10 ** 9) multiply or add calculation, 有的时候也用 FLOPs : floating point operations)
    # Number of parameters:           17.26 M 

    # 计算量：
    # 卷积乘：
    # (Kw*Kh)*(Mh*Mw)*(Cin*Cout)

    # 卷积加：
    # Kw*Kh*Cin-1*Mh*Mw*Cout

    # Ps：n个数相加，相加次数是n-1。

    # Bias加：
    # Mh*Mw*Cout

    # 总：
    # No Bias：（2*Kw*Kh*Cin-1）*Mh*Mw*Cout

    # bias: 2*Kw*Kh*Cin*Mh*Mw*Cout

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
