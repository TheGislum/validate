import os
import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt


import utils
import network
from network import load_model
from metrics import StreamSegMetrics, timeMetrics
from utils import ext_transforms as et


def get_argparser():
    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--has_variant", action='store_true', default=False,
                        help="use variant of model")
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument("--num_classes", type=int, default=21,
                        help="num classes (default: None)")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")

    return parser

def validate(opts, model, loader, device, metrics, timing, cl=15):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        model = model.eval()
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            sct = time.perf_counter()

            outputs = model(images)

            ect = time.perf_counter()
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            preds = (preds == cl).astype(np.int64)

            metrics.update(targets, preds)
            timing.update(ect - sct)

            if opts.save_val_results:
                    for i in range(len(images)):
                        image = images[i].detach().cpu().numpy()
                        target = targets[i]
                        pred = preds[i]

                        image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = loader.dataset.decode_target(target).astype(np.uint8)
                        pred = loader.dataset.decode_target(pred).astype(np.uint8)

                        Image.fromarray(image).save('results/%d_image.png' % img_id)
                        Image.fromarray(target).save('results/%d_target.png' % img_id)
                        Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                        fig = plt.figure()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.imshow(pred, alpha=0.7)
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        img_id += 1

        score = metrics.get_results()
        val_time = timing.get_results()
    return score, val_time

def predict(args, model, image_files, device):

    decode_fn = utils.custom_dataset.decode_target

    transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    
    with torch.no_grad():
        model = model.eval()
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            pred = model(img).max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            if args.save_val_results_to:
                colorized_preds.save(os.path.join(args.save_val_results_to, img_name+'.png'))

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_dst = utils.custom_dataset(root=opts.input, transform=val_transform)
    
    return val_dst

def main():
    args = get_argparser().parse_args()

    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup dataloader
    val_dst = get_dataset(args)
    if args.validate:
        val_loader = data.DataLoader(val_dst, batch_size=args.batch_size, shuffle=True, num_workers=2)
    else:
        image_files = []
        if os.path.isdir(args.input):
            for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
                files = glob(os.path.join(args.input, '**/*.%s'%(ext)), recursive=True)
                if len(files)>0:
                    image_files.extend(files)
        elif os.path.isfile(args.input):
            image_files.append(args.input)

    cl = 15

    if args.has_variant:
        model, variant = load_model(args.ckpt, device)
        model = model.to(device)
        dataset = variant['dataset_kwargs']['dataset']
        if dataset == 'ade20k':
            cl = 12
        elif dataset == 'cityscapes':
            cl = 11
        elif dataset == 'pascal':
            cl = 15
        
    else:
        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
        if args.separable_conv and 'plus' in args.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        
        if args.ckpt is not None and os.path.isfile(args.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            print("Resume model from %s" % args.ckpt)
            del checkpoint
        else:
            print("[!] No checkpoint found at '%s'" % args.ckpt)
            return

    # Set up metrics
    metrics = StreamSegMetrics(args.num_classes)
    timing = timeMetrics()

    if args.validate:
        val_score, val_time = validate(args, model, val_loader, device, metrics, timing, cl)
        print(metrics.to_str(val_score))
        print(timing.to_str(val_time))
    else:
        predict(args, model, image_files, device)
    

if __name__ == "__main__":
    main()