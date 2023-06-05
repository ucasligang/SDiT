# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torchvision as tv
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from diffusion.image_datasets import load_data
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader

PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        class_dropout_prob=args.class_dropout_prob,
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    print(ckpt_path)
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)

    model.eval()  # important!
    for name, param in model.named_parameters():
        #if 'y_embedder.embedding_table' in name:
        param.requires_grad = False
        print(name)
        print(param.requires_grad)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    print("creating data loader...")
    # data = load_data(
    #     dataset_mode=args.dataset_mode,
    #     data_dir=args.data_path,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    #     deterministic=True,
    #     random_crop=False,
    #     random_flip=False,
    #     is_train=False
    # )
    dataset = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_path,
        image_size=args.image_size,
        # class_cond=args.class_cond,
        random_crop=False,
        random_flip=False,
        is_train=False
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)

    print('sampling...')
    all_samples = []
    for i, (batch, cond) in enumerate(loader):
        image = ((batch+1.0) / 2.0).cuda()
        # label = (cond['label_ori'].float() / 255.0).cuda()
        label = (cond['label_ori']).cuda()
        y = cond['label_ori'].to(device)
        # Create sampling noise:
        n = len(y)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(y, device=device)
        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        # y_null = torch.tensor([150] * n, device=device)
        y_null = torch.ones(y.shape, device=device)*0
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        print(samples.shape)
        samples = vae.decode(samples / 0.18215).sample
        print(samples.shape)
        # Save and display images:

        for j in range(samples.shape[0]):
            save_image(image[j], os.path.join(image_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            #tv.utils.save_image(samples[j], os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            seg = np.array(label[j].cpu())
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(PALETTE):
                color_seg[seg == label, :] = color  # numpy 数组的“新奇”使用，就是把预测结果的灰度像素值改成RGB
                color_seg = color_seg[..., ::-1]  # convert to BGR （cv2的存储顺序是GBR,所以逆序读取RGB就行了）
                cv2.imwrite(os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'), color_seg)
                # save_image(os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0]+'.png'), torch.from_numpy(color_seg))

            # save_image(label[j], os.path.join(label_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'))
            save_image([samples[j]], os.path.join(sample_path, cond['path'][j].split('/')[-1].split('.')[0] + '.png'), nrow=1, normalize=True, value_range=(-1, 1))  # nrow=4

        if len(all_samples) * args.batch_size > args.num_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/pub/data/ligang/data/ADE/ADEChallengeData2016')
    parser.add_argument("--results-path", type=str, default="samples_result")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--class_cond", type=bool, default=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)

    parser.add_argument("--dataset_mode", type=str, default='ade20k')
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=150)
    parser.add_argument("--cfg-scale", type=float, default=0.0)  # 1.5 4.0
    parser.add_argument("--num-sampling-steps", type=int, default=250) # 250
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)  # 256
    parser.add_argument("--ckpt", type=str, default='/pub/data/ligang/projects/DiT/pretrained_models/DiT-XL-2-512x512.pt',
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
