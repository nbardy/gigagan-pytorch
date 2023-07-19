# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
from argparse import REMAINDER, ArgumentParser
from pathlib import Path

import torch_xla.distributed.xla_multiprocessing as xmp
import torch
from gigagan_pytorch import GigaGAN, ImageDataset
import wandb

LOG_EVERY = 1


def _mp_fn(index, flags=None):
    """
    Function to be run on each TPU core.
    """
    tpu_on = False
    if tpu_on:
        device = torch_xla.core.xla_model.xla_device()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = ImageDataset(
        folder='/home/nicholasbardy/Upscale_test',
        image_size=256
    )

    gan = GigaGAN(
        train_upsampler=True,
        generator=dict(
            style_network=dict(
                dim=64,
                depth=4
            ),
            dim=64,
            image_size=256,
            input_image_size=128,
            unconditional=True
        ),
        discriminator=dict(
            dim=64,
            dim_max=512,
            image_size=256,
            num_skip_layers_excite=4,
            unconditional=True
        ),
        log_steps_every=LOG_EVERY
    )

    gan.to(device)

    dataloader = dataset.get_dataloader(batch_size=1)

    gan(
        dataloader=dataloader,
        steps=100,
        grad_accum_every=1
    )


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(description="PyTorch TPU distributed training launch helper utility")

    # Optional arguments for the launch helper
    parser.add_argument("--num_cores", type=int, default=1, help="Number of TPU cores to use (1 or 8).")

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()

    # Patch sys.argv
    sys.argv = [args.training_script_args] + ["--tpu_num_cores", str(args.num_cores)]

    xmp.spawn(_mp_fn, args=())


if __name__ == "__main__":
    wandb.init(project="gigagan")
    main()
