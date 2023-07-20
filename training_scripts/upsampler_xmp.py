import torch
from gigagan_pytorch import GigaGAN, ImageDataset


import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt


import wandb
wandb.init(project="gigagan")

LOG_EVERY = 1
import torch
import torch_xla
import torch_xla.core.xla_model as xm

BATCH_SIZE = 4


dataset = ImageDataset(
    folder = '/home/nicholasbardy/Upscale_test',
    # folder = '/Users/nicholasbardy/Downloads/Upscale_test',
    image_size = 256,
    # lambda to do PIL convert RGB
    convert_image_to="RGB"
)

machine_id_path = "/home/nicholasbardy/machine_id"

def read_machine_id() -> int:
    """Read the unique integer from the given file path."""
    with open(machine_id_path, 'r') as f:
        return int(f.read().strip())

# Example usage
# unique_id = read_machine_id()
# print(unique_id)

gradient_penalty_every = 32

def main(index):
    print("Getting xla device")
    print("XMP_INDEX", index)
    device = xm.xla_device()
    dist.init_process_group('xla', init_method='pjrt://')

    gan = GigaGAN(
        apply_gradient_penalty_every=gradient_penalty_every,
        train_upsampler = True,     # set this to True
        generator = dict(
            style_network = dict(
                dim = 64,
                depth = 4
            ),
            dim = 64,
            image_size = 256,
            input_image_size = 128,
            unconditional = True
        ),
        discriminator = dict(
            dim = 64,
            dim_max = 512,
            image_size = 256,
            num_skip_layers_excite = 4,
            unconditional = True
        ),
        log_steps_every=LOG_EVERY,
    )


    gan.to(device)

    dataloader = dataset.get_dataloader(batch_size = BATCH_SIZE)
    mp_device_loader = pl.MpDeviceLoader(dataloader, device)

    # training the discriminator and generator alternating
    # for 100 steps in this example, batch size 1, gradient accumulated 8 times

    gan(
        dataloader = mp_device_loader,
        steps = 100,
        grad_accum_every = 8,
        xm=xm,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    xmp.spawn(main, args=())