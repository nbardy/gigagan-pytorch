import torch
from gigagan_pytorch import GigaGAN, ImageDataset
import argparse


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

LOG_EVERY = 1
import torch
import torch_xla
import torch_xla.core.xla_model as xm

BATCH_SIZE = 1


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

def main(*args):
    print("Getting xla device")
    # first arg
    index = args[0]
    group_id = args[1]
    print("XMP_INDEX", index)
    print("GROUP_ID", group_id)

    wandb.init(project="gigagan", group=group_id)
    print("Geting xla")
    device = xm.xla_device()
    print("init process group")
    # dist.init_process_group('xla', init_method='pjrt://')
    print("init process group done")


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


    print("broadcasting")
    print("flushing", flush=True)
    gan.to(device)

    # runtime warning says its depericated
    # but it works and the belw doesn't
    pjrt.broadcast_master_param(gan)
    # torch_xla.core.xla_model.broadcast_master_param(gan)
    print("broadcasting done")

    dataloader = dataset.get_dataloader(batch_size = BATCH_SIZE)
    mp_device_loader = pl.MpDeviceLoader(dataloader, device)

    # training the discriminator and generator alternating
    # for 100 steps in this example, batch size 1, gradient accumulated 8 times

    print("makig gan")
    gan(
        dataloader = mp_device_loader,
        steps = 100,
        grad_accum_every = 1,
        xm=xm,
        batch_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    # Use argparse to get group_id

    argparser = argparse.ArgumentParser()

    # group_id 
    print("parsing")
    argparser.add_argument("--group_id", type=str, required=True)

    args = argparser.parse_args()

    group_id = args.group_id

    print("group_id", group_id)


    xmp.spawn(main, args=(group_id,))