import torch
from gigagan_pytorch import GigaGAN, ImageDataset
import webdataset as wds
from torchvision import transforms
import argparse

import wandb



# Set up the argument parser
parser = argparse.ArgumentParser(description='GigaGAN Training Script')
parser.add_argument('--group_id', default='experiment_1', type=str, help='Group ID for wandb logging')
args = parser.parse_args()

wandb.init(project="gigagan", group=args.group_id)


# Load dataset
def identity(x):
    return x

LOG_EVERY = 1

tpu_on = True
if tpu_on is True:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
elif torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_wds(text=False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preproc = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
    ])

    # url = "pipe:aws s3 cp s3://paperspace-gradient-storage/datasets/LAION-High-Res-89-WEBP-2048-ma-3.0/{00000..00999}.tar - || true"


    # non f string
    data_range = "{00000..00999}"
    data_range =  "{00000..00001}"
    url = f"pipe:s3cmd get --quiet --continue s3://paperspace-gradient-storage/datasets/LAION-High-Res-89-WEBP-2048-ma-3.0/{data_range}.tar -"

    dataset = (
        wds.WebDataset(url)
        .shuffle(1000)
        .decode("rgb")
    )

    if text is True:
        dataset = dataset.to_tuple("webp", "txt")
        dataset = dataset.map_tuple(preproc, identity)
    else:
        dataset = dataset.map(lambda x: x["webp"])
        dataset = dataset.map(preproc)

    return dataset

dataset = get_wds()

from itertools import islice

gan = GigaGAN(
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
    log_steps_every=LOG_EVERY
)

gan.to(device)

# training the discriminator and generator alternating
# for 100 steps in this example, batch size 1, gradient accumulated 8 times

dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)

gan(
    dataloader = dataloader,
    steps = 100,
    grad_accum_every = 1
)
