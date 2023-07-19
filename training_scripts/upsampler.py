import torch
from gigagan_pytorch import GigaGAN, ImageDataset

import wandb
wandb.init(project="gigagan")

LOG_EVERY = 1

tpu_on = False
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

dataset = ImageDataset(
    folder = '/Users/nicholasbardy/Downloads/Upscale_test',
    image_size = 256
)


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

dataloader = dataset.get_dataloader(batch_size = 1)

# training the discriminator and generator alternating
# for 100 steps in this example, batch size 1, gradient accumulated 8 times

gan(
    dataloader = dataloader,
    steps = 100,
    grad_accum_every = 1
)