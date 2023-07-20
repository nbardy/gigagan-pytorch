import torch
from gigagan_pytorch import GigaGAN, ImageDataset

import wandb
wandb.init(project="gigagan", job_type="simple")

LOG_EVERY = 1
print("Using CUDA")
device = torch.device("cuda")

dataset = ImageDataset(
    folder = '/home/paperspace/datasets/upscale/',
    image_size = 256,
    # lambda to do PIL convert RGB
    convert_image_to="RGB",
)

machine_id_path = "/home/nicholasbardy/machine_id"

def read_machine_id() -> int:
    """Read the unique integer from the given file path."""
    with open(machine_id_path, 'r') as f:
        return int(f.read().strip())

# Example usage
# unique_id = read_machine_id()
# print(unique_id)


output_image_size = 256 * 4
generator_dim = 64 * 8
num_conv_kernels = 1024

self_attn_dim_head = 64 * 64
self_attn_heads = 8 * 2
self_attn_ff_mult = 4


print("building gan")

#
# gan = GigaGAN(
#     train_upsampler = True,     # set this to True
#     generator = dict(
#         style_network = dict(
#             dim = 128,
#             depth = 4
#         ),
#         dim = generator_dim,
#         image_size = output_image_size,
#         num_conv_kernels = num_conv_kernels,
#         self_attn_dim_head = self_attn_dim_head,
#         self_attn_heads = self_attn_heads,
#         self_attn_ff_mult = self_attn_ff_mult,
#         input_image_size = 128,
#         unconditional = True
#     ),
#     discriminator = dict(
#         dim_max = 512,
#         image_size = output_image_size,
#         num_skip_layers_excite = 4,
#         unconditional = True
#     ),
#    log_steps_every=LOG_EVERY
#)

print("Gan made")

gan.to(device)

print("gan moved")

dataloader = dataset.get_dataloader(batch_size = 1)

# training the discriminator and generator alternating
# for 100 steps in this example, batch size 1, gradient accumulated 8 times

print("gan training")
gan(
    dataloader = dataloader,
    steps = 100,
    grad_accum_every = 1
)
