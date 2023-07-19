# Model Design
#
# This is a model design to do 2x, 4x, 8x upscaling.

import os

import webdataset as wds

from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig, Checkpoint
from ray.train.huggingface import AccelerateTrainer
from ray.air import session

from accelerate import Accelerator

from gigagan_pytorch import  UnetUpsampler, Discriminator, TextEncoder, VisionAidedDiscriminator
from gigagan_pytorch.open_clip import OpenClipAdapter

import torch
import torch.nn as nn
import lpips
from torchvision import transforms

import wandb


# Resume
#
# Optional params for resuming runs
# wandb_run_path = None
# wandb_model_name = None
wandb_run_path = os.getenv("WANDB_RUN_PATH")
wandb_model_name = os.getenv("WANDB_MODEL_NAME")


# Will be set by restore if the above are set 
restored_model = None

if wandb_run_path and wandb_model_name:
    print("Restoring model from wandb")
    print("wandb_run_path: ", wandb_run_path)
    print("wandb_model_name: ", wandb_model_name)

    wandb.init(wandb_run_path=wandb_run_path)
    restored_model_path = f"{wandb_model_name}.h5"
    wandb.restore(restored_model_path, run_path=wandb_run_path)
else:
    wandb.init(project="open-gigagan")
    
# hyper-parameters
# Small image size(Concatted as a channel)
dim_text_feature = 64
num_epochs = 1000

use_GLU_discrim = True

# 4 machines 8 chips
NUM_WORKERS = 8 * 4

stat_log_rate = 5
model_log_rate = 100
ray_log_rate = 100

# We have lots small cards and lots of distributed nodes we'll probably want to make a big 
# batch size and do gradient accumulation to avoid passing model weights around too much. 
batch_size_config = 16
gradient_accumulation_steps = 8


model_opts = {
    "2xUpscaling": {
        "input_image_size": 128,
        "ouput_image_size": 256,
    },
    "4xUpscaling": {
        "input_image_size": 128,
        "ouput_image_size": 512,
    }
}

model_type = "4xUpscaling"

current_model_opts = model_opts[model_type]

# A config dict to save config settings to logging
config = {
    "model_type": model_type,
    "num_epochs": num_epochs,
    "use_GLU_discrim": use_GLU_discrim,
    "NUM_WORKERS": NUM_WORKERS,
    "stat_log_rate": stat_log_rate,
    "model_log_rate": model_log_rate,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "current_model_opts": current_model_opts,
}

wandb.config = config



def log_ray(loss, epoch):
    # Report and record metrics, checkpoint model at end of each
    # epoch
    session.report( {"loss": loss, "epoch": epoch})

def log(input):
    # stats
    if input["epoch"] % stat_log_rate == 0:
        epoch = input["epoch"]
        loss = input["loss"].item()
        print(f"epoch: {epoch}/{num_epochs}, loss: {loss:.3f}")
        wandb.log({
            "loss": loss,
            "epoch": epoch,
            "plip_loss": input["plip_loss"].item(),
            "clip_loss": input["clip_loss"].item(),
            "discrim_loss": input["discrim_loss"].item()
        })

    if input["epoch"] % model_log_rate == 0:
        epoch = input["epoch"]
        model = input["model"]
        torch.save(model.state_dict(), f"model_{epoch}.pth")
        wandb.save(f"model_{epoch}.pth")

    # log ray
    # We don't log much hear, more of a heartbeat for the dashboard
    if input["epoch"] % ray_log_rate == 0:
        log_ray(input["loss"], input["epoch"])


def get_wds():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preproc = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    url = "s3://paperspace-gradient-storage/datasets/LAION-High-Res-89-WEBP-2048-ma-3.0/"


    # Load dataset
    dataset = (
        wds.WebDataset(url)
        .shuffle(1000)
        .decode("rgb")
        .to_tuple("png", "json")
        .map_tuple(preproc, identity)
    )
    
    return dataset


class GigaGANTextConditionedUpscaler(nn.Module):
    def __init__(self):
        super(GigaGANTextConditionedUpscaler, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='alex')
        self.vision_aided_discriminator = VisionAidedDiscriminator()
        self.discriminator = Discriminator(use_glu=use_GLU_discrim)

        upsampler_dim = 128
        text_encoder_dim = 128
        text_encoder_depth = 4

        clip_adapter = OpenClipAdapter()

        text_encoder = TextEncoder(text_encoder_dim, text_encoder_depth, clip_adapter)

        self.upsampler = UnetUpsampler(
            dim=upsampler_dim,
            cross_attention=True,
            text_encoder=text_encoder,
            unconditional=False,
            **model_opts
        )

    def loss(self, inputs, outputs, texts):
        # Compute both discriminators for loss
        clip_loss = self.model.vision_aided_discriminator(outputs, texts)
        discrim_loss = self.model.discriminator(inputs, outputs, texts)
        plip_loss = self.model.lpip_loss(inputs, outputs)

        plips_loss_strength = 1
        clip_loss_strength = 1
        discrim_loss_strength = 1

        loss = plips_loss_strength * plip_loss +  \
            clip_loss * clip_loss_strength + \
            discrim_loss_strength * discrim_loss

        return loss, {
            "plip_loss": plip_loss,
            "clip_loss": clip_loss,
            "discrim_loss": discrim_loss,
        }

    def forward_train(self, images_in, texts):
        out = self.upsampler(images_in, texts=texts, return_all_rgbs=True)

        return out



# Takes crops of the images, prepares them for the upscaling task
class CroppedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Create a list to store crops and their labels
        labels = []

        # Define the range of possible base sizes for 64->256

        upscale_size = 4.0

        # Select random
        in_size = current_model_opts["input_image_size"]
        out_size = current_model_opts["ouput_image_size"]

        # Define the grid size for cropping

        # Use a FiveCrop transform to get crops from all corners and the center
        five_crop = transforms.FiveCrop(out_size)
        five_crops = five_crop(image)

        # Resize the crops
        resize = transforms.Resize((1/upscale_size, 1/upscale_size))
        resized_crops = [resize(crop) for crop in five_crops]

        # Create labels for each crop
        for i, (crop, resized) in enumerate(zip(five_crops, resized_crops)):
            crop_name = f"crop: {['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'][i]}"
            w, h = in_size
            ow, oh = out_size

            crop_label = f"{label} base: {w}x{h}, final: {ow}x{oh}, {crop_name}"

            labels.append(crop_label)

        return resized_crops, five_crops, labels

    def __len__(self):
        return len(self.dataset)


def train_wds(url):
    def distributed_training_loop():
        # Model and optimizer
        model = GigaGANTextConditionedUpscaler()
        model.load(restored_model_path)
        model.train()

        # opt_gen = Lion(model.generator.parameters(), lr=1e-4, weight_decay=1e-2)
        opt_gen = torch.optim.Adam(model.generator.parameters(), lr=1e-4)

        opt_disc = torch.optim.Adam(
            model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Initialize the Accelerator
        accelerator = Accelerator()

        # Fetch training set from the session
        dataset_shard = session.get_dataset_shard("train")

        model, opt_gen, opt_disc = accelerator.prepare(
            model, opt_gen, opt_disc)

        for epoch in range(num_epochs):
            for batches in dataset_shard.iter_torch_batches(
                batch_size=batch_size_config, dtypes=[torch.float, torch.float]
            ):
                with accelerator.accumulate(model):
                    x, y, text = batches
                    # Train model here
                    # model outputs
                    outputs = model(x, text)

                    loss, all_loss = model.loss(y, outputs)

                    # Zero out grads, do backward, and update optimizer
                    opt_disc.zero_grad()
                    opt_gen.zero_grad()
                    accelerator.backward(loss)
                    opt_disc.step()
                    opt_gen.step()

            log({
                "epochA:": epoch,
                "loss": loss,
                **all_loss,
                model: model,
            })

            


    # Define scaling and run configs
    scaling_config = ScalingConfig(num_workers=NUM_WORKERS)
    run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

    train_dataset = get_wds()

    trainer = AccelerateTrainer(
        train_loop_per_worker=distributed_training_loop,
        accelerate_config={
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_dataset},
    )

    result = trainer.fit()

    return result

def identity(x):
    return x