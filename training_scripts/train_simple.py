from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import lpips
from PIL import Image
from datasets import load_dataset
import io
import torch
import requests


from torchvision import transforms
from PIL import Image
import io

from torch.utils.data import Dataset, DataLoader


from datasets import load_dataset


# NOTE: Install repo from root to get this to wor
from gigagan_pytorch import  UnetUpsampler, Discriminator, TextEncoder, VisionAidedDiscriminator, StyleNetwork
from gigagan_pytorch.open_clip import OpenClipAdapter

import wandb


# A lot of these are just for tesing the code right now
# A don't expect this to converge
#
# Hyper-parameters
num_epochs = 1000
image_log_rate = 10
dim_text_feature = 64
discriminator_dim = 128
use_GLU_discrim = True
model_type = "4xUpscaling"
batch_size_config = 16
image_input_size = 64
image_output_size = 256
upsampler_dim = 128

# TODO: Is this the global text code?
# Or is this global+local?
# in the paper global is much smaller than local
#
# Need to follow up on these numbers
text_encoder_dim = 8
text_encoder_depth = 4
style_network_depth = 32
style_network_dim = 64


class GigaGANTextConditionedUpscaler(nn.Module):
    def __init__(self):
        super(GigaGANTextConditionedUpscaler, self).__init__()

        clip_adapter = OpenClipAdapter()
        text_encoder = TextEncoder(dim=text_encoder_dim, depth=text_encoder_depth, clip=clip_adapter)

        # Load all the models for loss
        self.discriminator = Discriminator(dim=discriminator_dim, 
            image_size=image_output_size,
            use_glu=use_GLU_discrim,
            text_encoder=text_encoder,
        )

        self.vision_aided_discriminator = VisionAidedDiscriminator(clip=clip_adapter)

        self.lpips_loss = lpips.LPIPS(net='alex')


        self.upsampler = UnetUpsampler(
            input_image_size=image_input_size,
            image_size=image_output_size,
            dim=upsampler_dim,
            text_encoder=text_encoder,
            unconditional=False,
            style_network=StyleNetwork(
                dim=style_network_dim,
                dim_text_latent=text_encoder_dim,
                depth=style_network_depth
                ),
        )

    def loss(self, inputs, outputs, texts):
        # Compute both discriminators for loss
        clip_loss = self.vision_aided_discriminator(outputs, texts)
        discrim_loss = self.discriminator(inputs, outputs, texts)
        plip_loss = self.lpip_loss(inputs, outputs)

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
    
    def forward_test(self, images_in, texts):
        out = self.upsampler(images_in, texts=texts)

        return out


def train(model, train_loader, num_epochs):
    opt_gen = torch.optim.Adam(model.upsampler.parameters(), lr=1e-4)
    opt_disc = torch.optim.Adam(
        model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    model.train()
    for epoch in range(num_epochs):
        for idx, sample in enumerate(train_loader):
            text = sample['text']
            x = sample['image_in']
            y = sample['image_out']

            outputs = model.forward_train(x, text)
            loss, all_loss = model.loss(y, outputs, text)

            # Zero out grads, do backward, and update optimizer
            opt_disc.zero_grad()
            opt_gen.zero_grad()
            loss.backward()
            opt_disc.step()
            opt_gen.step()

        if epoch % image_log_rate == 0:
            image = model.forward_test(x, text)
            image_log = image.detach().cpu().numpy()
            x_log = x.detach().cpu().numpy()

            image_log = Image.fromarray(image_log)
            x_log = Image.fromarray(x_log)

            wandb.log({"imageOut": image_log, "imageIn": x_log, "epoch": epoch, "loss": loss})


class DatasetWithTransform(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        if transform is None:
            raise Exception("Must provide transform")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print keys of dict
        example = self.data[idx]
        image = self.transform(example)
        return image


def get_ds(input_size=64, output_size=256):
    # Define your transformations
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Let's get a random crop o
    preproc = transforms.Compose([
        transforms.RandomCrop(output_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # Make another that will resize to input size
    preproc_input = transforms.Compose([
        transforms.Resize(input_size),
    ])


    # Function to transform the images
    def transform_features(example):
        url = example['URL']
        try:
            response = requests.get(url)

            if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                image_content = response.content
                image = Image.open(io.BytesIO(image_content)).convert('RGB')
            else:
                return None
        except Exception as e:
            print(e)
            return None

        image = preproc(image)
        input_image = preproc_input(image)


        return {'image_in': input_image, 'image_out': image, 'text': example['TEXT']}  
    


    # Load your dataset
    dataset = load_dataset("laion/laion-high-resolution", split='train[:10%]')


    # Slice the dataset to the first 1000 images
    print(type(dataset))

    # Apply the transformations
    dataset = DatasetWithTransform(dataset, transform_features)


    # collate_fn takes the list of samples and makes a batch, which is a dictionary of tensors
    def collate_fn(examples):
        # filter none values
        examples = [example for example in examples if example is not None]


        # stack images
        images_in = torch.stack([example['image_in'] for example in examples])
        images_out = torch.stack([example['image_out'] for example in examples])

        # stack text as list
        texts = [example['text'] for example in examples]

        return {
            'image_in': images_in, 
            'image_out': images_out,
            'text': texts,
        }


    dataloader = DataLoader(dataset, shuffle=True, batch_size=5, collate_fn=collate_fn)

    return dataloader


def main():
    model = GigaGANTextConditionedUpscaler()

    # Here, add your own logic to load your dataset and wrap it in a DataLoader
    train_loader = get_ds()

    train(model, train_loader, num_epochs)


if __name__ == "__main__":
    main()
