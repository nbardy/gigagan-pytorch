## GigaGAN Upscaler Training 

<img width="340" alt="image" src="https://github.com/nbardy/gigagan-pytorch/assets/1278972/0504550c-e81a-4db2-86ee-8421ccca0e4e">

### Intro
This is a fork from [lucidrains's GigaGAN](https://github.com/lucidrains/lightweight-gan). I am working here to develop training scripts for a distributed training run of a text conditioned upsampler.

### Notes
Q:Will the Weights be Open Sourced?
A:The weights will be open sourced.

The open source work is sponsored by [Facet](https://facet.ai/) and the compute by [Google](http://google.com)

Training logs are open here:
https://wandb.ai/nbardy-facet/gigagan?workspace=user-nbardy-facet


## TODO
- [x] Get original training scripts running locally
- [x] Connect compute cluster with ray 
- [x] Write a draft of a ray/accelerate cluster training script.
- [x] Setup on LAION High Res webdataset
- [x] Get running on TPU
- [ ] Get distributed training running
  - [x] Trying accelerate
  - [x] Trying XMP
- [ ] Train a few base lines
- [ ] Launch hyper-parameter sweeps on the small model
- [ ] Reproduce Upscaler with result quality compared to the paper at 256px=>1028px(4X)
- [ ] Reproduce Upscaler with result quality compared to the paper at 128px>1028px(8X)

Stretch Goals
- [ ] Reproduce a STOTA(For SpeedXQuality open T2I pipeline)
 - [ ] Arbitrary aspect ratios.
   - [ ] Update architecture to work on patches(Possibly get a speed here, can we modify patch size to be more effecient)
   - [ ] Modify architecture to work on arbitrary number of given patches
 - [ ] Train in series with a fast model that does thumbnails(openMUSE or PAELLA)

## Appreciation (Come's form the original repo)

- <a href="https://stability.ai/">StabilityAI</a> for the sponsorship, as well as my other sponsors, for affording me the independence to open source artificial intelligence.

- <a href="https://huggingface.co/">🤗 Huggingface</a> for their accelerate library

- All the maintainers at <a href="https://github.com/mlfoundations/open_clip">OpenClip</a>, for their SOTA open sourced contrastive learning text-image models

- <a href="https://github.com/XavierXiao">Xavier</a> for reviewing the discriminator code and pointing out that the scale invariance was not correctly built!

## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.05511,
    url     = {https://arxiv.org/abs/2303.05511},
    author  = {Kang, Minguk and Zhu, Jun-Yan and Zhang, Richard and Park, Jaesik and Shechtman, Eli and Paris, Sylvain and Park, Taesung},  
    title   = {Scaling up GANs for Text-to-Image Synthesis},
    publisher = {arXiv},
    year    = {2023},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@article{Liu2021TowardsFA,
    title   = {Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis},
    author  = {Bingchen Liu and Yizhe Zhu and Kunpeng Song and A. Elgammal},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2101.04775}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```
