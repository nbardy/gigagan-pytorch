## GigaGAN Upscaler Training 

This is a fork from [lucidrains's GigaGAN](https://github.com/lucidrains/lightweight-gan). I am working on the working_branch branch here to develop training scripts for a distributed training run on TPU preview chips. The open source work is sponsored by [Facet](https://facet.ai/) and the compute by [Google](http://google.com)

The weights will be open sourced.

Training logs are open here:
https://wandb.ai/nbardy-facet/gigagan?workspace=user-nbardy-facet


## TODO
- [x] Get original training scripts running locally
- [x] Connect compute cluster with ray 
- [x] Write a draft of a ray/accelerate cluster training script.
- [x] Setup on LAION High Res webdataset
- [ ] Get running on TPU (Currently working here
- [ ] Train a few base lines
- [ ] Launch hyper-parameter sweeps on the small model
- [ ] Scale up to a bigger model with distributed training run.
    - [ ] Port the draft ray training script over to lucidrain's new training code.
    - [ ] Check throughput
- [ ] Reproduce Upscaler with result quality compared to the paper at 256px=>1028px(4X)
- [ ] Reproduce Upscaler with result quality compared to the paper at 128px>1028px(8X)

Stretch Goals
- [ ] Reproduce a STOTA(For SpeedXQuality open T2I pipeline)
 - Arbitrary aspect ratios.
 - [ ]Update architecture to work on patches and make the architecture independant of the patch count so we can scale upt
 - Training in series with a fast model that does thumbnails or openMUSE or PAELLA.


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
