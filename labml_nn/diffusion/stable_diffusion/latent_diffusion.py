"""
---
title: Latent Diffusion Models
summary: >
 Annotated PyTorch implementation/tutorial of latent diffusion models from paper
 High-Resolution Image Synthesis with Latent Diffusion Models
---

# Latent Diffusion Models

Latent diffusion models use an auto-encoder to map between image space and
latent space. The diffusion model works on the latent space, which makes it
a lot easier to train.
It is based on paper
[High-Resolution Image Synthesis with Latent Diffusion Models](https://papers.labml.ai/paper/2112.10752).

They use a pre-trained auto-encoder and train the diffusion U-Net on the latent
space of the pre-trained auto-encoder.

For a simpler diffusion implementation refer to our [DDPM implementation](../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math

from labml_nn.diffusion.stable_diffusion.model.autoencoder import Autoencoder
from labml_nn.diffusion.stable_diffusion.model.clip_embedder import CLIPTextEmbedder
from labml_nn.diffusion.stable_diffusion.model.unet import UNetModel


class DiffusionWrapper(nn.Module):
    """
    *This is an empty wrapper class around the [U-Net](model/unet.html).
    We keep this to have the same model structure as
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
    so that we do not have to map the checkpoint weights explicitly*.
    """

    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    """
    ## Latent diffusion model

    This contains following components:

    * [AutoEncoder](model/autoencoder.html)
    * [U-Net](model/unet.html) with [attention](model/unet_attention.html)
    * [CLIP embeddings generator](model/clip_embedder.html)
    """
    model: DiffusionWrapper
    first_stage_model: Autoencoder
    cond_stage_model: CLIPTextEmbedder

    def __init__(self,
                 unet_model: UNetModel,
                 autoencoder: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 step_size_eps: float,
                 linear_start: float,
                 linear_end: float,
                 ):
        """
        :param unet_model: is the [U-Net](model/unet.html) that predicts noise
         $\epsilon_\text{cond}(x_t, c)$, in latent space
        :param autoencoder: is the [AutoEncoder](model/autoencoder.html)
        :param clip_embedder: is the [CLIP embeddings generator](model/clip_embedder.html)
        :param latent_scaling_factor: is the scaling factor for the latent space. The encodings of
         the autoencoder are scaled by this before feeding into the U-Net.
        :param n_steps: is the number of diffusion steps $T$.
        :param linear_start: is the start of the $\beta$ schedule.
        :param linear_end: is the end of the $\beta$ schedule.
        """
        super().__init__()
        # Wrap the [U-Net](model/unet.html) to keep the same model structure as
        # [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
        self.model = DiffusionWrapper(unet_model)
        # Auto-encoder and scaling factor
        self.first_stage_model = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        # [CLIP embeddings generator](model/clip_embedder.html)
        self.cond_stage_model = clip_embedder

        # Number of steps $T$
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, 1000, dtype=torch.float64) ** 2
        torch.set_printoptions(precision=10)
        # $\alpha_t = 1 - \beta_t$
        alpha = 1. - beta
        plt.plot(beta, label='old-betas')
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        their_times = -0.5*torch.log(alpha_bar)
        end_time = their_times[-1]
        # print("THEIR TIMES", self.their_times)
        # print("TIME END ", end_time)
        betas = []
        step_size = 1
        times = []
        curr_t = end_time
        while curr_t > 0.00001:
            times.append(curr_t.item())
            step_size = step_size_eps*(1 - math.exp(-2*curr_t))
            curr_beta = 1 - math.exp(-2*step_size)
            betas.append(curr_beta)
            curr_t -= step_size
        self.n_steps = len(betas)
        self.times = torch.tensor(times[::-1])
        # self.times = list(range(n_steps))
        beta = torch.tensor(betas[::-1])
        # print("our betas", beta, self.n_steps)
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        # $\alpha_t = 1 - \beta_t
        alpha = 1. - beta
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)
        # print(alpha_bar[-1])
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
        our_times = -0.5 * torch.log(alpha_bar)
        # print(self.times - self.our_times)
        # plt.plot(beta, label='our-betas')
        # plt.legend()
        # plt.savefig('outputs/plot.jpeg')

        def find_in_list(l, v):
            for i in range(len(l)):
                if l[i] >= v:
                    if abs(l[max(0, i-1)] - v) < abs(l[i] - v):
                        return max(0, i-1)
                    return i
            return len(l) - 1
        our_time_indices = []
        for t in our_times:
            our_time_indices.append(find_in_list(their_times, t))
        self.our_time_indices = torch.tensor(our_time_indices)

    @property
    def device(self):
        """
        ### Get model device
        """
        # print("PARAMETERS", self.model.paramters())
        return next(iter(self.model.parameters())).device

    def get_text_conditioning(self, prompts: List[str]):
        """
        ### Get [CLIP embeddings](model/clip_embedder.html) for a list of text prompts
        """
        return self.cond_stage_model(prompts)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### Get scaled latent space representation of the image

        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation

        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        """
        ### Predict noise

        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.

        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, context)
