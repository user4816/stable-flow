import argparse
import os

import torch
from diffusers import FluxPipeline
import numpy as np
from PIL import Image

def set_gpu_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

class StableFlow:
    MULTIMODAL_VITAL_LAYERS = [0, 1, 17, 18]
    SINGLE_MODAL_VITAL_LAYERS = list(np.array([28, 53, 54, 56, 25]) - 19) 

    def __init__(self):
        self._parse_args()
        set_gpu_device(self.args.gpu_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev")
        parser.add_argument("--hf_token", type=str, required=True)
        parser.add_argument("--prompts", type=str, nargs="+", required=True)
        parser.add_argument("--output_path", type=str, default="outputs/result_alpha{attention_alpha}.jpg")
        parser.add_argument("--input_img_path", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--cpu_offload", action="store_true")
        parser.add_argument("--gpu_id", type=int, default=0,help="0, 1, 2..")
        parser.add_argument("--attention_alpha", type=float, default=1.0)

        self.args = parser.parse_args()

        alpha_str = str(self.args.attention_alpha).replace('.', '_')
        self.args.output_path = self.args.output_path.format(attention_alpha=alpha_str)
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)

    def _load_pipeline(self):
        self.pipe = FluxPipeline.from_pretrained(
            self.args.model_path, 
            torch_dtype=torch.float16,
            visualize_attention=False,
            token=self.args.hf_token
        )

        if self.args.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(self.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        generator = torch.Generator(device=self.device).manual_seed(self.args.seed)
        latents = torch.randn(
            (4096, 64), 
            generator=generator, 
            device=self.device, 
            dtype=torch.float16
        ).tile(len(prompts), 1, 1)
        images = self.pipe(
            prompts,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            output_type="pil",
            num_inference_steps=15,
            max_sequence_length=512,
            latents=latents,
            mm_copy_blocks=StableFlow.MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=StableFlow.SINGLE_MODAL_VITAL_LAYERS,
        ).images
        images = [np.array(img) for img in images]
        res = Image.fromarray(np.hstack((images)))
        res.save(self.args.output_path)

    @torch.no_grad()
    def image2latent(self, image, latent_nudging_scalar=1.15):
        image = self.pipe.image_processor.preprocess(image).type(self.pipe.vae.dtype).to(self.device)
        latents = self.pipe.vae.encode(image)["latent_dist"].mean
        latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        latents = latents * latent_nudging_scalar
        latents = self.pipe._pack_latents(
            latents=latents,
            batch_size=1,
            num_channels_latents=16,
            height=128,
            width=128
        )
        return latents

    @torch.no_grad()
    def invert_and_save(self, prompts):
        inversion_prompt = prompts[0:1]
        latents_for_inversion = self.image2latent(Image.open(self.args.input_img_path))
        inverted_latent_list = self.pipe(
            inversion_prompt,
            height=1024,
            width=1024,
            guidance_scale=1,
            output_type="pil",
            num_inference_steps=50,
            max_sequence_length=512,
            latents=latents_for_inversion,
            invert_image=True
        )
        edited_latents = inverted_latent_list[-1].tile(len(prompts), 1, 1)

        ## Changed
        generator = torch.Generator(device=edited_latents.device).manual_seed(self.args.seed)
        new_generated_latents = torch.randn(
            edited_latents.shape,
            dtype=edited_latents.dtype,
            device=edited_latents.device,
            generator=generator
        )
              
        ## Changed
        alpha = self.args.attention_alpha
        combined_latents = alpha * edited_latents + (1 - alpha) * new_generated_latents

        ## Changed
        images = self.pipe(
            prompts,
            height=1024,
            width=1024,
            guidance_scale=[1] + [3] * (len(prompts) - 1),
            output_type="pil",
            num_inference_steps=50,
            max_sequence_length=512,
            latents=combined_latents,
            inverted_latent_list=inverted_latent_list,
            mm_copy_blocks=StableFlow.MULTIMODAL_VITAL_LAYERS,
            single_copy_blocks=StableFlow.SINGLE_MODAL_VITAL_LAYERS,
        ).images
        images = [np.array(img) for img in images]
        res = Image.fromarray(np.hstack((images)))
        res.save(self.args.output_path)

if __name__ == "__main__":
    stable_flow = StableFlow()
    if stable_flow.args.input_img_path is None:
        stable_flow.infer_and_save(prompts=stable_flow.args.prompts)
    else:
        stable_flow.invert_and_save(prompts=stable_flow.args.prompts)
