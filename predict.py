import dotenv

dotenv.load_dotenv(override=True)

import os
import random
from datetime import datetime
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from accelerate import Accelerator
from cog import BasePredictor, Input, Path
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from omnigen2.utils.img_util import create_collage


NEGATIVE_PROMPT = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.weight_dtype = torch.bfloat16
        self.pipeline = self.load_pipeline()

    def load_pipeline(self):
        pipeline = OmniGen2Pipeline.from_pretrained(
            "OmniGen2/OmniGen2",
            torch_dtype=self.weight_dtype,
            trust_remote_code=True,
        )
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            "OmniGen2/OmniGen2",
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )
        pipeline = pipeline.to(self.accelerator.device)
        return pipeline

    def predict(
        self,
        instruction: str = Input(description="Instruction for the model"),
        image_input_1: Path = Input(description="First input image", default=None),
        image_input_2: Path = Input(description="Second input image", default=None),
        image_input_3: Path = Input(description="Third input image", default=None),
        width_input: int = Input(description="Width of the output image", default=1024),
        height_input: int = Input(description="Height of the output image", default=1024),
        scheduler: str = Input(description="Scheduler to use", choices=["euler", "dpmsolver++"], default="euler"),
        num_inference_steps: int = Input(description="Number of inference steps", default=50),
        negative_prompt: str = Input(description="Negative prompt", default=NEGATIVE_PROMPT),
        guidance_scale_input: float = Input(description="Text guidance scale", default=5.0),
        img_guidance_scale_input: float = Input(description="Image guidance scale", default=2.0),
        cfg_range_start: float = Input(description="CFG range start", default=0.0),
        cfg_range_end: float = Input(description="CFG range end", default=1.0),
        num_images_per_prompt: int = Input(description="Number of images per prompt", default=1),
        max_input_image_side_length: int = Input(description="Maximum input image side length", default=2048),
        max_pixels: int = Input(description="Maximum pixels", default=1024 * 1024),
        seed_input: int = Input(description="Seed for random number generation", default=-1),
    ) -> Path:
        """Run a single prediction on the model"""
        input_images = [image_input_1, image_input_2, image_input_3]
        input_images = [img for img in input_images if img is not None]

        if len(input_images) == 0:
            input_images = None
        else:
            input_images = [to_pil_image(to_tensor(Image.open(str(img)))) for img in input_images]


        if seed_input == -1:
            seed_input = random.randint(0, 2**16 - 1)

        generator = torch.Generator(device=self.accelerator.device).manual_seed(seed_input)

        if scheduler == 'euler':
            self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
        elif scheduler == 'dpmsolver++':
            self.pipeline.scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )

        results = self.pipeline(
            prompt=instruction,
            input_images=input_images,
            width=width_input,
            height=height_input,
            max_input_image_side_length=max_input_image_side_length,
            max_pixels=max_pixels,
            num_inference_steps=num_inference_steps,
            max_sequence_length=1024,
            text_guidance_scale=guidance_scale_input,
            image_guidance_scale=img_guidance_scale_input,
            cfg_range=(cfg_range_start, cfg_range_end),
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            output_type="pil",
        )

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        output_path = os.path.join(output_dir, f"{timestamp}.png")

        results.images[0].save(output_path)

        return Path(output_path)
