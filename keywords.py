import os
import torch
from diffusers.utils import export_to_video

# -----------------------------
# Cache directory
# -----------------------------
CACHE_DIR = "E:/hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = CACHE_DIR

# Helps reduce VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# Imports AFTER cache setup
# -----------------------------
from diffusers import AnimateDiffPipeline, MotionAdapter
from diffusers import StableDiffusionXLPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Image Generation
# -----------------------------
def generate_image():

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()   # important for 6GB GPU

    image = pipe(
        """
Create a premium luxury perfume advertisement in vertical 4:5 format, designed for Instagram and high-end social media promotion.

Place an elegant crystal-clear perfume bottle at the center on a glossy reflective surface, with metallic gold and silver details, cinematic studio lighting, and sharp luxury reflections.

Use a bright vibrant luxury background with electric pink, turquoise blue, golden yellow, and soft purple gradients, enhanced by soft bokeh, glowing light streaks, floating particles, translucent smoke, and subtle abstract floral elements.

Keep the composition clean and symmetrical with clear negative space for large readable text.

Text overlay must be highly visible and professionally designed:
Top: "LUMÉA"
Center headline: "Awaken Your Senses"
Lower text: "A bold fragrance crafted for unforgettable moments."
Bottom CTA: "Shop Now" | "Limited Edition"

Typography should be the main focus: elegant luxury magazine style, premium serif + modern sans-serif fonts, white and gold text, sharp readability, strong contrast, subtle glow.

Add minimal premium extras: thin luxury borders, NEW ARRIVAL badge, soft text glow, refined Instagram ad layout.

Style: ultra realistic luxury perfume campaign, premium commercial photography, high-end branding, sharp focus, 8k.
        """,
        num_inference_steps=30
    ).images[0]

    image.save("output.png")

    del pipe
    torch.cuda.empty_cache()

    print("✅ Image saved locally")

# -----------------------------
# Video Generation
# -----------------------------

def generate_video():

    adapter = MotionAdapter.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )

    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )

    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt="a dragon flying in dark sky",
        num_frames=16,
        num_inference_steps=5
    )

    frames = output.frames[0]

    export_to_video(frames, "output.mp4", fps=8)

    del pipe
    del adapter
    torch.cuda.empty_cache()

    print("✅ MP4 saved")

# -----------------------------
# Run
# -----------------------------
generate_image()
# generate_video()