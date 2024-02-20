from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
import os

from comfy.model_patcher import ModelPatcher
from comfy.cli_args import args
import comfy.utils
import folder_paths


class ModelNameToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION: "extract_model_name"
    CATEGORY = "MyUtilsNodes"

    def extract_model_name(self, model):
        print("model_type+++++++++-----------\n", getattr(model))
        return "THIS"


class SaveImageWithLoraWeight:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4
        self.strength_model = None
        self.strength_clip = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"model": ("MODEL",),
                     "images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images_with_lora"

    OUTPUT_NODE = True
    CATEGORY = "MyUtilsNodes"

    def save_images_with_lora(self, model: ModelPatcher, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        self.strength_model = model.patches["diffusion_model.output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.weight"][0][0]
        print("STRENGTH", self.strength_model)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if self.strength_model != None:
                    metadata.add_itxt("lora_weight", str(self.strength_model))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{self.strength_model}_{filename_with_batch_num}_{counter:05}.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
        return  { "ui": { "images": results } }    


# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ModelNameToString": ModelNameToString,
    "SaveImageWithLoraWeight": SaveImageWithLoraWeight
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StringConversions": "ModelNameToString",
    "SaveImageWithLoraWeight": "SaveImageWithLoraWeight"
}