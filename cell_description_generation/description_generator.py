import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO
import random

class CellDescriptionGenerator:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

        self.system_prompt = """You are an expert pathologist analyzing cropped microscopy images stained with Hematoxylin and Eosin (width and height of the image correspond to 28 micrometers (224 x 224 pixels)). You are creating a dataset of cell images and their text descriptions. Each image has been specifically cropped to position a single target cell at the center of the image, with possible mirror padding at the edges. Your task is to create precise descriptions for each of the provided images focusing exclusively on the centered cell. Your descriptions should be:
- Single paragraph format
- Concise and specific, avoiding unnecessary words, adverbs and bullet points
- Focused only on the centered target cell and its visible features
For each one of the image samples you will be provided the cell type and the tissue type, but the resulting description should not provide this information."""

        self.user_prompts = [
     """Describe the center cell in this H&E stained image based solely on what is visible, without assumptions based on the provided cell type or tissue type. Cell type: {cell_type}. Tissue type: {tissue_type}.
 The description should be based on the cell's components (nucleus, nucleolus, nuclear membrane, cytoplasm, cell membrane), noting their shape, size, staining intensity, granularity, and any notable structures. Also focus on visible features that characterize the center cell.""",

     """Provide a description of the center cell in this H&E stained image, focusing only on the observable features without inferring characteristics from the provided cell type. Cell type: {cell_type}. Tissue type: {tissue_type}.
 Focus on visible features that characterize the cell's components (cell membrane, cytoplasm, nucleus, nuclear membrane, nucleolus), describing their shape, size, staining intensity, granularity, and any notable structures. Don't start with "The center cell" or ending with an "Overall" statement.""",

     """Generate a description for the center cell in this H&E stained image based strictly on the visible morphological features, not on expectations from the cell type. Cell type: {cell_type}. Tissue type: {tissue_type}.
 The description should contain the morphological features of the cell's components (cytoplasm, nucleus, nucleolus, cell membrane, nuclear membrane), including their size, shape, staining intensity, granularity, and any notable structures.""",

     """Describe the center cell in this H&E stained image based exclusively on its visible appearance, regardless of the provided cell type. Cell type: {cell_type}. Tissue type: {tissue_type}.
 Focus on the observable characteristics of the cell's components (nuclear membrane, nucleus, cell membrane, cytoplasm, nucleolus), noting any notable structures that define the center cell, their shape, size, staining patterns or granularity. Don't start with "The center cell" or ending with an "Overall" statement."""
 ] 

    def _process_image(self, image_path):
        """
        Resize and encode image in base64 format.
        The resize is done to improve the quality of the analysis.
        """
        with Image.open(image_path) as img:
            new_size = (img.width * 2, img.height * 2)
            resized = img.resize(new_size, Image.BICUBIC)

            buffer = BytesIO()
            resized.save(buffer, format=img.format)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def get_cell_description(self, image_path, cell_type, tissue_type):
        """
        Get description for a cell image using the OpenAI vision API.
        """
        base64_image = self._process_image(image_path)

        user_prompt = random.choice(self.user_prompts).format(
            cell_type=cell_type,
            tissue_type=tissue_type
        )

        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=128,
            temperature=0.4
        )

        return response.choices[0].message.content