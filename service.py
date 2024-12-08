from __future__ import annotations
import bentoml
from PIL.Image import Image

with bentoml.importing():
    import torch

TASK = "cord-v2"
TASK_PROMPT = f"<s_{TASK}>"
PRETRAINED_PATH = "naver-clova-ix/donut-base-finetuned-cord-v2"

@bentoml.service
class ReceiptExtract:
    def __init__(self) -> None:
        from transformers import DonutProcessor, VisionEncoderDecoderModel 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DonutProcessor.from_pretrained(PRETRAINED_PATH)
        self.model = VisionEncoderDecoderModel.from_pretrained(PRETRAINED_PATH).to(self.device)
    
    @bentoml.api
    async def extract_image(self, image: Image):
        return self.processor(image, return_tensors="pt").pixel_values 

    @bentoml.api
    async def extract(self, image: Image):
        """
        Generate text from an image using the trained model.
        """
        import re

        # Load and preprocess the image
        pixel_values = await self.extract_image(image)
        pixel_values = pixel_values.to(self.device)

        # Generate output using model
        self.model.eval()
        with torch.no_grad():
            decoder_input_ids = self.processor.tokenizer(TASK_PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(self.device)
            generated_outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings, 
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                early_stopping=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )

        # Decode generated output
        decoded_text = self.processor.batch_decode(generated_outputs.sequences)[0]
        decoded_text = decoded_text.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        decoded_text = re.sub(r"<.*?>", "", decoded_text, count=1).strip()  # remove first task start token
        decoded_text = self.processor.token2json(decoded_text)
        return decoded_text
