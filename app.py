import gradio as gr
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

processor = TrOCRProcessor.from_pretrained(".")
model = VisionEncoderDecoderModel.from_pretrained(".")
model.eval()

def split_lines_from_image(pil_image):
    img = np.array(pil_image.convert("L"))

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])

    line_images = []
    for x, y, w, h in bounding_boxes:
        line_img = img[y:y+h, x:x+w]
        pil_line = Image.fromarray(line_img).convert("RGB")
        line_images.append(pil_line)

    return line_images

def extract_text(image):
    line_images = split_lines_from_image(image)
    result = []
    for line in line_images:
        pixel_values = processor(images=line, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=100)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result.append(text)
    return "\n".join(result)

demo = gr.Interface(
    fn=extract_text,
    inputs=gr.Image(type="pil", label="Upload Multi-line Image"),
    outputs=gr.Textbox(label="Extracted Text"),
    title="TrOCR Line-by-Line OCR (Offline)",
    description="Upload a multi-line printed text image. TrOCR will split lines and extract text."
)

if __name__ == "__main__":
    demo.launch()
