import gradio as gr
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
from transformers import pipeline

# Modelle laden
yolo_model = YOLO("./best.pt")
dino_model = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-tiny", tokenizer_kwargs={"padding": True, "truncation": True})

# YOLOv8-Erkennung
def detect_with_yolo(image: Image.Image):
    results = yolo_model(np.array(image))[0]
    return Image.fromarray(results.plot())

# Grounding DINO-Erkennung
def detect_with_grounding_dino(image: Image.Image, prompt=["license plate.", "number plate.", "car plate.", "vehicle registration plate."]):
    results = dino_model(image, candidate_labels=prompt)
    image_np = np.array(image).copy()

    if not results:
        return Image.fromarray(image_np)

    results = [result for result in results if result["score"] > 0.4]

    for result in results:
        box = result["box"]
        score = result["score"]
        label = "license plate"

        x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
        image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        image_np = cv2.putText(image_np, f"{label} ({score:.2f})", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return Image.fromarray(image_np)

# Verarbeitung der Bilder
def process_image(image):
    yolo_out = detect_with_yolo(image)
    dino_out = detect_with_grounding_dino(image)
    return yolo_out, dino_out

# Beispielbilder definieren
example_images = [
    ["example_images/image1.jpg"],
    ["example_images/image2.jpg"],
    ["example_images/image3.jpg"],
    ["example_images/image4.jpg"],
    ["example_images/image5.jpg"],
    ["example_images/image6.jpg"],
    ["example_images/image7.jpg"]
]

# Gradio-Interface
app = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="YOLOv8 Detection"),
        gr.Image(label="Grounding DINO (Zero-Shot) Detection")
    ],
    examples=example_images,
    cache_examples=False,
    title="Kennzeichenerkennung",
    description="Lade ein Bild hoch oder wähle ein Beispielbild und vergleiche die Ergebnisse."
)

if __name__ == "__main__":
    app.launch()