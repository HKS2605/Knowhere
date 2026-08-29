import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)

def simple_vegetation_mask(rgb_image):
    r, g, b = rgb_image[:,:,0].astype(int), rgb_image[:,:,1].astype(int), rgb_image[:,:,2].astype(int)
    # Vegetation: green is clearly the dominant channel
    mask = (g > r + 10) & (g > b + 10)
    return mask.astype(np.uint8) * 255

def simple_water_mask(rgb_image):
    r, g, b = rgb_image[:,:,0].astype(int), rgb_image[:,:,1].astype(int), rgb_image[:,:,2].astype(int)
    # Water: blue is clearly the dominant channel
    mask = (b > r + 10) & (b > g + 10)
    return mask.astype(np.uint8) * 255

def mask_to_bbox_opencv(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h}

def draw_overlay(image_path, bbox, output_path, label="detected"):
    image = cv2.imread(image_path)
    if bbox:
        cv2.rectangle(image, (bbox["x_min"], bbox["y_min"]), (bbox["x_max"], bbox["y_max"]), (0, 140, 255), 3)
        cv2.putText(image, label, (bbox["x_min"], max(bbox["y_min"] - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.imwrite(output_path, image)
    return output_path

def detect_keywords(query):
    q = query.lower()
    if "water" in q or "river" in q or "lake" in q:
        return "water"
    elif "vegetation" in q or "forest" in q or "green" in q:
        return "vegetation"
    return None

def caption_and_ground(image_path, query, output_dir="sample_images"):
    caption = generate_caption(image_path)
    image_array = np.array(Image.open(image_path).convert("RGB"))
    feature = detect_keywords(query)

    bbox = None
    overlay_path = None
    if feature == "vegetation":
        bbox = mask_to_bbox_opencv(simple_vegetation_mask(image_array))
    elif feature == "water":
        bbox = mask_to_bbox_opencv(simple_water_mask(image_array))

    if bbox:
      
       unique_id = f"{feature}_{int(time.time()*1000) % 100000}"
       overlay_path = draw_overlay(image_path, bbox, f"{output_dir}/grounded_{unique_id}.png", feature)

    return {
        "caption": caption,
        "grounded_feature": feature,
        "bbox": bbox,
        "overlay_image_path": overlay_path
    }

if __name__ == "__main__":
    print("\n--- Test 1: Water query ---")
    result1 = caption_and_ground("sample_images/sample_0.png", "Where is the water body?")
    print(result1)

    print("\n--- Test 2: Vegetation query ---")
    result2 = caption_and_ground("sample_images/sample_0.png", "Highlight the vegetation")
    print(result2)

    print("\n--- Test 3: No keyword (just describe) ---")
    result3 = caption_and_ground("sample_images/sample_0.png", "What is in this image?")
    print(result3)

    

result = caption_and_ground("real_images/converted.png", "Where is the water body?")
print(result)