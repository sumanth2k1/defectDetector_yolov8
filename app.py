import streamlit as st
from PIL import Image
from ultralytics import YOLO
from io import BytesIO  # Import BytesIO

# Load the YOLO model
model = YOLO("best1.pt")

def predict_and_display(image_file):
  """
  Performs prediction on an uploaded image and displays the result.
  """
  # Read the image file
  image = Image.open(image_file)

  # Perform prediction
  results = model.predict(image)
  result = results[0]

  # Convert image from numpy array to PIL Image and resize
  pil_image = Image.fromarray(result.plot()[:,:,::-1])
  desired_width = 600
  desired_height = 600
  pil_image = pil_image.resize((desired_width, desired_height), Image.LANCZOS)

  # Convert PIL Image to bytes
  image_bytes = BytesIO()
  pil_image.save(image_bytes, format='PNG')
  image_bytes.seek(0)

  # Display the image with prediction results
  st.image(image_bytes, caption="Predicted image with bounding boxes")

# Title and description
st.title("YOLO Object Detection App")
st.write("Upload an image and see the object detection results using YOLO model.")

# Upload image section
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

# Run prediction when a file is uploaded
if uploaded_file is not None:
  predict_and_display(uploaded_file)

