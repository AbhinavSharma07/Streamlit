''
import streamlit as st
from PIL import Image, ImageDraw
import random

# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Function to generate a random geometric shape
def draw_shape(draw, shape_type, width, height):
    if shape_type == "Circle":
        x1, y1 = random.randint(0, width // 2), random.randint(0, height // 2)
        x2, y2 = random.randint(x1, width), random.randint(y1, height)
        draw.ellipse([x1, y1, x2, y2], fill=random_color(), outline=random_color())

    elif shape_type == "Rectangle":
        x1, y1 = random.randint(0, width // 2), random.randint(0, height // 2)
        x2, y2 = random.randint(x1, width), random.randint(y1, height)
        draw.rectangle([x1, y1, x2, y2], fill=random_color(), outline=random_color())

    elif shape_type == "Triangle":
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        x3, y3 = random.randint(0, width), random.randint(0, height)
        draw.polygon([x1, y1, x2, y2, x3, y3], fill=random_color(), outline=random_color())

# Streamlit App UI
st.title("Random Geometric Shape Generator")

# User input for canvas size
st.sidebar.header("Canvas Options")
canvas_width = st.sidebar.slider("Canvas Width", 200, 800, 400)
canvas_height = st.sidebar.slider("Canvas Height", 200, 800, 400)

# User input for number of shapes and shape type
st.sidebar.header("Shape Options")
num_shapes = st.sidebar.slider("Number of Shapes", 1, 20, 5)
shape_type = st.sidebar.selectbox("Select Shape Type", ["Circle", "Rectangle", "Triangle"])

# Button to generate shapes
if st.sidebar.button("Generate Shapes"):
    # Create a blank image
    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw random shapes on the image
    for _ in range(num_shapes):
        draw_shape(draw, shape_type, canvas_width, canvas_height)

    # Display the image in Streamlit
    st.image(image, caption="Generated Image", use_column_width=True)

# About section
st.sidebar.header("About")
st.sidebar.info(
    """
    This simple app generates random geometric shapes based on user-defined parameters.
    Choose the canvas size, shape type, and number of shapes, and watch the magic happen!
    """
)

# Extra feature: Allow users to download the generated image
if 'image' in locals():
    st.sidebar.header("Download Image")
    img_format = st.sidebar.selectbox("Image Format", ["PNG", "JPEG"])
    img_name = st.sidebar.text_input("Enter Image Name", "generated_image")

    if st.sidebar.button("Download Image"):
        img_path = f"{img_name}.{img_format.lower()}"
        image.save(img_path, format=img_format)
        with open(img_path, "rb") as file:
            btn = st.sidebar.download_button(
                label="Download Image",
                data=file,
                file_name=img_path,
                mime=f"image/{img_format.lower()}"
            )

# Display random shapes generated so far
st.subheader("Generated Shapes")
if 'image' in locals():
    st.image(image, caption="Generated Shapes", use_column_width=True)
else:
    st.write("Click 'Generate Shapes' in the sidebar to create random shapes!")
