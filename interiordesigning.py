import gradio as gr
from google import genai
from google.genai import types
import PIL.Image
from io import BytesIO

# --- API Configuration ---
# Your key remains the same
GEMINI_API_KEY = "AIzaSyDt0xwBJNhUmBP2fEXBKFuf0RqGfhykMfc"

# Initialize client using the modern GenAI SDK
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_gruhabuddy_design(room_type, design_style, custom_idea, budget):
    if not budget:
        budget = "Not specified"

    # 1. Prepare Prompts
    text_prompt = f"Professional interior design plan for a {room_type} in {design_style} style. Idea: {custom_idea}. Budget: ₹{budget}."
    image_prompt = f"4k realistic interior design, {design_style} {room_type}, {custom_idea}, architectural photography."

    try:
        # A. Generate Text (Using 2.0 Flash - the current stable version)
        text_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=text_prompt,
        )
        design_plan = text_response.text

        # B. Generate Image (Using Imagen 3)
        # Note: If this fails with 404, your API key may not have Imagen access yet.
        image_response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=image_prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio="1:1"
            )
        )

        # FIX: Correctly extract image from the response object
        # The SDK returns a 'GeneratedImage' object containing an 'Image' object with 'image_bytes'
        first_image = image_response.generated_images[0]
        img_bytes = first_image.image.image_bytes
        generated_image = PIL.Image.open(BytesIO(img_bytes))

        return design_plan, generated_image

    except Exception as e:
        error_msg = str(e)
        # Check for specific quota or access errors
        if "429" in error_msg:
            return "❌ Quota Exceeded. Please wait 1 minute.", None
        if "404" in error_msg:
            return f"❌ Model Not Found. Your key might not have access to Imagen 3 yet. Error: {error_msg}", None
        return f"Error: {error_msg}", None

# --- UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 GruhaBuddy AI")

    with gr.Row():
        with gr.Column():
            room = gr.Dropdown(["Bedroom", "Living Room", "Office"], label="Room", value="Living Room")
            style = gr.Dropdown(["Modern", "Minimalist", "Luxury"], label="Style", value="Modern")
            idea = gr.Textbox(label="Custom Idea")
            price = gr.Number(label="Budget (₹)", value=50000)
            submit = gr.Button("✨ Generate", variant="primary")

        with gr.Column():
            out_img = gr.Image(label="AI View")
            out_text = gr.Markdown(label="Plan Details")

    submit.click(
        fn=generate_gruhabuddy_design,
        inputs=[room, style, idea, price],
        outputs=[out_text, out_img]
    )

# --- Launch specifically for Colab ---
if __name__ == "__main__":
    # Removed deprecated theme/css from constructor and added to launch for compatibility
    demo.launch(share=True, debug=True)
