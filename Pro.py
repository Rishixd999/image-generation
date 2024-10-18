import streamlit as st
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Initialize GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and cache the Stable Diffusion model
@st.cache_resource
def load_sd_model():
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)  # Use the detected device (GPU/CPU)
    return pipe

# Load and cache the BLIP image captioning model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

# Load and cache the Zephyr model for story generation
@st.cache_resource
def load_zephyr_model():
    model_name = "HuggingFaceH4/zephyr-7b-alpha"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
    return tokenizer, model

# Function to generate images from prompts using Stable Diffusion
def generate_images(prompts, pipe):
    images = []
    for prompt in prompts:
        image = pipe(prompt).images[0]  # Generate an image for each prompt
        images.append(image)
    return images

# Function to generate captions for images using BLIP
def generate_image_caption(image, blip_processor, blip_model):
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    caption_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# Function to generate a story from captions using the Zephyr model
def generate_cumulative_story(captions_list, zephyr_tokenizer, zephyr_model):
    combined_prompt = " ".join([f"{i+1}. {caption}" for i, caption in enumerate(captions_list)])
    prompt = f"Based on these image captions: {combined_prompt}, generate a creative and detailed story that ties them all together."
    inputs = zephyr_tokenizer(prompt, return_tensors="pt").to(device)
    output = zephyr_model.generate(**inputs, max_new_tokens=1000, eos_token_id=zephyr_tokenizer.eos_token_id)
    story = zephyr_tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Streamlit app structure
st.title("Image Generation and Storytelling App")

# Load models
pipe = load_sd_model()
blip_processor, blip_model = load_blip_model()
zephyr_tokenizer, zephyr_model = load_zephyr_model()

# Input prompts for image generation
prompts = []
for i in range(4):
    prompt = st.text_input(f"Enter prompt {i + 1} for image generation:")
    prompts.append(prompt)

# Generate images, captions, and story when the button is clicked
if st.button("Generate Images and Story"):
    if all(prompts):  # Ensure all prompts are provided
        with st.spinner('Generating images...'):
            images = generate_images(prompts, pipe)

        st.subheader("Generated Images")
        captions_list = []
        
        # Display images and generate captions
        for i, img in enumerate(images):
            st.image(img, caption=f"Image {i + 1}: {prompts[i]}", use_column_width=True)
            with st.spinner(f'Generating caption for Image {i + 1}...'):
                caption = generate_image_caption(img, blip_processor, blip_model)
                captions_list.append(caption)
                st.write(f"Caption for Image {i + 1}: {caption}")
        
        # Generate the story from captions
        with st.spinner('Generating story from captions...'):
            story = generate_cumulative_story(captions_list, zephyr_tokenizer, zephyr_model)
        
        st.subheader("Generated Story")
        st.write(story)
    
    else:
        st.warning("Please enter prompts for all four inputs.")
