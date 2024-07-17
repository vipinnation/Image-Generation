import streamlit as st
import pandas as pd

import torch 
from diffusers import StableDiffusion3Pipeline
import random 

def genImage(text):
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()

    image = pipe(
    prompt=text,
    negative_prompt="",
    num_inference_steps=18,
    height=512,
    width=512,
    guidance_scale=7.0,
    
    ).images[0]
 
 
    random_number = random.randint(1000, 9999)
    filename = f"sd3_hello_world_{random_number}.png"
    image.save(filename)

    print("image saved successfully as",filename)
    st.image(filename, caption=text)

st.header('Generate an image', divider='rainbow')
st.caption('just provide a textual description')

title = st.text_input("Image Prompt", "", placeholder="Write your prompt here")
st.write("Your Prompt", title)

if st.button("Submit"):
    genImage(title)



