import streamlit as st
from PIL import Image
from io import BytesIO

from useful import load_model, run_inference_single, process_image

MODEL_PATH = './model/best.pt'
model = load_model(MODEL_PATH)

st.set_page_config(
    page_title='Helmet Safety Detection App',
    initial_sidebar_state='expanded',
    layout='wide'
)

st.title('Helmet Safety Detection App')

uploaded_file = st.file_uploader(
    'Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        img = Image.open(uploaded_file)
        st.image(uploaded_file,
                 caption='Uploaded Image',
                 use_column_width=True)

    with col2:
        with st.spinner('...'):
            img_out = process_image(run_inference_single(model, img))
        st.image(img_out,
                 caption='Output Image',
                 use_column_width=True)
    # Assuming the rest of your code is correct and img_out is a numpy array
    # Convert img_out to a PIL Image
    img_out_pil = Image.fromarray(img_out.astype('uint8'))

    # Convert the PIL Image to bytes for downloading
    img_bytes = BytesIO()
    img_out_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Create a download button for the output image
    st.download_button(
        label="Download Output Image",
        data=img_bytes,
        file_name="output_image.png",
        mime="image/png"
)
        
        
