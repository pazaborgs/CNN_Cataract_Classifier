import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px


@st.cache_resource
def load_model():
    url = 'https://drive.google.com/uc?id=15pPkPuEk4YJwEkYBAwE9MEnQJ8JmJF84'
    gdown.download(url, 'catarata_model.keras')
    interpreter = tf.lite.Interpreter(model_path='catarata_model.keras')
    interpreter.allocate_tensors()
    return interpreter

def load_image():
    uploaded_file = st.file_uploader('Arraste e solte uma imagem aqui ou clique para selecionar uma', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem carregada com sucesso.')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def prev(interpreter, image):
 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'],image) 

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']
    df = pd.DataFrame()
    
    df['classes'] = classes                             # classes
    df['probabilidades (%)'] = 100*output_data[0]       # output prob

    fig = px.bar(df, y = 'classes', x = 'probabilidades (%)', orientation = 'h', text='probabilidades (%)', title='Probabilidade de maturidade de catarata')
    st.plotly.chart(fig)





def main():

    st.set_page_config(
        page_title = 'Cataract Classifier',
        page_icon = '&#128065;',
    )

    st.write('# Classificador: Catarata &#128065;')

    interpreter = load_model()
    image = load_image()

    if image is not None:
        prev(interpreter, image)


    if __name__ == '__main__':
        main()

