import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import os


# 모델 로드
def load_model():
    try:
        current_dir = os.path.dirname(__file__)
        print(current_dir)
        model = tf.keras.models.load_model('./cat_dog_classifier.keras')
        st.success('모델을 load했습니다.')
        return model
    
    except Exception as e:
        print(e)
        st.error('모델을 로드할 수 없습니다. 경로를 확인해주세요!')


model = load_model()

# 사용자 업로드한 이미지 전처리
def preprocess_image(image):
    try:
        image = image.resize((150, 150))
        image = np.array(image) / 255.0
        if image.shape[-1] != 3:
            raise ValueError('이미지는 RGB 형식의 컬러이미지만 처리가 가능합니다.')

        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"이미지 전처리 중 문제가 발생했습니다. {e}")



# UI 
st.title('Cat/Dog 분류기')
st.write('이미지를 업로드하면 개 또는 고양이를 판별합니다.')


upload_file = st.file_uploader('이미지를 업로드하세요.', type=['jpg', 'png', 'jpeg'], key='image_upload_1')

if upload_file:
    try:
        # 이미지 로드 -> 이미지 파일을 이미지 객체로 변환
        image = Image.open(upload_file)
        st.image(image, caption='업로드된 이미지', use_column_width=True)

        preprocessed_image = preprocess_image(image)

        if preprocessed_image is not None:
            prediction = model.predict(preprocessed_image)
            print(prediction)
        
        if prediction[0][0] > 0.5 :
            st.success('이 이미지는 개로 분류되었습니다.')
        else:
            st.success('이 이미지는 고양이로 분류되었습니다.')

    except UnidentifiedImageError: # 이미지 형식이 아닌경우
        st.error('이미지를 로드할 수 없습니다.')
    except Exception as e:
        st.error(f'예측 처리 중 오류 발생 : {e}')


