from tensorflow import keras
import numpy as np
import streamlit as st
import cv2
IMAGE_SIZE = 150


def load_image(image):
    f_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(f_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(img, caption="Uploaded MRI Image", width=400)

    # Preprocessing the image
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
    return img


def display_image_details(image):

    st.markdown(" The details of the uploaded file are : ")
    details = {
        "File Name": image.name,
        "File Type": image.type,
        "File Size": str(image.size/100) + "KB",
    }
    st.write(details)


def describe_model(desc):
    if desc:
        st.code(
            '''
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2))) # Pooling

model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2))) # Pooling


model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])
            '''
        )
        st.image(
            'https://raw.githubusercontent.com/hemanth-kotagiri/brain-tumor-detection/master/nn.png', caption="Model Architecture")


def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
    st.title("Brain Tumor Detection")
    st.caption('A predictive Deep Learning Model trained on MRI images of Brain\
               for Tumor Detection.  This application aims to provide prior\
               diagnosis for the existence of a tumor in a given brain MRI\
               image.')
    image = st.file_uploader(
        "Please upload your Brain MRI Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    show_description = st.checkbox("Show model description")
    describe_model(show_description)
    if image is not None:
        display_image_details(image)
        given_image = load_image(image)

        # Loading the model
        model = keras.models.load_model("model.h5")
        prediction = round(model.predict(given_image)[0][0] * 100, 2)
        st.subheader("Model Prediction")
        if prediction > 75:
            st.error(
                f"The Model Predicts that the image has tumor. Chance: {prediction} %")
        else:
            st.success(
                f"The model predicts there is no tumor in the given image. Chance: {prediction} %")


if __name__ == "__main__":
    main()
