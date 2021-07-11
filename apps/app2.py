import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import awesome_streamlit as ast
from PIL import *
import cv2

import urllib.request
url = 'https://github.com/ziadimahdi/Gender-Classification-Object-Detection/blob/main/apps/yolov3.weights'
yolov3 = url.split('/')[-1]

urllib.request.urlretrieve(url, yolov3)


def detect_objects(our_image):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    col1, col2 = st.beta_columns(2)

    col1.subheader("Original Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(our_image)
    col1.pyplot(use_column_width=True)

    # YOLO ALGORITHM
    net = cv2.dnn.readNet("yolov3", "apps/yolov3.cfg")
    if not yolov3.weights.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive('https://drive.google.com/file/d/1kQvKgAcfs0TR9vpJAwwyHKqm4CzSaoF_/view?usp=sharing', yolov3.weights)



    classes = []
    with open("apps/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0,255,size=(len(classes), 3))


    # LOAD THE IMAGE
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    height,width,channels = img.shape


    # DETECTING OBJECTS (CONVERTING INTO BLOB)
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop = False)   #(image, scalefactor, size, mean(mean subtraction from each layer), swapRB(Blue to red), crop)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes =[]

    # SHOWING INFORMATION CONTAINED IN 'outs' VARIABLE ON THE SCREEN
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # OBJECT DETECTED
                #Get the coordinates of object: center,width,height
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)  #width is the original width of image
                h = int(detection[3] * height) #height is the original height of the image

                # RECTANGLE COORDINATES
                x = int(center_x - w /2)   #Top-Left x
                y = int(center_y - h/2)   #Top-left y

                #To organize the objects in array so that we can extract them later
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    score_threshold = st.sidebar.slider("Confidence Threshold", 0.00,1.00,0.5,0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.00, 1.00, 0.4, 0.01)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold,nms_threshold)
    print(indexes)

    font = cv2.FONT_HERSHEY_SIMPLEX
    items = []
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            #To get the name of object
            label = str.upper((classes[class_ids[i]]))
            color = colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            items.append(label)


    st.text("")
    col2.subheader("Object-Detected Image")
    st.text("")
    plt.figure(figsize = (15,15))
    plt.imshow(img)
    col2.pyplot(use_column_width=True)

    if len(indexes)>1:
        st.success("Found {} Objects - {}".format(len(indexes),[item for item in set(items)]))
    else:
        st.success("Found {} Object - {}".format(len(indexes),[item for item in set(items)]))

def object_main():
    """OBJECT DETECTION APP"""

    st.title("Object Detection")

    choice = st.radio("", ("Show Demo", "Browse an Image"))
    st.write()

    if choice == "Browse an Image":
        st.set_option('deprecation.showfileUploaderEncoding', False)
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            detect_objects(our_image)

    elif choice == "Show Demo":
        our_image = Image.open("apps/images/person.jpg")
        detect_objects(our_image)




def write():
    """Method used to write the page in the app.py file"""
    ast.shared.components.title_awesome("Gender Classification & Object Detection")
    with st.spinner("Loading  ..."):
        vision = object_main()
        st.markdown(vision)

