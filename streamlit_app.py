import streamlit as st
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import mediapipe as mp
import base64

hide_github_link_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visiblity: hidden;}
    header {visibility: hidden;}
        .viewerBadge_container__1QSob {
            display: none !important;
        }
    </style>
"""
st.markdown(hide_github_link_style, unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Annotate the function with @st.cache(allow_output_mutation=True)
@st.cache(allow_output_mutation=True)
def create_history_dataframe():
    return pd.DataFrame(columns=["ID", "Name", "Timestamp", "Location"])

# Create a set to store unique user IDs
unique_user_ids = set()

# Create folder for saving uploaded images
UPLOADS_FOLDER = "uploads"
st.set_option('deprecation.showfileUploaderEncoding', False)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Function to detect faces using Mediapipe and update history
def detect_faces_and_update_history(image, user_id, user_name, user_location, history_df):
    # Check if the user ID is already logged in
    if user_id in unique_user_ids:
        st.warning(f"User with ID {user_id} already logged in.")
        return

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(image_rgb)

    # Update history DataFrame
    timestamp = datetime.now()
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            location = user_location
            history_df.loc[len(history_df)] = [user_id, user_name, timestamp, location]

            # Save face images with ID and timestamp
            face_img = image[y:y+h, x:x+w]
            face_filename = f"uploads/face_{user_id}_{timestamp.strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(face_filename, face_img)

        # Display the uploaded image
        st.image(image, channels="BGR", use_column_width=True, caption="Uploaded Image")

        # Add the user ID to the set of unique user IDs
        unique_user_ids.add(user_id)
    else:
        location = "No face detected"
        st.warning("Image rejected: No face detected.")

# Function to get user's IP address
def get_user_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        if response.status_code == 200:
            data = response.json()
            # Extract latitude and longitude from the response
            lat, lon = map(float, data["loc"].split(","))
            return lat, lon
        else:
            return None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None

# Function to display a warning pop-up if no face is detected
def display_no_face_warning():
    st.warning("No face detected in the image or camera feed.")

# Streamlit app
def main():
    st.title("Face Detection and History")

    # Sidebar for user input
    user_id = st.sidebar.text_input("Enter User ID:")
    user_name = st.sidebar.text_input("Enter User Name:")
    section = st.sidebar.radio("Select Section", ["Upload Image", "Detect Faces", "History"])

    # Use the cached function with allow_output_mutation
    history_df = create_history_dataframe()

    # Hide sensitive information for deployment
    if st.sidebar.checkbox("Hide Sensitive Information"):
        st.markdown("Sensitive information is hidden.")
    else:
        st.warning("Please be cautious about displaying sensitive information.")

    if section == "Upload Image":
        # Check if the user ID is already logged in
        if user_id in unique_user_ids:
            st.warning(f"User with ID {user_id} already logged in.")
        else:
            uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), -1)
                user_location = get_user_location()
                detect_faces_and_update_history(image, user_id, user_name, user_location, history_df)

                # Option to save data to history
                if st.button("Save to History"):
                    timestamp = datetime.now()
                    location = user_location
                    history_df.loc[len(history_df)] = [user_id, user_name, timestamp, location]
                    st.success("Data saved to history.")

    elif section == "Detect Faces":
        # Open camera and start detection
        cap = cv2.VideoCapture(0)
        checkbox_idx = 0  # Initialize index for unique checkbox keys
        while st.sidebar.checkbox("Detect Faces", key=f"checkbox_{checkbox_idx}"):
            ret, frame = cap.read()
            user_location = get_user_location()
            detect_faces_and_update_history(frame, user_id, user_name, user_location, history_df)
            checkbox_idx += 1  # Increment index for the next checkbox

        cap.release()

    elif section == "History":
        # Display history DataFrame
        st.title("History")
        st.dataframe(history_df)

        # Download history as CSV
        if st.button("Download History as CSV"):
            csv_file = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="face_detection_history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

# Streamlit app entry point
if __name__ == "__main__":
    main()
