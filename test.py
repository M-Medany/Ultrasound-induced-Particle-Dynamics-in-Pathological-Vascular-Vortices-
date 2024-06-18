import cv2
import pandas as pd
import xlsxwriter
import os
from datetime import datetime

# Define the path to the directory containing images
image_folder = 'D:/Aneurysm_Exp/Mahmoud 06.06.2022 HIGH SPEED CAMERA/05_vid_2022-06-08_11-25-41_edited'

# Initialize a DataFrame to store the results
results = pd.DataFrame(columns=['Timestamp', 'Frame', 'Area'])

# Function to process images
def process_images(folder):
    for idx, filename in enumerate(sorted(os.listdir(folder))):
        if filename.endswith(".png"):  # assuming the images are in JPG format
            file_path = os.path.join(folder, filename)
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # filter out very small contours
                    area = cv2.contourArea(contour)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    results.loc[len(results)] = [timestamp, idx, area]

# Run the image processing function
process_images(image_folder)

# Write results to Excel
with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
    results.to_excel(writer, index=False)

print('Processing complete, results are saved in "output.xlsx"')
