import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import io
import zipfile
from datetime import datetime
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil

# ----------------------------------------
# Functions for Noise Addition
# ----------------------------------------

def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.

    :param image: NumPy array of the image in RGB format.
    :param mean: Mean of the Gaussian noise.
    :param std: Standard deviation of the Gaussian noise.
    :return: Noisy image as a NumPy array.
    """
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# ----------------------------------------
# Function to Handle Annotations
# ----------------------------------------

def update_coco_annotations(annotation_data, original_image_id, new_image_id, new_file_name):
    """
    Updates COCO annotations for the processed image.

    :param annotation_data: The original COCO annotation JSON data as a dictionary.
    :param original_image_id: The original image ID in the annotations.
    :param new_image_id: The new image ID for the processed image.
    :param new_file_name: The filename of the processed image.
    :return: Updated annotation data with the new image and its annotations.
    """
    # Duplicate the image entry
    original_image = next((img for img in annotation_data['images'] if img['id'] == original_image_id), None)
    if original_image is None:
        return annotation_data  # Original image not found

    # Create a new image entry
    new_image = original_image.copy()
    new_image['id'] = new_image_id
    new_image['file_name'] = new_file_name
    annotation_data['images'].append(new_image)

    # Duplicate all annotations for this image
    for ann in annotation_data['annotations']:
        if ann['image_id'] == original_image_id:
            new_ann = ann.copy()
            new_ann['id'] = max([a['id'] for a in annotation_data['annotations']]) + 1
            new_ann['image_id'] = new_image_id
            annotation_data['annotations'].append(new_ann)
    
    return annotation_data

# ----------------------------------------
# Function to Create ZIP
# ----------------------------------------

def create_zip(original_folder, processed_folder, merged_folder=None):
    """
    Creates a ZIP archive containing original and processed images and annotations.

    :param original_folder: Path to the original data folder.
    :param processed_folder: Path to the processed data folder.
    :param merged_folder: Path to the merged data folder (if any).
    :return: BytesIO object containing the ZIP archive.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add original images and annotations
        for root, _, files in os.walk(original_folder):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.join('original', os.path.relpath(filepath, original_folder))
                zipf.write(filepath, arcname)
        
        # Add processed images and annotations
        for root, _, files in os.walk(processed_folder):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.join('processed', os.path.relpath(filepath, processed_folder))
                zipf.write(filepath, arcname)
        
        # Add merged images and annotations if applicable
        if merged_folder and os.path.exists(merged_folder):
            for root, _, files in os.walk(merged_folder):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.join('merged', os.path.relpath(filepath, merged_folder))
                    zipf.write(filepath, arcname)
    zip_buffer.seek(0)
    return zip_buffer

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

def main():
    st.set_page_config(page_title="Dataset Augmentation Tool", layout="wide")
    st.title('Dataset Augmentation Tool')
    
    st.markdown("""
    This application allows you to upload a `dataset.zip` file containing images and their corresponding JSON annotation files (e.g., COCO format). You can apply Gaussian noise to all images in the dataset. The processed images and updated annotations will be saved in a new folder, and you can download the merged results as a ZIP file.
    """)
    
    st.header("Upload Dataset ZIP File")
    
    # File uploader for ZIP files
    uploaded_zip = st.file_uploader("Upload `dataset.zip`", type=["zip"])
    
    if uploaded_zip is not None:
        with TemporaryDirectory() as temp_dir:
            # Extract the ZIP file
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            st.success("Dataset ZIP file uploaded and extracted successfully.")
            
            # Display the contents of the ZIP
            st.subheader("Dataset Contents")
            # Define paths for original and processed data
            original_images_dir = os.path.join(temp_dir, "original", "images")
            original_annotations_dir = os.path.join(temp_dir, "original", "annotations")
            processed_images_dir = os.path.join(temp_dir, "processed", "images")
            processed_annotations_dir = os.path.join(temp_dir, "processed", "annotations")
            os.makedirs(original_images_dir, exist_ok=True)
            os.makedirs(original_annotations_dir, exist_ok=True)
            os.makedirs(processed_images_dir, exist_ok=True)
            os.makedirs(processed_annotations_dir, exist_ok=True)
            
            # Define image and annotation extensions
            image_extensions = ['.jpg', '.jpeg', '.png']
            annotation_extensions = ['.json']
            
            # Gather all image and annotation files
            all_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            image_files = [f for f in all_files if Path(f).suffix.lower() in image_extensions]
            annotation_files = [f for f in all_files if Path(f).suffix.lower() in annotation_extensions]
            
            st.write(f"**Total Images Found:** {len(image_files)}")
            st.write(f"**Total Annotation Files Found:** {len(annotation_files)}")
            
            # Optional: Display sample files
            if st.checkbox("Show Sample Image Files"):
                sample_images = image_files[:3]
                cols = st.columns(len(sample_images))
                for idx, img_path in enumerate(sample_images):
                    img = Image.open(img_path)
                    with cols[idx]:
                        st.image(img, caption=Path(img_path).name, use_column_width=True)
            
            if st.checkbox("Show Sample Annotation Files"):
                sample_annotations = annotation_files[:3]
                for ann_path in sample_annotations:
                    with open(ann_path, 'r') as f:
                        try:
                            ann_data = json.load(f)
                            st.json(ann_data)
                        except json.JSONDecodeError:
                            st.warning(f"Failed to parse {Path(ann_path).name} as JSON.")
            
            # Assuming COCO format: single JSON file with all images and annotations
            # Search for a single annotation file
            if len(annotation_files) == 1:
                annotation_file = annotation_files[0]
                with open(annotation_file, 'r') as f:
                    try:
                        annotation_data = json.load(f)
                        st.write(f"**Loaded Annotation File:** {Path(annotation_file).name}")
                    except json.JSONDecodeError:
                        st.error(f"Failed to parse {Path(annotation_file).name} as JSON.")
                        st.stop()
            else:
                st.error("Please ensure that the ZIP file contains exactly one JSON annotation file in COCO format.")
                st.stop()
            
            # Display summary of annotations
            st.write(f"**Number of Images in Annotations:** {len(annotation_data.get('images', []))}")
            st.write(f"**Number of Annotations:** {len(annotation_data.get('annotations', []))}")
            st.write(f"**Number of Categories:** {len(annotation_data.get('categories', []))}")
            
            # Noise and Color Transformation Options
            st.header("Gaussian Noise Options")
            noise_type = st.selectbox("Select Noise Type", ["None", "Gaussian"])
            params = {}
            if noise_type == 'Gaussian':
                mean = st.slider("Gaussian Noise Mean", min_value=0, max_value=100, value=0)
                std_dev = st.slider("Gaussian Noise Standard Deviation", min_value=0, max_value=100, value=25)
                params['mean'] = mean
                params['std_dev'] = std_dev
            else:
                st.info("No noise will be applied to the images.")
            
            # Optional: Color Transformation Options
            st.header("Color Transformation Options (Optional)")
            apply_color_transform = st.checkbox("Apply Color Transformation")
            if apply_color_transform:
                target_color = st.color_picker("Pick a target color", "#ff0000")
                hsv_lower = st.slider("Lower HSV Hue Value", 0, 179, 0, key='lower_h')
                hsv_upper = st.slider("Upper HSV Hue Value", 0, 179, 179, key='upper_h')
            else:
                target_color = None
                hsv_lower = None
                hsv_upper = None
            
            # Merge Processed Images with Original Images Option
            st.header("Merge Processed Images with Original Images")
            merge_option = st.checkbox("Merge Processed Images with Original Images")
            # If merged, processed images will be placed in a single 'merged' folder with distinct filenames
            
            # Apply Processing Button
            if st.button("Apply Gaussian Noise to All Images"):
                if noise_type == "None" and not apply_color_transform:
                    st.warning("No processing options selected. Please choose at least one processing option.")
                else:
                    with st.spinner('Processing images and annotations...'):
                        # Initialize new annotations data
                        processed_annotation_data = annotation_data.copy()
                        processed_annotation_data['images'] = annotation_data['images'].copy()
                        processed_annotation_data['annotations'] = annotation_data['annotations'].copy()
                        # Note: 'categories' remain unchanged

                        # Determine the maximum image and annotation IDs
                        existing_image_ids = [img['id'] for img in annotation_data.get('images', [])]
                        existing_annotation_ids = [ann['id'] for ann in annotation_data.get('annotations', [])]
                        max_image_id = max(existing_image_ids) if existing_image_ids else 0
                        max_annotation_id = max(existing_annotation_ids) if existing_annotation_ids else 0

                        # Process each image
                        for img in annotation_data.get('images', []):
                            original_image_id = img['id']
                            original_file_name = img['file_name']
                            original_image_path = None

                            # Locate the original image file
                            for img_file in image_files:
                                if Path(img_file).name == original_file_name:
                                    original_image_path = img_file
                                    break
                            
                            if original_image_path is None:
                                st.warning(f"Image file {original_file_name} not found in the ZIP. Skipping.")
                                continue
                            
                            # Load image
                            image = Image.open(original_image_path).convert('RGB')
                            image_array = np.array(image)
                            
                            # Apply Gaussian noise if selected
                            if noise_type == 'Gaussian':
                                noised_image_array = add_gaussian_noise(image_array, mean=params['mean'], std=params['std_dev'])
                            else:
                                noised_image_array = image_array.copy()
                            
                            # Apply color transformation if selected
                            if apply_color_transform:
                                noised_image_array = color_transform(
                                    noised_image_array,
                                    target_color,
                                    np.array([hsv_lower, 50, 50]),
                                    np.array([hsv_upper, 255, 255])
                                )
                            
                            # Save processed image
                            noised_image = Image.fromarray(noised_image_array)
                            name, ext = os.path.splitext(original_file_name)
                            noised_file_name = f"{name}_noised{ext}"
                            
                            if merge_option:
                                # Save processed image in the merged folder
                                merged_images_dir = os.path.join(temp_dir, "merged", "images")
                                os.makedirs(merged_images_dir, exist_ok=True)
                                processed_image_path = os.path.join(merged_images_dir, noised_file_name)
                            else:
                                # Save processed image in the processed images directory
                                processed_image_path = os.path.join(processed_images_dir, noised_file_name)
                            
                            noised_image.save(processed_image_path)
                            
                            # Update annotations
                            max_image_id += 1
                            new_image_entry = img.copy()
                            new_image_entry['id'] = max_image_id
                            new_image_entry['file_name'] = noised_file_name
                            processed_annotation_data['images'].append(new_image_entry)
                            
                            # Duplicate and update annotations for the new image
                            for ann in annotation_data.get('annotations', []):
                                if ann['image_id'] == original_image_id:
                                    max_annotation_id += 1
                                    new_ann = ann.copy()
                                    new_ann['id'] = max_annotation_id
                                    new_ann['image_id'] = max_image_id
                                    processed_annotation_data['annotations'].append(new_ann)
                        
                        # Save processed annotations
                        if merge_option:
                            # If merged, all annotations are in one JSON file
                            processed_annotations_path = os.path.join(temp_dir, "merged", "annotations")
                            os.makedirs(processed_annotations_path, exist_ok=True)
                            processed_annotations_file = os.path.join(processed_annotations_path, "annotations_noised.json")
                        else:
                            # If not merged, save annotations in the processed folder
                            processed_annotations_path = processed_annotations_dir
                            processed_annotations_file = os.path.join(processed_annotations_path, "annotations_noised.json")
                        
                        with open(processed_annotations_file, 'w') as f:
                            json.dump(processed_annotation_data, f, indent=4)
                    
                    st.success("All images have been processed with Gaussian noise and annotations updated.")
                    
                    # Create a merged ZIP
                    if merge_option:
                        merged_zip_buffer = create_zip(
                            original_folder=os.path.join(temp_dir, "original"),
                            processed_folder=os.path.join(temp_dir, "processed"),
                            merged_folder=os.path.join(temp_dir, "merged")
                        )
                    else:
                        merged_zip_buffer = create_zip(
                            original_folder=os.path.join(temp_dir, "original"),
                            processed_folder=os.path.join(temp_dir, "processed")
                        )
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"processed_dataset_{timestamp}.zip"
                    
                    # Download Button
                    st.download_button(
                        label="Download Processed Dataset ZIP",
                        data=merged_zip_buffer,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
                    
                    # Display Sample Processed Images
                    st.header("Sample Processed Images")
                    if merge_option:
                        sample_processed_images_dir = os.path.join(temp_dir, "merged", "images")
                    else:
                        sample_processed_images_dir = processed_images_dir
                    sample_processed_images = list(os.listdir(sample_processed_images_dir))[:5]  # Show up to 5 images
                    if sample_processed_images:
                        cols = st.columns(3)
                        for idx, img_name in enumerate(sample_processed_images):
                            img_path = os.path.join(sample_processed_images_dir, img_name)
                            img = Image.open(img_path)
                            with cols[idx % 3]:
                                st.image(img, caption=img_name, use_column_width=True)
                    else:
                        st.write("No processed images to display.")

# ----------------------------------------
# Function for Color Transformation
# ----------------------------------------

def color_transform(image, target_color, lower_val, upper_val):
    """
    Changes the color of specific regions of the image based on HSV range.

    :param image: NumPy array of the image in RGB format.
    :param target_color: Hex color string (e.g., "#ff0000").
    :param lower_val: NumPy array with lower HSV bounds.
    :param upper_val: NumPy array with upper HSV bounds.
    :return: Image with color transformation applied as a NumPy array.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, lower_val, upper_val)
    # Convert hex color to RGB
    target_rgb = tuple(int(target_color[i:i+2], 16) for i in (1, 3, 5))
    # Apply the target color to the masked regions
    image[mask > 0] = target_rgb
    return image

# ----------------------------------------
# Run the Application
# ----------------------------------------

if __name__ == "__main__":
    main()
