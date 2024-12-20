import streamlit as st
import numpy as np
from PIL import Image
import os
import io
import zipfile
from datetime import datetime
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
from skimage.metrics import structural_similarity as ssim  # Imported but not used

# ----------------------------------------
# Function to Add Gaussian Noise to Images
# ----------------------------------------

def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image (numpy.ndarray): The input image in RGB format.
    - mean (float): Mean of the Gaussian noise.
    - std (float): Standard deviation of the Gaussian noise.

    Returns:
    - numpy.ndarray: The noised image.
    """
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# ----------------------------------------
# Function to Add Salt-and-Pepper Noise to Images
# ----------------------------------------

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Adds salt-and-pepper noise to an image.

    Parameters:
    - image (numpy.ndarray): The input image in RGB format.
    - salt_prob (float): Probability of adding salt noise.
    - pepper_prob (float): Probability of adding pepper noise.

    Returns:
    - numpy.ndarray: The noised image.
    """
    noisy = image.copy()
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0

    return noisy

# ----------------------------------------
# Function to Update COCO Annotations
# ----------------------------------------

def update_coco_annotations(annotation_data, original_image_id, new_image_id, new_file_name):
    """
    Updates COCO annotations by adding a new image entry and duplicating its annotations.

    Parameters:
    - annotation_data (dict): The original COCO annotation data.
    - original_image_id (int): The ID of the original image in the annotations.
    - new_image_id (int): The ID for the new augmented image.
    - new_file_name (str): The filename for the new augmented image.

    Returns:
    - dict: The updated annotation data with the new image and its annotations.
    """
    # Find the original image entry
    original_image = next((img for img in annotation_data['images'] if img['id'] == original_image_id), None)
    if original_image is None:
        return annotation_data  # Original image not found

    # Create a new image entry for the augmented image
    new_image = original_image.copy()
    new_image['id'] = new_image_id
    new_image['file_name'] = new_file_name
    annotation_data['images'].append(new_image)

    # Duplicate all annotations associated with the original image
    for ann in annotation_data['annotations']:
        if ann['image_id'] == original_image_id:
            new_ann = ann.copy()
            new_ann['id'] = max(a['id'] for a in annotation_data['annotations']) + 1
            new_ann['image_id'] = new_image_id
            annotation_data['annotations'].append(new_ann)

    return annotation_data

# ----------------------------------------
# Function to Create ZIP Archive
# ----------------------------------------

def create_zip(images_folder, annotations_file):
    """
    Creates a ZIP archive containing all images and the updated annotations.

    Parameters:
    - images_folder (str): Path to the folder containing all images.
    - annotations_file (str): Path to the updated annotations JSON file.

    Returns:
    - BytesIO: The ZIP archive in memory.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add all images to the 'images/' directory in the ZIP
        for root, _, files in os.walk(images_folder):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.join('images', os.path.relpath(filepath, images_folder))
                zipf.write(filepath, arcname)

        # Add the updated annotations to the 'annotations/' directory in the ZIP
        zipf.write(annotations_file, os.path.join('annotations', 'annotations.json'))

    zip_buffer.seek(0)
    return zip_buffer

# ----------------------------------------
# Function to Plot SSIM and PSNR Scores
# (Removed as per request)
# ----------------------------------------

# ----------------------------------------
# Streamlit User Interface
# ----------------------------------------

def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Dataset Augmentation Tool", layout="wide")
    st.title('Dataset Augmentation Tool')

    st.markdown("""
    This application allows you to upload a `dataset.zip` file containing images and a single COCO-format JSON annotation file. 
    You can select the type of noise to add to the images, and after processing, download a merged ZIP file containing both original and noised images with updated annotations.
    """)

    st.header("1. Upload Dataset ZIP File")

    # File uploader for ZIP files
    uploaded_zip = st.file_uploader("Upload `dataset.zip`", type=["zip"])

    if uploaded_zip is not None:
        with TemporaryDirectory() as temp_dir:
            # Extract the uploaded ZIP file into the temporary directory
            try:
                with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                st.success("Dataset ZIP file uploaded and extracted successfully.")
            except zipfile.BadZipFile:
                st.error("The uploaded file is not a valid ZIP archive.")
                st.stop()

            # Define paths for organizing images and annotations
            images_folder = os.path.join(temp_dir, "images")
            annotations_folder = os.path.join(temp_dir, "annotations")
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(annotations_folder, exist_ok=True)

            # Define acceptable file extensions
            image_extensions = ['.jpg', '.jpeg', '.png']
            annotation_extensions = ['.json']

            # Gather all image and annotation files from the extracted content
            all_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))

            image_files = [f for f in all_files if Path(f).suffix.lower() in image_extensions]
            annotation_files = [f for f in all_files if Path(f).suffix.lower() in annotation_extensions]

            st.write(f"**Total Images Found:** {len(image_files)}")
            st.write(f"**Total Annotation Files Found:** {len(annotation_files)}")

            # Optional: Display sample images
            if st.checkbox("Show Sample Image Files"):
                sample_images = image_files[:3]
                cols = st.columns(len(sample_images))
                for idx, img_path in enumerate(sample_images):
                    try:
                        img = Image.open(img_path)
                        with cols[idx]:
                            st.image(img, caption=Path(img_path).name, use_column_width=True)
                    except Exception as e:
                        st.warning(f"Failed to load image {Path(img_path).name}: {e}")

            # Optional: Display sample annotations
            if st.checkbox("Show Sample Annotation Files"):
                sample_annotations = annotation_files[:3]
                for ann_path in sample_annotations:
                    with open(ann_path, 'r') as f:
                        try:
                            ann_data = json.load(f)
                            st.json(ann_data)
                        except json.JSONDecodeError:
                            st.warning(f"Failed to parse {Path(ann_path).name} as JSON.")

            # Validate that there is exactly one JSON annotation file
            if len(annotation_files) != 1:
                st.error("Please ensure that the ZIP file contains exactly one JSON annotation file in COCO format.")
                st.stop()
            else:
                annotation_file = annotation_files[0]
                with open(annotation_file, 'r') as f:
                    try:
                        annotation_data = json.load(f)
                        st.write(f"**Loaded Annotation File:** {Path(annotation_file).name}")
                    except json.JSONDecodeError:
                        st.error(f"Failed to parse {Path(annotation_file).name} as JSON.")
                        st.stop()

            # Display a summary of the annotations
            st.write(f"**Number of Images in Annotations:** {len(annotation_data.get('images', []))}")
            st.write(f"**Number of Annotations:** {len(annotation_data.get('annotations', []))}")
            st.write(f"**Number of Categories:** {len(annotation_data.get('categories', []))}")

            # ----------------------------------------
            # Noise Selection Options
            # ----------------------------------------
            st.header("2. Select Noise Augmentation Options")

            # Gaussian Noise Options
            st.subheader("Gaussian Noise")
            apply_gaussian = st.checkbox("Apply Gaussian Noise")
            gaussian_params = {}
            if apply_gaussian:
                mean_gaussian = st.slider("Gaussian Noise Mean", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
                std_dev_gaussian = st.slider("Gaussian Noise Standard Deviation", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
                gaussian_params['mean'] = mean_gaussian
                gaussian_params['std_dev'] = std_dev_gaussian

            # Salt-and-Pepper Noise Options
            st.subheader("Salt-and-Pepper Noise")
            apply_sp = st.checkbox("Apply Salt-and-Pepper Noise")
            sp_params = {}
            if apply_sp:
                salt_prob = st.slider("Salt Probability", min_value=0.0, max_value=0.05, value=0.01, step=0.005)
                pepper_prob = st.slider("Pepper Probability", min_value=0.0, max_value=0.05, value=0.01, step=0.005)
                sp_params['salt_prob'] = salt_prob
                sp_params['pepper_prob'] = pepper_prob

            # ----------------------------------------
            # Apply Augmentations Button
            # ----------------------------------------
            st.header("3. Apply Augmentations")

            if st.button("Apply Augmentations to All Images"):
                if not apply_gaussian and not apply_sp:
                    st.warning("Please select at least one noise type to apply.")
                else:
                    with st.spinner('Processing images and updating annotations...'):
                        # Initialize the processed annotations data
                        processed_annotation_data = annotation_data.copy()
                        processed_annotation_data['images'] = annotation_data['images'].copy()
                        processed_annotation_data['annotations'] = annotation_data['annotations'].copy()
                        # 'categories' remain unchanged

                        # Determine the maximum image and annotation IDs to assign unique IDs
                        existing_image_ids = [img['id'] for img in annotation_data.get('images', [])]
                        existing_annotation_ids = [ann['id'] for ann in annotation_data.get('annotations', [])]
                        max_image_id = max(existing_image_ids) if existing_image_ids else 0
                        max_annotation_id = max(existing_annotation_ids) if existing_annotation_ids else 0

                        # Move all original images to the images_folder
                        for img_path in image_files:
                            shutil.move(img_path, images_folder)

                        # Update image_files list to reflect the new location
                        image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) 
                                       if Path(f).suffix.lower() in image_extensions]

                        # Process each image in the annotations
                        for img in annotation_data.get('images', []):
                            original_image_id = img['id']
                            original_file_name = img['file_name']
                            original_image_path = os.path.join(images_folder, original_file_name)

                            # Check if the image file exists
                            if not os.path.exists(original_image_path):
                                st.warning(f"Image file {original_file_name} not found in the ZIP. Skipping.")
                                continue

                            # Load the original image
                            try:
                                image = Image.open(original_image_path).convert('RGB')
                                image_array = np.array(image)
                            except Exception as e:
                                st.warning(f"Failed to load image {original_file_name}: {e}")
                                continue

                            # Apply Gaussian noise if selected
                            if apply_gaussian:
                                image_array = add_gaussian_noise(image_array, mean=gaussian_params['mean'], std=gaussian_params['std_dev'])

                            # Apply Salt-and-Pepper noise if selected
                            if apply_sp:
                                image_array = add_salt_pepper_noise(image_array, salt_prob=sp_params['salt_prob'], pepper_prob=sp_params['pepper_prob'])

                            # Convert the array back to an image
                            augmented_image = Image.fromarray(image_array)

                            # Define the new filename with augmentation suffix
                            augmentation_suffix = ""
                            if apply_gaussian:
                                augmentation_suffix += "_gaussian"
                            if apply_sp:
                                augmentation_suffix += "_sp"
                            if augmentation_suffix == "":
                                augmentation_suffix = "_augmented"

                            name, ext = os.path.splitext(original_file_name)
                            augmented_file_name = f"{name}{augmentation_suffix}{ext}"
                            processed_image_path = os.path.join(temp_dir, augmented_file_name)

                            # Save the augmented image
                            try:
                                augmented_image.save(processed_image_path)
                            except Exception as e:
                                st.warning(f"Failed to save augmented image {augmented_file_name}: {e}")
                                continue

                            # Update annotations for the augmented image
                            max_image_id += 1
                            new_image_entry = img.copy()
                            new_image_entry['id'] = max_image_id
                            new_image_entry['file_name'] = augmented_file_name
                            processed_annotation_data['images'].append(new_image_entry)

                            # Duplicate and update annotations for the new image
                            for ann in annotation_data.get('annotations', []):
                                if ann['image_id'] == original_image_id:
                                    max_annotation_id += 1
                                    new_ann = ann.copy()
                                    new_ann['id'] = max_annotation_id
                                    new_ann['image_id'] = max_image_id
                                    processed_annotation_data['annotations'].append(new_ann)

                            # Move the augmented image to the images folder
                            shutil.move(processed_image_path, images_folder)

                        # Save the updated annotations JSON
                        processed_annotations_path = os.path.join(annotations_folder, "annotations.json")
                        try:
                            with open(processed_annotations_path, 'w') as f:
                                json.dump(processed_annotation_data, f, indent=4)
                        except Exception as e:
                            st.error(f"Failed to save updated annotations: {e}")
                            st.stop()

                    st.success("All augmentations have been applied and annotations updated.")

                    # Create a merged ZIP containing all images and the updated annotations
                    try:
                        merged_zip_buffer = create_zip(
                            images_folder=images_folder,
                            annotations_file=processed_annotations_path
                        )
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        zip_filename = f"merged_dataset_{timestamp}.zip"
                    except Exception as e:
                        st.error(f"Failed to create merged ZIP: {e}")
                        st.stop()

                    # Provide a download button for the merged ZIP
                    st.download_button(
                        label="Download Merged Dataset ZIP",
                        data=merged_zip_buffer,
                        file_name=zip_filename,
                        mime="application/zip"
                    )

                    # ----------------------------------------
                    # Removed Sections:
                    # - SSIM and PSNR Analysis
                    # - Dataset Statistics
                    # - Image Count Comparison
                    # - Annotation Count per Category
                    # - Sample Image Comparisons
                    # - Pixel Intensity Histograms
                    # ----------------------------------------

if __name__ == "__main__":
    main()
