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
    
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# ----------------------------------------
# Function to Add Salt-and-Pepper Noise to Images
# ----------------------------------------

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
   
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
# PREVIEW HELPER FUNCTIONS
# ----------------------------------------

def show_image_and_hist(image_array, title=""):
    
    # Display the image with a caption
    st.image(image_array, caption=title, use_column_width=True)

    # Flatten the 3D array (for RGB) or 2D array (for grayscale) into 1D
    flattened_pixels = image_array.ravel()

   # Create a histogram of pixel intensities using Plotly Express
    fig = px.histogram(
        x=flattened_pixels,
        nbins=256,  # 256 bins to cover 0–255 intensities
        range_x=[0, 255],  # Intensity range
        title=(
            f"{title} – Pixel Intensity Distribution<br>"
            "<sup>X-axis: Pixel Intensity Values (ranging from 0 to 255 "
            "for 8-bit images), Y-axis: Number of Pixels</sup>"
        ),
        labels={
            "x": "Pixel Intensity Values (ranging from 0 to 255 for 8-bit images)",
            "y": "Count of Pixels"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

def preview_sample_image(
    image_files,
    apply_gaussian=False,
    gaussian_mean=0,
    gaussian_std=25,
    apply_sp=False,
    sp_salt=0.01,
    sp_pepper=0.01
):
    
    if not image_files:
        st.warning("No sample images available to preview.")
        return
    
    # Take the first image (or any you choose)
    sample_image_path = image_files[0]
    try:
        image = Image.open(sample_image_path).convert('RGB')
        original_arr = np.array(image)
    except Exception as e:
        st.warning(f"Could not open sample image: {e}")
        return

    # Prepare copies for each noise type
    gaussian_arr = original_arr.copy()
    saltpepper_arr = original_arr.copy()

    # Apply Gaussian noise if selected
    if apply_gaussian:
        gaussian_arr = add_gaussian_noise(
            gaussian_arr, 
            mean=gaussian_mean, 
            std=gaussian_std
        )

    # Apply Salt-and-Pepper noise if selected
    if apply_sp:
        saltpepper_arr = add_salt_pepper_noise(
            saltpepper_arr,
            salt_prob=sp_salt,
            pepper_prob=sp_pepper
        )

    # Decide how many columns: 1 if no noise, 2 if one noise, 3 if both
    columns_to_create = 1 + int(apply_gaussian) + int(apply_sp)

    if columns_to_create == 1:
        # Only the Original image
        col = st.columns(1)
        with col[0]:
            show_image_and_hist(original_arr, "Original Image")

    elif columns_to_create == 2:
        # Original + one noise
        col = st.columns(2)
        idx = 0
        with col[idx]:
            show_image_and_hist(original_arr, "Original Image")
        if apply_gaussian:
            idx += 1
            with col[idx]:
                show_image_and_hist(gaussian_arr, "Gaussian-Noise Image")
        else:
            idx += 1
            with col[idx]:
                show_image_and_hist(saltpepper_arr, "Salt-and-Pepper Image")

    else:
        # 3 columns: Original, Gaussian, Salt-and-Pepper
        col1, col2, col3 = st.columns(3)
        with col1:
            show_image_and_hist(original_arr, "Original Image")
        with col2:
            show_image_and_hist(gaussian_arr, "Gaussian-Noise Image")
        with col3:
            show_image_and_hist(saltpepper_arr, "Salt-and-Pepper Image")

# ----------------------------------------
# Function to Update COCO Annotations
# ----------------------------------------

def update_coco_annotations(annotation_data, original_image_id, new_image_id, new_file_name):
    
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
# Streamlit User Interface
# ----------------------------------------

def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Dataset Augmentation Tool", layout="wide")
    st.title('Dataset Augmentation Tool')

    st.markdown("""
    This application allows you to upload a `dataset.zip` file containing images and a single COCO-format JSON annotation file. 
    You can select the type of noise to add to the images, preview a sample image, and after processing, download a merged ZIP file 
    containing both original and noised images with updated annotations.
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
                mean_gaussian = st.slider(
                    "Gaussian Noise Mean (μ)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.5
                )
                std_dev_gaussian = st.slider(
                    "Gaussian Noise Standard Deviation (σ)",
                    min_value=1.0,
                    max_value=100.0,
                    value=25.0,
                    step=1.0
                )
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
            # Preview Sample Image (Optional)
            # ----------------------------------------
            st.header("3. Preview a Sample Image with Selected Noise Options")
            if st.button("Preview Transformations"):
                # Show a single sample image in original, Gaussian, and/or Salt-Pepper form
                preview_sample_image(
                    image_files=image_files,
                    apply_gaussian=apply_gaussian,
                    gaussian_mean=gaussian_params.get('mean', 0),
                    gaussian_std=gaussian_params.get('std_dev', 25),
                    apply_sp=apply_sp,
                    sp_salt=sp_params.get('salt_prob', 0.01),
                    sp_pepper=sp_params.get('pepper_prob', 0.01)
                )

            # ----------------------------------------
            # Apply Augmentations Button
            # ----------------------------------------
            st.header("4. Apply Augmentations to All Images")

            if st.button("Apply Augmentations to All Images"):
                if not apply_gaussian and not apply_sp:
                    st.warning("Please select at least one noise type to apply.")
                else:
                    with st.spinner('Processing images and updating annotations...'):
                        # Initialize the processed annotations data
                        processed_annotation_data = annotation_data.copy()
                        processed_annotation_data['images'] = annotation_data['images'].copy()
                        processed_annotation_data['annotations'] = annotation_data['annotations'].copy()

                        # Determine the maximum image and annotation IDs to assign unique IDs
                        existing_image_ids = [img['id'] for img in annotation_data.get('images', [])]
                        existing_annotation_ids = [ann['id'] for ann in annotation_data.get('annotations', [])]
                        max_image_id = max(existing_image_ids) if existing_image_ids else 0
                        max_annotation_id = max(existing_annotation_ids) if existing_annotation_ids else 0

                        # Move all original images to the images_folder
                        for img_path in image_files:
                            shutil.move(img_path, images_folder)

                        # Update image_files list to reflect the new location
                        image_files = [
                            os.path.join(images_folder, f) 
                            for f in os.listdir(images_folder)
                            if Path(f).suffix.lower() in image_extensions
                        ]

                        # Process each image in the annotations
                        for img in annotation_data.get('images', []):
                            original_image_id = img['id']
                            original_file_name = img['file_name']
                            original_image_path = os.path.join(images_folder, original_file_name)

                            if not os.path.exists(original_image_path):
                                st.warning(f"Image file {original_file_name} not found in the ZIP. Skipping.")
                                continue

                            # Load the original image
                            try:
                                original_image_pil = Image.open(original_image_path).convert('RGB')
                                original_array = np.array(original_image_pil)
                            except Exception as e:
                                st.warning(f"Failed to load image {original_file_name}: {e}")
                                continue

                            # -------------------------------------------------
                            # 1) If Gaussian is selected, create a Gaussian copy
                            # -------------------------------------------------
                            if apply_gaussian:
                                # Work on the *original* array, not a previously noised array
                                gaussian_arr = add_gaussian_noise(
                                    original_array,
                                    mean=gaussian_params['mean'],
                                    std=gaussian_params['std_dev']
                                )

                                gaussian_image = Image.fromarray(gaussian_arr)

                                # Generate a new filename with a _gaussian suffix
                                name, ext = os.path.splitext(original_file_name)
                                gaussian_file_name = f"{name}_gaussian{ext}"
                                processed_gaussian_path = os.path.join(temp_dir, gaussian_file_name)

                                # Save the Gaussian image
                                try:
                                    gaussian_image.save(processed_gaussian_path)
                                except Exception as e:
                                    st.warning(f"Failed to save Gaussian image {gaussian_file_name}: {e}")
                                    continue

                                # Update annotations for the Gaussian image
                                max_image_id += 1
                                new_image_entry = img.copy()
                                new_image_entry['id'] = max_image_id
                                new_image_entry['file_name'] = gaussian_file_name
                                processed_annotation_data['images'].append(new_image_entry)

                                # Duplicate and update annotations for the new Gaussian image
                                for ann in annotation_data.get('annotations', []):
                                    if ann['image_id'] == original_image_id:
                                        max_annotation_id += 1
                                        new_ann = ann.copy()
                                        new_ann['id'] = max_annotation_id
                                        new_ann['image_id'] = max_image_id
                                        processed_annotation_data['annotations'].append(new_ann)

                                # Move the Gaussian image to the images folder
                                shutil.move(processed_gaussian_path, images_folder)

                            # ----------------------------------------------------------------
                            # 2) If Salt-and-Pepper is selected, create Salt-Pepper copy
                            # ----------------------------------------------------------------
                            if apply_sp:
                                # Again, start from the *original* array
                                sp_arr = add_salt_pepper_noise(
                                    original_array,
                                    salt_prob=sp_params['salt_prob'],
                                    pepper_prob=sp_params['pepper_prob']
                                )

                                sp_image = Image.fromarray(sp_arr)

                                # Generate a new filename with a _sp suffix
                                name, ext = os.path.splitext(original_file_name)
                                sp_file_name = f"{name}_sp{ext}"
                                processed_sp_path = os.path.join(temp_dir, sp_file_name)

                                # Save the Salt-and-Pepper image
                                try:
                                    sp_image.save(processed_sp_path)
                                except Exception as e:
                                    st.warning(f"Failed to save Salt-and-Pepper image {sp_file_name}: {e}")
                                    continue

                                # Update annotations for the Salt-and-Pepper image
                                max_image_id += 1
                                new_image_entry = img.copy()
                                new_image_entry['id'] = max_image_id
                                new_image_entry['file_name'] = sp_file_name
                                processed_annotation_data['images'].append(new_image_entry)

                                # Duplicate and update annotations for the new SP image
                                for ann in annotation_data.get('annotations', []):
                                    if ann['image_id'] == original_image_id:
                                        max_annotation_id += 1
                                        new_ann = ann.copy()
                                        new_ann['id'] = max_annotation_id
                                        new_ann['image_id'] = max_image_id
                                        processed_annotation_data['annotations'].append(new_ann)

                                # Move the Salt-and-Pepper image to the images folder
                                shutil.move(processed_sp_path, images_folder)

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


if __name__ == "__main__":
    main()
