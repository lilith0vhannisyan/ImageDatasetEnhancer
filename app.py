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
# Function to Update COCO Annotations
# ----------------------------------------

def update_coco_annotations(annotation_data, original_image_id, new_image_id, new_file_name):
    """
    Updates COCO annotations by adding a new image entry and duplicating its annotations.

    Parameters:
    - annotation_data (dict): The original COCO annotation data.
    - original_image_id (int): The ID of the original image in the annotations.
    - new_image_id (int): The ID for the new noised image.
    - new_file_name (str): The filename for the new noised image.

    Returns:
    - dict: The updated annotation data with the new image and its annotations.
    """
    # Find the original image entry
    original_image = next((img for img in annotation_data['images'] if img['id'] == original_image_id), None)
    if original_image is None:
        return annotation_data  # Original image not found

    # Create a new image entry for the noised image
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
# Function to Plot Pixel Intensity Histograms
# ----------------------------------------

def plot_pixel_histograms(original_image_path, noised_image_path):
    # Load images
    original_image = Image.open(original_image_path).convert('RGB')
    noised_image = Image.open(noised_image_path).convert('RGB')
    
    original_array = np.array(original_image)
    noised_array = np.array(noised_image)
    
    # Define color channels
    color_channels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    
    for i, (channel, color) in enumerate(zip(color_channels, colors)):
        axs[0, i].hist(original_array[:, :, i].flatten(), bins=256, color=color, alpha=0.7)
        axs[0, i].set_title(f'Original Image - {channel} Channel')
        axs[0, i].set_xlim([0, 255])
        
        axs[1, i].hist(noised_array[:, :, i].flatten(), bins=256, color=color, alpha=0.7)
        axs[1, i].set_title(f'Noised Image - {channel} Channel')
        axs[1, i].set_xlim([0, 255])
    
    plt.tight_layout()
    st.pyplot(fig)

# ----------------------------------------
# Function to Plot Image Counts
# ----------------------------------------

def plot_image_counts(original_count, noised_count):
    data = {
        'Dataset': ['Original', 'Noised'],
        'Image Count': [original_count, noised_count]
    }
    fig = px.bar(data, x='Dataset', y='Image Count', color='Dataset', text='Image Count',
                 color_discrete_map={'Original':'blue', 'Noised':'orange'})
    fig.update_traces(textposition='auto')
    fig.update_layout(title='Image Count Comparison', yaxis_title='Number of Images', xaxis_title='')
    st.plotly_chart(fig)

# ----------------------------------------
# Function to Plot Annotation Counts per Category
# ----------------------------------------

def plot_annotation_counts(annotation_data):
    # Extract categories
    categories = {cat['id']: cat['name'] for cat in annotation_data.get('categories', [])}
    
    # Initialize counts
    counts = {}
    for cat_id, cat_name in categories.items():
        counts[cat_name] = {'Original':0, 'Noised':0}
    
    # Count annotations
    for ann in annotation_data['annotations']:
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, 'Unknown')
        img_id = ann['image_id']
        img = next((img for img in annotation_data['images'] if img['id'] == img_id), None)
        if img:
            if "_noised" in img['file_name']:
                counts[cat_name]['Noised'] +=1
            else:
                counts[cat_name]['Original'] +=1
    
    # Prepare data for plotting
    data = {
        'Category': [],
        'Dataset': [],
        'Annotation Count': []
    }
    
    for cat, datasets in counts.items():
        for dataset, count in datasets.items():
            data['Category'].append(cat)
            data['Dataset'].append(dataset)
            data['Annotation Count'].append(count)
    
    fig = px.bar(data, x='Category', y='Annotation Count', color='Dataset', barmode='group',
                 title='Annotation Count per Category',
                 color_discrete_map={'Original':'blue', 'Noised':'orange'})
    fig.update_layout(xaxis_title='Category', yaxis_title='Number of Annotations')
    st.plotly_chart(fig)

# ----------------------------------------
# Function to Display Sample Image Comparisons
# ----------------------------------------

def display_sample_comparisons(annotation_data, images_folder, num_samples=3):
    # Extract image pairs
    noised_images = [img for img in annotation_data['images'] if "_noised" in img['file_name']]
    
    if not noised_images:
        st.write("No noised images to display.")
        return
    
    for noised_img in noised_images[:num_samples]:
        original_file_name = noised_img['file_name'].replace('_noised', '')
        original_img = next((img for img in annotation_data['images'] if img['file_name'] == original_file_name), None)
        if original_img:
            cols = st.columns(2)
            with cols[0]:
                st.image(os.path.join(images_folder, original_img['file_name']), caption=f"Original: {original_img['file_name']}", use_column_width=True)
            with cols[1]:
                st.image(os.path.join(images_folder, noised_img['file_name']), caption=f"Noised: {noised_img['file_name']}", use_column_width=True)

# ----------------------------------------
# Function to Display Dataset Statistics
# ----------------------------------------

def display_dataset_statistics(original_count, noised_count, original_ann, noised_ann, categories):
    st.header("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Images", original_count)
    
    with col2:
        st.metric("Noised Images", noised_count)
    
    with col3:
        st.metric("Original Annotations", original_ann)
    
    with col4:
        st.metric("Noised Annotations", noised_ann)
    
    st.write(f"**Total Categories:** {len(categories)}")

# ----------------------------------------
# Streamlit User Interface
# ----------------------------------------

def main():
    # Configure the Streamlit page
    st.set_page_config(page_title="Dataset Augmentation Tool", layout="wide")
    st.title('Dataset Augmentation Tool')
    
    st.markdown("""
    This application allows you to upload a `dataset.zip` file containing images and a single JSON annotation file (e.g., COCO format). 
    It applies Gaussian noise to all images in the dataset, merges the noised images with the original ones, updates the annotations accordingly, 
    and provides downloadable statistics and a merged dataset ZIP file.
    """)
    
    st.header("Upload Dataset ZIP File")
    
    # File uploader for ZIP files
    uploaded_zip = st.file_uploader("Upload `dataset.zip`", type=["zip"])
    
    if uploaded_zip is not None:
        with TemporaryDirectory() as temp_dir:
            # Extract the uploaded ZIP file into the temporary directory
            with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        
            st.success("Dataset ZIP file uploaded and extracted successfully.")
        
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
                    img = Image.open(img_path)
                    with cols[idx]:
                        st.image(img, caption=Path(img_path).name, use_column_width=True)
        
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
        
            # Gaussian Noise Options
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
        
            # Apply Processing Button
            if st.button("Apply Gaussian Noise to All Images"):
                if noise_type == "None":
                    st.warning("No noise type selected. Please choose a noise type to apply.")
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
                            image = Image.open(original_image_path).convert('RGB')
                            image_array = np.array(image)
    
                            # Apply Gaussian noise
                            noised_image_array = add_gaussian_noise(image_array, mean=params['mean'], std=params['std_dev'])
    
                            # Convert the noised array back to an image
                            noised_image = Image.fromarray(noised_image_array)
    
                            # Define the new filename with '_noised' suffix
                            name, ext = os.path.splitext(original_file_name)
                            noised_file_name = f"{name}_noised{ext}"
                            processed_image_path = os.path.join(temp_dir, noised_file_name)
    
                            # Save the noised image
                            noised_image.save(processed_image_path)
    
                            # Update annotations for the noised image
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
    
                            # Move the noised image to the images folder
                            shutil.move(processed_image_path, images_folder)
    
                        # Save the updated annotations JSON
                        processed_annotations_path = os.path.join(annotations_folder, "annotations.json")
                        with open(processed_annotations_path, 'w') as f:
                            json.dump(processed_annotation_data, f, indent=4)
    
                    st.success("All images have been processed with Gaussian noise and annotations updated.")
    
                    # Create a merged ZIP containing all images and the updated annotations
                    merged_zip_buffer = create_zip(
                        images_folder=images_folder,
                        annotations_file=processed_annotations_path
                    )
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    zip_filename = f"merged_dataset_{timestamp}.zip"
    
                    # Provide a download button for the merged ZIP
                    st.download_button(
                        label="Download Merged Dataset ZIP",
                        data=merged_zip_buffer,
                        file_name=zip_filename,
                        mime="application/zip"
                    )
    
                    # Calculate counts
                    original_count = len([img for img in annotation_data['images'] if "_noised" not in img['file_name']])
                    noised_count = len([img for img in annotation_data['images'] if "_noised" in img['file_name']])
                    
                    # Calculate annotation counts
                    original_ann = len([ann for ann in annotation_data['annotations'] if not any("_noised" in img['file_name'] for img in annotation_data['images'] if img['id'] == ann['image_id'])])
                    noised_ann = len([ann for ann in processed_annotation_data['annotations'] if any("_noised" in img['file_name'] for img in processed_annotation_data['images'] if img['id'] == ann['image_id'])])
                    
                    # Get categories
                    categories = [cat['name'] for cat in processed_annotation_data.get('categories', [])]
    
                    # Display dataset statistics
                    display_dataset_statistics(original_count, noised_count, original_ann, noised_ann, categories)
    
                    # Plot image counts
                    st.header("Image Count Comparison")
                    plot_image_counts(original_count, noised_count)
    
                    # Plot annotation counts per category
                    st.header("Annotation Count per Category")
                    plot_annotation_counts(processed_annotation_data)
    
                    # Display sample image comparisons
                    st.header("Sample Image Comparisons")
                    display_sample_comparisons(processed_annotation_data, images_folder, num_samples=3)
    
                    # Display pixel intensity histograms for sample images
                    st.header("Pixel Intensity Histograms")
                    sample_pairs = zip(
                        [os.path.join(images_folder, img['file_name']) for img in processed_annotation_data['images'] if "_noised" in img['file_name']][:3],
                        [os.path.join(images_folder, img['file_name'].replace('_noised', '')) for img in processed_annotation_data['images'] if "_noised" in img['file_name']][:3]
                    )
    
                    for noised_path, original_path in sample_pairs:
                        st.subheader(f"Histogram Comparison: {Path(original_path).name} vs {Path(noised_path).name}")
                        plot_pixel_histograms(original_path, noised_path)

# ----------------------------------------
# Run the Application
# ----------------------------------------

if __name__ == "__main__":
    main()
