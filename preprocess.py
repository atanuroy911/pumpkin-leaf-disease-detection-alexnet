"""
Leaf Disease Detection - Data Preprocessing
This script:
1. Reads images from class folders
2. Applies data augmentation (rotation, flip, brightness, zoom)
3. Resizes images to 224x224
4. Saves resized images to Dataset_Resized folder
5. Creates a CSV with all image metadata
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
DATASET_PATH = "Dataset"
OUTPUT_PATH = "Dataset_Resized"
CSV_OUTPUT = "dataset_info.csv"
IMG_SIZE = 224
AUGMENTATIONS_PER_IMAGE = 2

# Class mapping
CLASS_FOLDERS = [
    "Bacterial Leaf Spot",
    "Downy Mildew",
    "Healthy Leaf",
    "Mosaic Disease",
    "Powdery_Mildew"
]

def create_augmentation_pipeline():
    """Create augmentation pipeline using albumentations"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
        ], p=0.3),
    ])

def resize_image(image, size=224):
    """Resize image to specified size"""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

def save_image(image, path):
    """Save image to specified path"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)

def process_dataset():
    """Main processing function"""
    print("Starting data preprocessing...")
    print(f"Target image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Initialize augmentation
    augment = create_augmentation_pipeline()
    
    # Data storage
    data_records = []
    
    # Process each class
    for class_idx, class_name in enumerate(CLASS_FOLDERS):
        print(f"\nProcessing class: {class_name} (Label: {class_idx})")
        
        class_input_path = os.path.join(DATASET_PATH, class_name)
        class_output_path = os.path.join(OUTPUT_PATH, class_name)
        
        # Check if class folder exists
        if not os.path.exists(class_input_path):
            print(f"Warning: Folder {class_input_path} not found. Skipping...")
            continue
        
        # Get all images in class folder
        image_files = [f for f in os.listdir(class_input_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images")
        
        # Process each image
        for img_idx, img_file in enumerate(tqdm(image_files, desc=f"Processing {class_name}")):
            original_path = os.path.join(class_input_path, img_file)
            
            # Read original image
            img = cv2.imread(original_path)
            if img is None:
                print(f"Warning: Could not read {original_path}. Skipping...")
                continue
            
            original_height, original_width = img.shape[:2]
            
            # Save original resized image (no augmentation)
            img_resized = resize_image(img, IMG_SIZE)
            base_name = Path(img_file).stem
            ext = Path(img_file).suffix
            
            output_filename = f"{base_name}_orig{ext}"
            output_path = os.path.join(class_output_path, output_filename)
            save_image(img_resized, output_path)
            
            # Record original image data
            data_records.append({
                'filename': output_filename,
                'original_filename': img_file,
                'class_name': class_name,
                'class_label': class_idx,
                'original_width': original_width,
                'original_height': original_height,
                'resized_width': IMG_SIZE,
                'resized_height': IMG_SIZE,
                'augmentation_type': 'original',
                'relative_path': os.path.join(class_name, output_filename),
                'original_image_path': original_path
            })
            
            # Create augmented versions
            for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                # Apply augmentation
                augmented = augment(image=img)
                img_aug = augmented['image']
                
                # Resize augmented image
                img_aug_resized = resize_image(img_aug, IMG_SIZE)
                
                # Save augmented image
                aug_filename = f"{base_name}_aug{aug_idx+1}{ext}"
                aug_output_path = os.path.join(class_output_path, aug_filename)
                save_image(img_aug_resized, aug_output_path)
                
                # Record augmented image data
                data_records.append({
                    'filename': aug_filename,
                    'original_filename': img_file,
                    'class_name': class_name,
                    'class_label': class_idx,
                    'original_width': original_width,
                    'original_height': original_height,
                    'resized_width': IMG_SIZE,
                    'resized_height': IMG_SIZE,
                    'augmentation_type': f'augmented_{aug_idx+1}',
                    'relative_path': os.path.join(class_name, aug_filename),
                    'original_image_path': original_path
                })
    
    # Create DataFrame and save to CSV
    print(f"\nCreating CSV file: {CSV_OUTPUT}")
    df = pd.DataFrame(data_records)
    df.to_csv(CSV_OUTPUT, index=False)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(df)}")
    print(f"CSV file saved: {CSV_OUTPUT}")
    print(f"Resized dataset folder: {OUTPUT_PATH}")
    print("\nClass distribution:")
    print(df['class_name'].value_counts().sort_index())
    print("\nAugmentation distribution:")
    print(df['augmentation_type'].value_counts())
    print("="*60)
    
    return df

if __name__ == "__main__":
    # Install required packages if needed
    print("Make sure you have installed required packages:")
    print("pip install opencv-python pandas numpy tqdm albumentations")
    print("\n")
    
    # Run preprocessing
    df = process_dataset()
    print("\nPreprocessing complete!")
