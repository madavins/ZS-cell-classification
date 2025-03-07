import argparse
import json
from collections import Counter
from pathlib import Path
import cv2

def extract_cropped_images(input_folder, output_folder, crop_size=112,
                           min_distance_from_edge=56, min_area=150, verbose=False):
    """
    Extracts cropped images centered on cells from a dataset with COCO-style annotations.
    Creates cropped images and a corresponding annotations file.
    Generic version, not specific to PanNuke or PUMA.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_images_dir = output_folder / 'images'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_path = output_folder / "annotations.json"

    annotations_output = {'annotations': [], 'categories': []}
    crop_id = 0

    print(f"Processing: {input_folder}")
    annotations_path = input_folder / "annotations.json"
    images_dir = input_folder / "images"


    if not annotations_path.exists():
      raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
        
    annotations_output['categories'] = annotations['categories']

    image_to_annotations = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(ann)
            
    for img_info in annotations['images']:
        image_id = img_info['id']
        file_name = img_info['file_name']
        image_path = images_dir / file_name

        if not image_path.exists():
          print(f"Warning: Could not read image {image_path}")
          continue
        
        image = cv2.imread(str(image_path))
        image_height, image_width = image.shape[:2]
        
        for ann in image_to_annotations.get(image_id, []):
            category_id = ann['category_id']
            bbox = ann['bbox']
            area = ann['area']
            segmentation = ann['segmentation']

            x, y, w, h = map(int, bbox)
            
            if area < min_area:
                if verbose:
                    print(f"Cell: {crop_id} in image: {image_id} discarded by area. Area: {area}")
                continue
            
            center_x, center_y = x + w // 2, y + h // 2

            # Check if cell is too close to image border
            if not (min_distance_from_edge <= center_x < image_width - min_distance_from_edge and
                min_distance_from_edge <= center_y < image_height - min_distance_from_edge):
                if verbose:
                    print(f"Cell: {crop_id} in image: {image_id} discarded by position. Center: ({center_x}, {center_y})")
                continue
                
            # Calculate crop coordinates centered on the cell
            crop_x1 = center_x - crop_size//2
            crop_y1 = center_y - crop_size//2
            crop_x2 = center_x + crop_size//2
            crop_y2 = center_y + crop_size//2

            # Calculate padding needed if crop extends beyond image borders
            pad_left = max(0, -crop_x1)
            pad_right = max(0, crop_x2 - image_width)
            pad_top = max(0, -crop_y1)
            pad_bottom = max(0, crop_y2 - image_height)
            
            is_padded = pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0
            
            # Apply mirror padding if necessary
            if is_padded:
                padded_image = cv2.copyMakeBorder(
                    image,
                    pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_REFLECT
                )
                # Adjust crop coordinates for padded image
                crop_x1 += pad_left
                crop_x2 += pad_left
                crop_y1 += pad_top
                crop_y2 += pad_top
            else:
                padded_image = image                
            
            try:
                # Extract the square crop centered on the cell
                crop = padded_image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                filename = f"image_{crop_id}.png"
                save_path = str(output_images_dir / filename)
                cv2.imwrite(save_path, crop)

                # Map category id to cell type
                category_name = next(cat['name'] for cat in annotations['categories'] if cat['id'] == category_id)

                annotations_output['annotations'].append({
                    'cell_id': crop_id,
                    'whole_image_id': image_id,
                    'file_name': filename,
                    'category_id': category_id,  
                    'category_name': category_name,  
                    'tissue': img_info.get('tissue', 'unknown'), #Not all datasets will have tissue information
                    'height': crop_size,
                    'width': crop_size,
                    'original_image': str(image_path),  
                    'original_center': [center_x, center_y],
                    'original_bbox': bbox,
                    'original_segmentation': segmentation,
                    'is_padded': is_padded 
                })
                
                crop_id += 1
                
            except Exception as e:
                print(f"Error processing crop for {image_path}: {str(e)}")
                continue
    
    with open(output_annotations_path, 'w') as f:
        json.dump(annotations_output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cropped cell images from a dataset with COCO-style annotations.")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to the dataset root directory (containing annotations.json and images/).")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the directory where cropped images and annotations will be saved.")
    parser.add_argument("--crop_size", type=int, default=112,
                        help="Size of the square crop window in pixels (default: 112).")
    parser.add_argument("--min_distance_from_edge", type=int, default=56,
                        help="Minimum distance of the cell center from image edges (default: 56).")
    parser.add_argument("--min_area", type=int, default=150,
                        help="Minimum cell area to be considered (default: 150).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed processing information.")

    args = parser.parse_args()

    extract_cropped_images(args.input_folder, args.output_folder, args.crop_size,
                           args.min_distance_from_edge, args.min_area, args.verbose)