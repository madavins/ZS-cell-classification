import json
import random
import time
import argparse
import os
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from openai_api.description_generator import CellDescriptionGenerator 


class DatasetProcessor:

    def __init__(self, description_generator: CellDescriptionGenerator):
        self.description_generator = description_generator

    def select_balanced_subset(self, annotations: List[Dict[str, Any]], num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Selects a balanced subset of image annotations from the PanNuke dataset.

        This method aims to create a subset that represents the distribution
        of cell types in the original PanNuke dataset as closely as possible."""

        cell_type_groups = {}
        for ann in annotations:
            cell_type_groups.setdefault(ann['cell_type'], []).append(ann)

        num_cell_types = len(cell_type_groups)
        base_samples_per_type = num_samples // num_cell_types

        selected_annotations = []
        for cell_annotations in cell_type_groups.values():
            num_to_select = min(base_samples_per_type, len(cell_annotations))
            selected_annotations.extend(random.sample(cell_annotations, num_to_select))

        remaining = num_samples - len(selected_annotations)
        if remaining > 0:
            remaining_pool = [ann for ann in annotations if ann not in selected_annotations]
            if remaining_pool:
                selected_annotations.extend(random.sample(remaining_pool, min(remaining, len(remaining_pool))))
        return selected_annotations

    def generate_balanced_subset(self, input_annotations_path: str, output_subset_path: str, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generates and saves (or loads) a balanced subset of image samples.
        
        The function checks if a subset file already exists.  If it does, the subset is loaded.
        If not, it loads the full dataset, calls `select_balanced_subset` to create the subset, and saves it to the specified output path.  
        This avoids redundant computation and provides a consistent way to access the balanced subset."""

        input_path = Path(input_annotations_path)
        output_path = Path(output_subset_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            print(f"Loading existing subset from {output_path}")
            with open(output_path, 'r') as f:
                subset_data = json.load(f)
                selected_annotations = subset_data['annotations']
        else:
            print(f"Generating new subset of {num_samples} samples...")
            with open(input_path, 'r') as f:
                original_data = json.load(f)

            selected_annotations = self.select_balanced_subset(original_data['annotations'], num_samples)
            subset_data = {'annotations': selected_annotations}

            with open(output_path, 'w') as f:
                json.dump(subset_data, f, indent=2)
            print(f"Created new subset at {output_path}")
        
        self.print_subset_analysis(selected_annotations) #call print analysis function
        return selected_annotations

    def print_subset_analysis(self, selected_annotations):
        """Prints the analysis of cell type and tissue distribution."""
        print(f"\nSubset contains {len(selected_annotations)} images")
        cell_type_counts = Counter(ann['cell_type'] for ann in selected_annotations)
        tissue_counts = Counter(ann['tissue'] for ann in selected_annotations)

        print("\nCell type distribution:")
        for cell_type, count in cell_type_counts.items():
            print(f"{cell_type}: {count}")

        print("\nTissue distribution:")
        for tissue, count in tissue_counts.items():
            print(f"{tissue}: {count}")

    def add_descriptions(self, subset_annotations: List[Dict[str, Any]], output_final_path: str,
                         images_dir: str, calls_per_minute: int = 80, log_frequency: int = 10) -> None:
        """Adds descriptions to annotations, with rate limiting and intermediate saves."""

        delay = 60.0 / calls_per_minute
        output_path = Path(output_final_path)
        output_path.parent.mkdir(parents=True, exist_ok=True) 
        total_images = len(subset_annotations)
        annotations_with_descriptions = []

        print(f"Starting processing of {total_images} images with {delay:.2f}s delay between calls")
        print("=" * 50)

        try:
            for i, ann in enumerate(subset_annotations, 1):
                ann_with_desc = ann.copy()
                image_path = Path(images_dir) / ann['file_name']

                try:
                    ann_with_desc['description'] = self.description_generator.get_cell_description(
                        image_path=str(image_path),
                        cell_type=ann['cell_type'],
                        tissue_type=ann['tissue']
                    )
                except Exception as e:
                    print(f"\nError processing image {i}/{total_images}: {e}")
                    continue

                annotations_with_descriptions.append(ann_with_desc)

                if i % log_frequency == 0:
                    print(f"Processed {i}/{total_images} images ({(i / total_images) * 100:.1f}%)")

                if i % 50 == 0:
                    self._save_intermediate_results(annotations_with_descriptions, output_path, i)

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Saving partial results...")
            self._save_partial_results(annotations_with_descriptions, output_path)
            return
        except Exception as e:
            print(f"\nUnexpected error: {e}. Saving partial results...")
            self._save_partial_results(annotations_with_descriptions, output_path)
            raise

        self._save_final_results(annotations_with_descriptions, output_path)

    def _save_intermediate_results(self, annotations: List[Dict[str, Any]], output_path: Path, index: int) -> None:
        """Saves intermediate results."""
        intermediate_path = output_path.parent / f"intermediate_{index}_annotations.json"
        with open(intermediate_path, 'w') as f:
            json.dump({'annotations': annotations}, f, indent=2)
        print(f"\nSaved intermediate results to {intermediate_path}")
        print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

    def _save_partial_results(self, annotations: List[Dict[str, Any]], output_path: Path) -> None:
        """Saves partial results on interruption."""
        partial_path = output_path.parent / f"partial_annotations_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(partial_path, 'w') as f:
            json.dump({'annotations': annotations}, f, indent=2)
        print(f"Partial results ({len(annotations)} images) saved to {partial_path}")

    def _save_final_results(self, annotations: List[Dict[str, Any]], output_path: Path) -> None:
        """Saves final results."""
        with open(output_path, 'w') as f:
            json.dump({'annotations': annotations}, f, indent=2)
        print(f"\nSuccessfully processed {len(annotations)} images")
        print(f"Final results saved to {output_path}")
        print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    parser = argparse.ArgumentParser(description="Generate a dataset of cell images with descriptions.")
    parser.add_argument("input_annotations", help="Path to the original annotations JSON file.")
    parser.add_argument("images_dir", help="Path to the directory containing the cell images.")
    parser.add_argument("output_subset", help="Path to save the balanced subset annotations JSON file.")
    parser.add_argument("output_final", help="Path to save the final annotations with descriptions JSON file.")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples for the balanced subset (default: 1000).")
    parser.add_argument("--calls_per_minute", type=int, default=80, help="Maximum API calls per minute (default: 80).")
    parser.add_argument("--log_frequency", type=int, default=10, help="Log progress every N images (default: 10).")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("The OPENAI_API_KEY environment variable is not set.")

    description_generator = CellDescriptionGenerator(api_key)
    dataset_processor = DatasetProcessor(description_generator)

    # 1. Generate (or load) balanced subset
    subset_annotations = dataset_processor.generate_balanced_subset(
        args.input_annotations, args.output_subset, args.num_samples
    )

    # 2. Add descriptions to the subset
    dataset_processor.add_descriptions(
        subset_annotations, args.output_final, args.images_dir, args.calls_per_minute, args.log_frequency
    )

if __name__ == "__main__":
    main()