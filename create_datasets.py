import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple

random.seed(42)

TRAIN_RATIO = 0.8
VEGGIES = ['Potato', 'Tomato', 'Pepper']

# Clean class name for better name like Tomato___Early_blight -> early_blight
def clean_class_name(class_name: str, veggie_type: str) -> str:
    name = class_name.lower()
    veggie_lower = veggie_type.lower()

    name = name.replace(f'{veggie_lower}___', '')
    name = name.replace(f'{veggie_lower}__', '')
    name = name.replace(f'{veggie_lower}_', '', 1)
    name = name.replace('___', '_').replace('__', '_')
    name = name.strip('_')

    return name

def get_veggie_type(class_name: str) -> str:
    for veggie in VEGGIES:
        if veggie.lower() in class_name.lower():
            return veggie
    return None

# returns {veggie_type: {class_name: [img_paths]}}
def collect_images(source_path: Path) -> Dict[str, Dict[str, List[Path]]]:
    data = defaultdict(lambda: defaultdict(list))

    subdirs = sorted([d for d in source_path.iterdir() if d.is_dir()])

    for subdir in subdirs:
        class_name = subdir.name
        veggie_type = get_veggie_type(class_name)

        if veggie_type is None:
            print(f"Warning: Could not identify vegetable type for {class_name}")
            continue

        images = sorted([
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        data[veggie_type][class_name] = images

    return data

def train_test_split(images: List[Path], train_ratio: float = TRAIN_RATIO) -> Tuple[List[Path], List[Path]]:
    images_copy = images.copy()
    random.shuffle(images_copy)

    split_idx = int(len(images_copy) * train_ratio)
    return images_copy[:split_idx], images_copy[split_idx:]

# reduces majority class to be at most (1/target_ratio) times the minority class
def undersample_balance(data: Dict[str, List[Path]], target_ratio: float = 0.5) -> Dict[str, List[Path]]:
    class_counts = {cls: len(imgs) for cls, imgs in data.items()}
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    target_max = int(min_count / target_ratio)

    balanced_data = {}
    for cls, imgs in data.items():
        if len(imgs) > target_max:
            balanced_data[cls] = random.sample(imgs, target_max)
        else:
            balanced_data[cls] = imgs

    return balanced_data

def copy_images(images: List[Path], dest_dir: Path, class_name: str, start_idx: int = 0) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(images, start=start_idx):
        dest_file = dest_dir / f"{class_name}.{idx}.jpg"
        try:
            shutil.copy2(img_path, dest_file)
        except Exception as e:
            print(f"Error copying {img_path}: {e}")

    return len(images)

def create_binary_per_veggie(data: Dict[str, Dict[str, List[Path]]], output_path: Path):
    for veggie_type, classes in data.items():
        print(f"Creating binary dataset for {veggie_type}...")

        healthy_images = []
        unhealthy_images = []

        for class_name, images in classes.items():
            if 'healthy' in class_name.lower():
                healthy_images.extend(images)
            else:
                unhealthy_images.extend(images)

        normal_dir = output_path / f"{veggie_type}_binary"
        create_binary_dataset(
            {'healthy': healthy_images, 'unhealthy': unhealthy_images},
            normal_dir,
            f"{veggie_type} Binary"
        )

        balanced_dir = output_path / f"{veggie_type}_binary_balanced"
        balanced_data = undersample_balance(
            {'healthy': healthy_images, 'unhealthy': unhealthy_images},
            target_ratio=0.4
        )
        create_binary_dataset(
            balanced_data,
            balanced_dir,
            f"{veggie_type} Binary Balanced"
        )

def create_binary_dataset(data: Dict[str, List[Path]], output_dir: Path, dataset_name: str):
    print(f"Creating {dataset_name}...")

    stats = {'train': {}, 'test': {}}

    for class_name, images in data.items():
        train_imgs, test_imgs = train_test_split(images)
        train_dir = output_dir / 'train' / class_name
        train_count = copy_images(train_imgs, train_dir, class_name)
        stats['train'][class_name] = train_count

        test_dir = output_dir / 'test' / class_name
        test_count = copy_images(test_imgs, test_dir, class_name)
        stats['test'][class_name] = test_count

def create_multiclass_per_veggie(data: Dict[str, Dict[str, List[Path]]], output_path: Path):
    for veggie_type, classes in data.items():
        print(f"Creating mutli-class dataset {veggie_type}...")

        cleaned_data = {}
        for class_name, images in classes.items():
            clean_name = clean_class_name(class_name, veggie_type)
            cleaned_data[clean_name] = images

        normal_dir = output_path / f"{veggie_type}_multiclass"
        create_multiclass_dataset(
            cleaned_data,
            normal_dir,
            f"{veggie_type} Multiclass"
        )

        balanced_dir = output_path / f"{veggie_type}_multiclass_balanced"
        balanced_data = undersample_balance(
            cleaned_data,
            target_ratio=0.2  # More lenient for multiclass
        )
        create_multiclass_dataset(
            balanced_data,
            balanced_dir,
            f"{veggie_type} Multiclass Balanced"
        )

def create_multiclass_dataset(data: Dict[str, List[Path]], output_dir: Path, dataset_name: str):
    print(f"Creating {dataset_name}...")

    stats = {'train': {}, 'test': {}}

    for class_name, images in data.items():
        train_imgs, test_imgs = train_test_split(images)
        
        train_dir = output_dir / 'train' / class_name
        train_count = copy_images(train_imgs, train_dir, class_name)
        stats['train'][class_name] = train_count

        test_dir = output_dir / 'test' / class_name
        test_count = copy_images(test_imgs, test_dir, class_name)
        stats['test'][class_name] = test_count

def create_binary_all_veggies(data: Dict[str, Dict[str, List[Path]]], output_path: Path):
    healthy_images = []
    unhealthy_images = []

    for veggie_type, classes in data.items():
        for class_name, images in classes.items():
            if 'healthy' in class_name.lower():
                healthy_images.extend(images)
            else:
                unhealthy_images.extend(images)

    normal_dir = output_path / "all_veggies_binary"
    create_binary_dataset(
        {'healthy': healthy_images, 'unhealthy': unhealthy_images},
        normal_dir,
        "All Veggies Binary"
    )

    balanced_dir = output_path / "all_veggies_binary_balanced"
    balanced_data = undersample_balance(
        {'healthy': healthy_images, 'unhealthy': unhealthy_images},
        target_ratio=0.4
    )
    create_binary_dataset(
        balanced_data,
        balanced_dir,
        "All Veggies Binary Balanced"
    )

def print_summary(data: Dict[str, Dict[str, List[Path]]]):
    for veggie_type, classes in data.items():
        print(f"\n{veggie_type}:")
        total = 0
        for class_name, images in sorted(classes.items()):
            clean_name = clean_class_name(class_name, veggie_type)
            count = len(images)
            total += count
            print(f"{clean_name}: {count} images")
        print(f"{'Total'}: {total} images")

def main():
    source_dir = Path("/Users/nurmybtw/Desktop/uni/iot/project/PlantVillage")
    output_dir = Path("/Users/nurmybtw/Desktop/uni/iot/project/PlantVillage_Datasets")

    data = collect_images(source_dir)
    print_summary(data)
    output_dir.mkdir(exist_ok=True)
    create_binary_per_veggie(data, output_dir)
    create_multiclass_per_veggie(data, output_dir)
    create_binary_all_veggies(data, output_dir)

if __name__ == "__main__":
    main()
