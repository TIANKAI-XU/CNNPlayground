import random
from pathlib import Path
from shutil import copy2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'data'
CAT_DOG_ROOT = DATA_ROOT / 'CatDog'
RAW_DATA_DIR = CAT_DOG_ROOT / 'raw'
LEGACY_RAW_DATA_DIR = SCRIPT_DIR / 'data_cat_dog'
TRAIN_DIR = CAT_DOG_ROOT / 'train'
TEST_DIR = CAT_DOG_ROOT / 'test'
SPLIT_RATE = 0.1
RANDOM_SEED = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_raw_data_dir() -> Path:
    if RAW_DATA_DIR.exists():
        return RAW_DATA_DIR
    if LEGACY_RAW_DATA_DIR.exists():
        return LEGACY_RAW_DATA_DIR
    raise FileNotFoundError(
        'Cat-vs-dog raw dataset not found. Please place the class folders under '
        f'"{RAW_DATA_DIR}", or continue using the legacy directory "{LEGACY_RAW_DATA_DIR}".'
    )


def main() -> None:
    random.seed(RANDOM_SEED)
    raw_data_dir = resolve_raw_data_dir()
    class_names = sorted([item.name for item in raw_data_dir.iterdir() if item.is_dir()])

    if not class_names:
        raise ValueError(f'No class directories were found under "{raw_data_dir}".')

    for dataset_dir in (TRAIN_DIR, TEST_DIR):
        ensure_dir(dataset_dir)
        for class_name in class_names:
            ensure_dir(dataset_dir / class_name)

    for class_name in class_names:
        class_dir = raw_data_dir / class_name
        images = sorted([item for item in class_dir.iterdir() if item.is_file()])
        image_count = len(images)

        eval_images = set(random.sample(images, k=int(image_count * SPLIT_RATE)))
        for index, image_path in enumerate(images, start=1):
            target_dir = TEST_DIR if image_path in eval_images else TRAIN_DIR
            copy2(image_path, target_dir / class_name / image_path.name)
            print(f'\r[{class_name}] processing [{index}/{image_count}]', end='')
        print()

    print(f'processing done! train/test data saved to: {CAT_DOG_ROOT}')


if __name__ == '__main__':
    main()
