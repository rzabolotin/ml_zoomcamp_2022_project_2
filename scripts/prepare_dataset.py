import os
import random
from path import Path
import shutil

from loguru import logger

PATH_TO_DATA = Path('data')
CATS_FOLDER = PATH_TO_DATA / 'cats'

NO_CATS_SOURCE_FOLDER = PATH_TO_DATA / "flats"
NO_CATS_DEST_FOLDER = CATS_FOLDER / "No cat"
FILES_LIMIT = 995

os.makedirs(CATS_FOLDER, exist_ok=True)
os.makedirs(NO_CATS_DEST_FOLDER, exist_ok=True)


def copy_files_recursively(source_folder, dest_folder):
    logger.info(f"Starting moving files from {source_folder}")
    for dir, _, files in os.walk(source_folder):
        if files:
            logger.info(f"Moving {len(files)} files from {dir}")
            for file in files:
                shutil.move(os.path.join(dir, file), dest_folder)


def shrink_dataset_recursively(root_folder, files_limit):
    logger.info("Shrinking dataset")
    sub_folders = os.listdir(root_folder)
    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(root_folder, sub_folder))
        if len(files) > files_limit:
            logger.info(f"Shrinking {sub_folder} folder with {len(files)} files")
            random.shuffle(files)
            for file in files[files_limit:]:
                os.remove(os.path.join(root_folder, sub_folder, file))


def print_stats(root_folder):
    logger.info(f"Count of files in {root_folder}")
    files_count = {}
    sub_folders = os.listdir(root_folder)
    for sub_folder in sub_folders:
        files_count[sub_folder] = len(os.listdir(os.path.join(root_folder, sub_folder)))

    for key, value in sorted(files_count.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"\t{key}:\t {value}")


def main():
    logger.info("Preparing dataset")
    logger.info(f"  ROOT DIR: {PATH_TO_DATA}")
    logger.info(f"  DEST DIR: {CATS_FOLDER}")
    logger.info(f"  NO CATS SOURCE DIR: {NO_CATS_SOURCE_FOLDER}")
    logger.info(f"  NO CATS DEST DIR: {NO_CATS_DEST_FOLDER}")

    copy_files_recursively(NO_CATS_SOURCE_FOLDER, NO_CATS_DEST_FOLDER)

    shrink_dataset_recursively(CATS_FOLDER, FILES_LIMIT)

    print_stats(CATS_FOLDER)

    logger.success("Dataset prepared")


if __name__ == "__main__":
    main()
