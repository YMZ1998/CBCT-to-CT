import os
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from parse_args import remove_and_create_dir, parse_args


def copy_files(file_list, origin_path, dest_path, item):
    for file in tqdm(file_list, desc=f"Copying {item} files"):
        src_path = os.path.join(origin_path, file)
        dst_path = os.path.join(dest_path, file)

        try:
            shutil.copytree(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {file}: {e}")


if __name__ == '__main__':
    args = parse_args()

    data_path = os.path.join(r'D:\Data\SynthRAD\Task2', args.anatomy)
    result_path = os.path.join(r'D:\Python_code\CBCT-to-CT\data', args.anatomy)

    remove_and_create_dir(result_path)

    train_path = os.path.join(result_path, 'train')
    test_path = os.path.join(result_path, 'test')

    paths = os.listdir(data_path)[:20]
    print(f"Total files: {len(paths)}")

    train, test = train_test_split(paths, test_size=0.2, random_state=42)
    print(f"Train files: {len(train)}")
    print(f"Test files: {len(test)}")

    copy_files(train, data_path, train_path, 'train')
    copy_files(test, data_path, test_path, 'test')
