import argparse
from pathlib import Path
import cv2
import os
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--data_path', type=str, default=None, help='specify the path of PV images')
parser.add_argument('--output_path', type=str, default=None,
                    help='specify the output path of cells that splitted from PV images')
args = parser.parse_args()
ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()


def cell_crop(img, img_id, output_path):
    for r in range(6):
        for c in range(22):
            y = 8 + (r*600)

            if c < 11:
                x = 10 + (c*300)
            else:
                x = 12 + (c*300)

            image = img[y:y+600, x:x+300]
            cv2.imwrite(output_path / f"{img_id}_{r+1}_{c+1}.jpg", image)
            # image_ = Image.fromarray(image)
            # image_.save(output_path / f"{img_id}_{r+1}_{c+1}.jpg")


def main():
    path_data = ROOT_DIR / args.data_path
    folder = os.listdir(path_data)
    k = 1
    print("*************** PreParing data for model by split PV image to 132 cells ***************")
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    for img_p in tqdm(folder):
        path = os.path.join(path_data, img_p)
        img = cv2.imread(path, 0)
        img_id = img_p.split(".")[0]
        # print(img_id)
        cell_crop(img, img_id, output_path=Path(args.output_path))
        # print(k)
        k += 1
    print("*************** PreParing data done ***************")


if __name__ == '__main__':
    main()
