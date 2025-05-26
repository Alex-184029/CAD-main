# -- labelme格式到掩膜
import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
from labelme.logger import logger
from labelme import utils
import glob
import numpy as np
 
def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It will handle multiple JSON files to generate a "
        "real-use dataset."
    )
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=False, type=str, default=r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\labels')
    parser.add_argument("-o", "--out", required=False, type=str, default=r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\masks')
    parser.add_argument("-v", "--visualize", required=False, type=str, default=r'E:\School\Grad1\CAD\Datasets\DwgFiles\WallLineData\dataset3\dataset33\dataset-onlywall\data-origin\create_mask\masks_visualize')
    args = parser.parse_args()
 
    json_dir = args.json_dir
    output_dir = args.out
    visualize_dir = args.visualize
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)
 
    if osp.isfile(json_dir):
        json_list = [json_dir] if json_dir.endswith('.json') else []
    else:
        json_list = glob.glob(os.path.join(json_dir, '*.json'))
 
    num = len(json_list)
    for i, json_file in enumerate(json_list):
        if i % 200 == 0:
            print('%d / %d' % (i, num))
        # logger.info(f"Processing file: {json_file}")
        # json_name = osp.basename(json_file).split('.')[0]
        json_name = osp.splitext(osp.basename(json_file))[0]
        # out_dir = osp.join(output_dir, json_name)     # 注意这里没有后缀
 
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file}: {e}")
            continue  # Skip to the next file
 
        imageData = data.get("imageData")

        if imageData is None:
            imagePath = osp.join(args.json_dir, data["imagePath"])
            try:
                with open(imagePath, "rb") as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode("utf-8")
            except FileNotFoundError:
                print(f"File not found: {imagePath}")
                continue  # Skip to the next JSON file
            except Exception as e:
                print(f"Error reading image file {imagePath}: {e}")
                continue
 
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {"_background_": 0, "WallArea1": 1}
        # for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        #     label_name = shape["label"]
        #     if label_name in label_name_to_value:
        #         label_value = label_name_to_value[label_name]
        #     else:
        #         label_value = len(label_name_to_value)
        #         label_name_to_value[label_name] = label_value
        
        lbl, _ = utils.shapes_to_label(img.shape, data["shapes"], label_name_to_value)

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        lbl_viz = imgviz.label2rgb(lbl, imgviz.asgray(img), label_names=label_names, loc="rb")

        # Save files to corresponding subdirectory
        # PIL.Image.fromarray(img).save(osp.join(out_dir, "img.png"))

        binary_mask = (lbl > 0).astype(np.uint8)
        # mask_image = PIL.Image.fromarray(binary_mask * 1)
        mask_image = PIL.Image.fromarray(binary_mask * 255)
        mask_image.save(osp.join(output_dir, json_name + '.png'))

        # utils.lblsave(osp.join(output_dir, json_name + '.png'), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(visualize_dir, json_name + '.png'))

        # with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        #     for lbl_name in label_names:
        #         f.write(str(lbl_name if lbl_name is not None else "unknown") + "\n")

        # yaml_data = {
        #     "label_name_to_value": label_name_to_value,
        #     "label_names": label_names
        # 
        # with open(osp.join(out_dir, "labels.yaml"), "w") as yaml_file:
        #     yaml.dump(yaml_data, yaml_file)

        # logger.info(f"Saved to: {out_dir}")

 
if __name__ == "__main__":
    main()