import os
import numpy as np
import torch
from PIL import Image
from detectron2.structures import BoxMode
from pycocotools import mask as pyco
from detectron2.data import DatasetCatalog, MetadataCatalog

__all__ = ["load_penn_instances", "register_pennfudan"]

CLASS_NAMES = (
    "person",
)
def load_penn_instances(dirname: str, split: str):

    annotation_names= list(sorted(os.listdir(os.path.join(dirname, "PedMasks"))))
    img_names = list(sorted(os.listdir(os.path.join(dirname, "PNGImages"))))
    fileids = len(img_names)

    dicts = []
    for fileid in range(fileids):

        anno_file = os.path.join(dirname, "PedMasks", annotation_names[fileid])
        jpeg_file = os.path.join(dirname, "PNGImages", img_names[fileid])
        img = Image.open(jpeg_file).convert("RGB")
        w, h = img.size

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": h,
            "width": w,
        }
        mask = Image.open(anno_file)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        # boxes = []
        instances = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bbox = [xmin, ymin, xmax, ymax]
            instance_seg = pyco.encode(np.asfortranarray(masks[i]))

            instances.append(
                {"category_id": 0, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS, "segmentation":instance_seg}
            )

        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_pennfudan(name, dirname, split, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_penn_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, split=split
    )

# class PennFudanDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transforms):
#         self.root = root
#         self.transforms = transforms
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
#
#     def __getitem__(self, idx):
#         # load images and masks
#         img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
#         mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
#         img = Image.open(img_path).convert("RGB")
#         # note that we haven't converted the mask to RGB,
#         # because each color corresponds to a different instance
#         # with 0 being background
#         mask = Image.open(mask_path)
#         # convert the PIL Image into a numpy array
#         mask = np.array(mask)
#         # instances are encoded as different colors
#         obj_ids = np.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#
#         # split the color-encoded mask into a set
#         # of binary masks
#         masks = mask == obj_ids[:, None, None]
#
#         # get bounding box coordinates for each mask
#         num_objs = len(obj_ids)
#         boxes = []
#         for i in range(num_objs):
#             pos = np.where(masks[i])
#             xmin = np.min(pos[1])
#             xmax = np.max(pos[1])
#             ymin = np.min(pos[0])
#             ymax = np.max(pos[0])
#             boxes.append([xmin, ymin, xmax, ymax])
#
#         # convert everything into a torch.Tensor
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
#         masks = torch.as_tensor(masks, dtype=torch.uint8)
#
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
#
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#
#         # if self.transforms is not None:
#         #     img, target = self.transforms(img, target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)