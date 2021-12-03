# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

A2D2_SEM_SEG_FULL_CATEGORIES = [
    {'name':'Car 1',                'id': 0     },
    {'name':'Bicycle 1',            'id': 1     },
    {'name':'Pedestrian 1',         'id': 2     },
    {'name':'Truck 1',              'id': 3     },
    {'name':'Small vehicles 1',     'id': 4     },
    {'name':'Traffic signal 1',     'id': 5     },
    {'name':'Traffic sign 1',       'id': 6     },
    {'name':'Utility vehicle 1',    'id': 7     },
    {'name':'Sidebars',             'id': 8     },
    {'name':'Speed bumper',         'id': 9     },
    {'name':'Curbstone',            'id': 10    },
    {'name':'Solid line',           'id': 11    },
    {'name':'Irrelevant signs',     'id': 12    },
    {'name':'Road blocks',          'id': 13    },
    {'name':'Tractor',              'id': 14    },
    {'name':'Non-drivable street',  'id': 15    },
    {'name':'Zebra crossing',       'id': 16    },
    {'name':'Obstacles / trash',    'id': 17    },
    {'name':'Poles',                'id': 18    },
    {'name':'RD restricted area',   'id': 19    },
    {'name':'Animals',              'id': 20    },
    {'name':'Grid structure',       'id': 21    },
    {'name':'Signal corpus',        'id': 22    },
    {'name':'Drivable cobblestone', 'id': 23    },
    {'name':'Electronic traffic',   'id': 24    },
    {'name':'Slow drive area',      'id': 25    },
    {'name':'Nature object',        'id': 26    },
    {'name':'Parking area',         'id': 27    },
    {'name':'Sidewalk',             'id': 28    },
    {'name':'Ego car',              'id': 29    },
    {'name':'Painted driv. instr.', 'id': 30    },
    {'name':'Traffic guide obj.',   'id': 31    },
    {'name':'Dashed line',          'id': 32    },
    {'name':'RD normal street',     'id': 33    },
    {'name':'Sky',                  'id': 34    },
    {'name':'Buildings',            'id': 35    },
    {'name':'Blurred area',         'id': 36    },
    {'name':'Rain dirt',            'id': 37    },
]


def _get_ade20k_full_meta():
    # Id 0 is reserved for ignore_label, we change ignore_label for 0
    # to 255 in our pre-processing, so all ids are shifted by 1.

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(range(38))}
    stuff_classes = [k["name"] for k in A2D2_SEM_SEG_FULL_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def load_sem_seg_a2d2(lst_path):
    img_list = [line.strip().split() for line in open(lst_path)]

    dataset_dicts = []
    for img_path, gt_path in img_list:
        record = {}
        record["file_name"] = img_path
        gt_path = gt_path[::-1].replace('label'[::-1], 'mask'[::-1], 2)[::-1]
        record["sem_seg_file_name"] = gt_path
        dataset_dicts.append(record)

    return dataset_dicts


def register_all_ade20k_full(root):
    root = os.path.join(root, "A2D2")
    meta = _get_ade20k_full_meta()
    for name, lst_name in [("train", "train.lst"), ("val", "val.lst")]:
        lst_file = os.path.join(root, "split", lst_name)
        name = f"a2d2_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda p=lst_file: load_sem_seg_a2d2(p)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            lst_path=lst_file,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 8-bit
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_ade20k_full(_root)
