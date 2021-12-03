## MaskFormer Demo

We provide a command line tool to run a simple demo of builtin configs.
The usage is explained in [GETTING_STARTED.md](../GETTING_STARTED.md).

```
python demo.py --config-file ../configs/a2d2-38/swin/maskformer_swin_small_bs16_160k.yaml \
  --input imgs/* \
  --output inf \
  --opts MODEL.WEIGHTS ../output_a2d2_swin-s/model_final.pth

python demo.py --config-file ../configs/a2d2-38/swin/maskformer_swin_small_bs16_160k.yaml \
  --input /nfs/data/KITTI/detection3d/testing/image_2/* \
  --output /nfs/data/KITTI/detection3d/testing/image_2_inf \
  --opts MODEL.WEIGHTS ../output_a2d2_swin-s/model_final.pth

python demo.py --config-file ../configs/a2d2-38/swin/maskformer_swin_small_bs16_160k.yaml \
  --input /nfs/data/KITTI/detection3d/testing/prev_2/* \
  --output /nfs/data/KITTI/detection3d/testing/prev_2_inf \
  --opts MODEL.WEIGHTS ../output_a2d2_swin-s/model_final.pth
```
