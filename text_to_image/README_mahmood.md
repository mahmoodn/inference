Commands:
```
cd ../inference/text_to_image/
cp ../mlperf.conf .
```
Download model:
```
mkdir model
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp32 ./stable_diffusion_fp32 -P
rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp16 ./stable_diffusion_fp16 -P
mv stable_diffusion_fp* model/
```
Download dataset:
```
cd tools/
./download-coco-2014.sh -n 2
cd ../
```
Install packages in Conda env
```
module load miniconda3
conda activate pt24-cuda121
conda install conda-forge::opencv
conda install conda-forge::open-clip-torch
conda install conda-forge::diffusers
conda install scipy
```
Run command:
```
python3 main.py --dataset "coco-1024" --dataset-path coco2014 --profile stable-diffusion-xl-pytorch --model-path model/stable_diffusion_fp32 --dtype fp32 --device cuda --scenario Offline
```
