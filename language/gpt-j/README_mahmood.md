On cluster:
```
module load miniconda3
conda activate pt24-cuda121
conda install conda-forge::datasets
conda install conda-forge::simplejson

```
Download dataset:
```
cd inference
cd language/gpt-j/
conda instal  
python download_cnndm.py
python prepare-calibration.py --calibration-list-file calibration-list.txt --output-dir ./calibrated
```

Download model:
```
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/gpt-j ./model -P

```

Run inference:
```
python main.py --scenario=Offline --model-path=./model/checkpoint-final/ --dataset-path=./data/cnn_eval.json  --gpu
```
