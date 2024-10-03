Download model:
```
cd $SCRATCH
export CHECKPOINT_PATH=$SCRATCH/Llama-2-70b-chat-hf
git clone https://USER:TOKEN@huggingface.co/meta-llama/Llama-2-70b-chat-hf ${CHECKPOINT_PATH}
```

Download Dataset:
```
cd $SCRATCH
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P
cd open_orca/
gzip -d reference_impl_gpu_bs32_fp32_output.pkl.gz
gzip -d open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
gzip -d open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz
export DATASET_PATH=$SCRATCH/open_orca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl
```

Create conda environment, install Pytorch, build Loadgen and run benchmark:
```
# create an env
conda create --name pt24-cuda121
conda activate pt24-cuda121

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# build loadgen
cd ~/inference/loadgen
conda install conda-build
conda install absl-py
conda install numpy
CFLAGS="-std=c++14 -O3" python -m pip install --user .

# install packages for llama run
cd ~/inference/language/llama2-70b
conda install conda-forge::transformers
conda install conda-forge::sentencepiece
conda install anaconda::protobuf
conda install -c conda-forge accelerate
python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --mlperf-conf mlperf.conf \
  --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} \
  --output-log-dir offline-logs --dtype float32 --device cuda:0 2>&1 | tee offline_performance_log.log
```
