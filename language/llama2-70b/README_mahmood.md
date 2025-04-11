Download model:
```
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar -xvzf git-lfs-linux-amd64-v3.4.1.tar.gz
export PATH=~/git-lfs-3.4.1:$PATH
cd $SCRATCH
export CHECKPOINT_PATH=$SCRATCH/Llama-2-70b-chat-hf
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf ${CHECKPOINT_PATH}

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
conda install pandas=2.1.4 numpy=1.23.5
conda install sympy

# install pytorch
pip3 install torch torchvision torchaudio

# build loadgen
cd ~/inference/loadgen
conda install conda-build
conda install absl-py
CFLAGS="-std=c++14 -O3" python -m pip install --user .

# install packages for llama run
cd ~/inference/language/llama2-70b

conda install conda-forge::transformers
conda install conda-forge::sentencepiece
conda install anaconda::protobuf
conda install -c conda-forge accelerate
pip install transformers -U
pip install accelerate
python3 -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH}  \
  --user-conf user.conf --total-sample-count 24576 --dataset-path ${DATASET_PATH} \
  --output-log-dir offline-logs --dtype float32 --device cuda:0 2>&1 | tee offline_performance_log.log
```
