Commands:
```
export ROOT=/scratch/mnaderantahan/inference
export LLAMA_FOLDER=$ROOT/language/llama3.1-8b
export LOADGEN_FOLDER=$ROOT/loadgen
export DATASET_PATH=$LLAMA_FOLDER/dataset/cnn_eval.json
export CHECKPOINT_PATH=$LLAMA_FOLDER/Llama-3.1-8B-Instruct
export GPU_COUNT=1

module load miniconda3/4.12.0
conda create --name pt29-cuda128 python=3.10
conda activate pt29-cuda128
cd $LLAMA_FOLDER
pip3 install -r requirements.txt
cd $LOADGEN_FOLDER
pip3 install -e .
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz
tar -xvzf git-lfs-linux-amd64-v3.4.1.tar.gz
export PATH=~/git-lfs-3.4.1:$PATH

git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout be673f326cab4cd22ccfef76109faf68e41aa5f1
mkdir dataset && cd dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri

python -u main.py --scenario Offline --model-path ${CHECKPOINT_PATH} --batch-size 16 --dtype bfloat16 --user-conf user.conf --total-sample-count 13368 --dataset-path ${DATASET_PATH} --output-log-dir output --tensor-parallel-size ${GPU_COUNT} --vllm
```

inteactive job command:
```
srun --mpi=pmix --job-name="int_gpu_job" --partition=gpu-a100-small --time=01:00:00 --ntasks=1 --cpus-per-task=2 --gpus-per-task=1 --mem-per-cpu=5G --account=research-eemcs-qce --pty /bin/bash -il

/scratch/mnaderantahan/nsight-systems-2025.5.1/bin/nsys profile --output nsys.out --trace=cuda,cublas,cudnn,osrt,nvtx --sample cpu --cpuctxsw process-tree python -u main.py --scenario Offline --model-path $CHECKPOINT_PATH --batch-size $BATCH_SIZE --dtype bfloat16 --user-conf user.conf --total-sample-count 1 --dataset-path $DATASET_PATH --output-log-dir output --tensor-parallel-size $GPU_COUNT --vllm
```
