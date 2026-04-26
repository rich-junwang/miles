
#git clone https://github.com/radixark/miles /opt/tiger/miles

# install nccl
sudo bash -c '
export NCCL_VERSION="2.27.5-1"
cd /usr/local/src && git clone -b v${NCCL_VERSION} https://github.com/NVIDIA/nccl && cd nccl && \
make -j src.build CUDA_HOME=/usr/local/cuda \
NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100" && \
make install
'


MEGATRON_REPO=radixark/Megatron-LM
MEGATRON_BRANCH=miles-main
MEGATRON_PATH=/opt/tiger/Megatron-LM
SGLANG_PATH=/opt/tiger/sglang

# install apex
NVCC_APPEND_FLAGS="--threads 4" \
pip3 -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
--config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4 --user


# install megatron-lm
# git clone https://github.com/radixark/Megatron-LM ${MEGATRON_PATH}
# cd $MEGATRON_PATH
# pip install -e . --user

git clone https://github.com/${MEGATRON_REPO}.git --recursive -b ${MEGATRON_BRANCH} ${MEGATRON_PATH} && \
    cd Megatron-LM && pip install -e . --user

# cuda 13.1
pip install nvidia-mathdx==25.6.0 --user && \
pip -v install --no-build-isolation "transformer_engine[core_cu13,pytorch]==2.10.0" --user


# install sglang
git clone -b sglang-miles https://github.com/sgl-project/sglang ${SGLANG_PATH} && cd ${SGLANG_PATH} && pip3 install -e "python[all]" --no-deps --user

# install mooncake
pip3 install mooncake-transfer-engine-cuda13 --user

#pip3 install -U "ray[all]"
pip3 install -U "ray[default]" --user
pip3 install typer httpx --user
pip3 install sglang --user
pip3 install sglang-router>=0.2.3 --user
pip3 SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps --user
pip3 install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps --user

pip3 install "protobuf<=3.20.3" --user
pip3 install "numpy<2" --user