
#git clone https://github.com/radixark/miles /opt/tiger/miles

# install nccl
#NCCL_VERSION="2.27.5-1"
#cd /usr/local/src && git clone -b v${NCCL_VERSION} https://github.com/NVIDIA/nccl && cd nccl && \
#make -j src.build CUDA_HOME=/usr/local/cuda \
#NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100" && \
#make install


MEGATRON_PATH=/opt/tiger/Megatron-LM
git clone https://github.com/radixark/Megatron-LM ${MEGATRON_PATH}
cd $MEGATRON_PATH
pip install -e .

pip3 install typer ray httpx --user
pip3 install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps --user