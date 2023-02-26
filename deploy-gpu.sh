# 创建新的conda环境, 并切换
conda create --name signboard-ocr-test python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ -y
activate signboard-ocr-test
where python
# 安装paddlepaddle-gpu
conda install paddlepaddle-gpu==2.4.1 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge -y
# 安装python package
pip install -r requirements.txt
pip install straug
