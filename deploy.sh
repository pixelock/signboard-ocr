# 创建新的conda环境, 并切换
conda create --name signboard-ocr-test python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
activate signboard-ocr-test
where python
# 安装paddlepaddle-cpu
conda install paddlepaddle==2.4.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -y
# 安装python package
pip install -r requirements.txt
pip install straug
