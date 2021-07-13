pip install --upgrade pip
pip install wechaty==0.7dev17
hub install lac==2.2.0

# 设置环境变量
export WECHATY_PUPPET=wechaty-puppet-service
export WECHATY_PUPPET_SERVICE_TOKEN=puppet_paimon_53e0aca9929073bdb4e69c44ce1f85e6

# 设置使用GPU进行模型预测
export CUDA_VISIBLE_DEVICES=0

# 创建两个保存图片的文件夹
mkdir -p image
mkdir -p video

# 运行python文件 
python run.py