import os
import cv2
import asyncio
import numpy as np
import paddlehub as hub
import random
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

from wechaty import (
    Contact,
    FileBox,
    Message,
    Wechaty,
    ScanStatus,
)

from wechaty_puppet import MessageType

robot_state = 0
target_output = 'Unknown'
target = 'Unknown'

################### 1 EasyDL自定义数据集手势识别（基于EasyDL图像分类训练） ###################
import json
import base64
import requests

# 配置参数
# 目标图片的 本地文件路径，支持jpg/png/bmp格式
IMAGE_FILEPATH = ""
# top_num: 返回的分类数量，不声明的话默认为 6 个
PARAMS = {"top_num": 2}
# 服务详情 中的 接口地址
MODEL_API_URL = "Your EasyDL API URL"
#该模型部署的 API_KEY 以及 SECRET_KEY
API_KEY = 'Your API_KEY'
SECRET_KEY = 'Your SECRET_KEY'
# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + API_KEY + '&client_secret='+SECRET_KEY

def getAccessToken(host_url):
    access_token = ''
    response = requests.get(host_url)
    if response:
        print(response.json())
        access_token = response.json()['access_token']
    print(access_token)
    return access_token

def getUserInputJson(IMAGE_FILEPATH):
    ACCESS_TOKEN = getAccessToken(host)
    print("1. 读取目标图片 '{}'".format(IMAGE_FILEPATH))
    with open(IMAGE_FILEPATH, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        base64_str = base64_data.decode('UTF8')
    print("将 BASE64 编码后图片的字符串填入 PARAMS 的 'image' 字段")
    PARAMS["image"] = base64_str
    
    if not ACCESS_TOKEN:
        print("2. ACCESS_TOKEN 为空，调用鉴权接口获取TOKEN")
        auth_url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"\
                   "&client_id={}&client_secret={}".format(API_KEY, SECRET_KEY)
        auth_resp = requests.get(auth_url)
        auth_resp_json = auth_resp.json()
        ACCESS_TOKEN = auth_resp_json["access_token"]
        print("新 ACCESS_TOKEN: {}".format(ACCESS_TOKEN))
    else:
        print("2. 使用已有 ACCESS_TOKEN")
    
    print("3. 向模型接口 'MODEL_API_URL' 发送请求")
    request_url = "{}?access_token={}".format(MODEL_API_URL, ACCESS_TOKEN)
    response = requests.post(url=request_url, json=PARAMS)
    response_json = response.json()
    response_str = json.dumps(response_json, indent=4, ensure_ascii=False)
    print("结果:\n{}".format(response_str))
    
    return response_json

################### 2 星际分歧终端机游戏规则 （基于《生活大爆炸》谢耳朵的游戏） ###################

def who_win(input1,input2):
    if input1 == "scissors":
        if input2 == 'rock':
            return 2
        elif input2 == 'spock':
            return 2
        elif input2 == 'paper':
            return 1
        elif input2 == 'lizard':
            return 1
        else:
            return 0
    elif input1 == "rock":   
        if input2 == 'paper':
            return 2
        elif input2 == 'spock':
            return 2
        elif input2 == 'scissors':
            return 1
        elif input2 == 'lizard':
            return 1
        else:
            return 0
    elif input1 == "paper":  
        if input2 == 'scissors':
            return 2
        elif input2 == 'lizard':
            return 2
        elif input2 == 'spock':
            return 1
        elif input2 == 'rock':
            return 1
        else:
            return 0
    elif input1 == "lizard":  
        if input2 == 'rock':
            return 2
        elif input2 == 'scissors':
            return 2
        elif input2 == 'paper':
            return 1
        elif input2 == 'spock':
            return 1
        else:
            return 0
    elif input1 == "spock":  
        if input2 == 'paper':
            return 2
        elif input2 == 'lizard':
            return 2
        elif input2 == 'rock':
            return 1
        elif input2 == 'scissors':
            return 1
        else:
            return 0
################### 3 获取纠纷对象 （基于lac词法分析）###################

lac = hub.Module(name="lac")

def getTarget(text):
    test_text = []
    test_text.append(text)
    results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
    print(results)
    for result in results:
        print(result['word'])
        print(result['tag'])
    target = ''
    for i in range(len(result['word'])):
        if result['tag'][i]=='ORG' or result['tag'][i]=='nt' :
            target = result['word'][i] + "的代表"
            return target
        elif result['tag'][i]=='LOC' or result['tag'][i]=='ns' :
            target = result['word'][i] + "的代表"
            return target
        elif result['tag'][i]=='PER' or result['tag'][i]=='nr' :
            target = result['word'][i]
            return target
        elif result['tag'][i]=='n':
            target = result['word'][i]
            while i + 1 < len(result['word']) and result['tag'][i+1]=='n':
                target = target + result['word'][i+1]
                i = i + 1
            target = target + "的代表"
            return target

################### 4 组合双方出拳图片并显示结果 ###################
def image_compose(first_image, second_image, target_input, your_input, who_win):
    final_image_path = './image/final_image.png'
    #改变图像大小
    size = 500,250
    first_image.thumbnail(size)
    second_image.thumbnail(size)
    print (first_image.size, first_image.mode)
    print (second_image.size, second_image.mode)
    
    bia = 40

    final_image = Image.new('RGB', (2 * 210 + bia*2 , 1* 250 + bia*2), (0, 166, 255)) #创建一个新图
    # 图片粘贴到对应位置上
    final_image.paste(first_image, (0+ bia, 0+ bia))
    final_image.paste(second_image, (233+ bia, 0+ bia))
    final_image.save(final_image_path) # 保存新图

    lena = mpimg.imread(final_image_path) 
    plt.imshow(lena)
    plt.text(198+ bia,150+ bia,'VS',fontsize=15)
    plt.text(25+ bia,275+ bia,target_input,fontsize=15)
    plt.text(258+ bia,275+ bia,your_input,fontsize=15)
    if who_win == 1:
        plt.text(120+ bia,95+ bia,'You lose!',color='red',fontsize=30)
    elif who_win == 2:
        plt.text(120+ bia,95+ bia,'You win!',color='green',fontsize=30)
    else:
        plt.text(120+ bia,95+ bia,'Tie!',color='black',fontsize=30)
    plt.axis('off') # 不显示坐标轴
    plt.savefig(final_image_path)
    plt.close('all')

def getInfroduction_transform(img_path, img_name):
    
    # 图片转换后存放的路径
    img_new_path = './image-new/' + img_name

    # 模型预测
    result = model.style_transfer(images=[cv2.imread(img_path)])

    # 将图片保存到指定路径
    cv2.imwrite(img_new_path, result[0])

    # 返回新图片的路径
    return img_new_path


async def on_message(msg: Message):
    global target_output
    global robot_state
    global target
    
    #如果收到的message是一句话
    if isinstance(msg.text(), str) and len(msg.text()) > 0 and msg._payload.type == MessageType.MESSAGE_TYPE_TEXT:
        print("robot_state = " + str(robot_state))
        if robot_state == 0:
            robot_state = 1
            customer_service_number = random.randint(0,999)
            await msg.say("欢迎来到星际纠纷调解事务所\n" + str(customer_service_number) +"号调解员正在为您服务\n请您对纠对象及细节进行描述")
        elif robot_state == 1:
            robot_state = 2
            target = getTarget(msg.text())
            await msg.say("为了解决 您 和 " + target + " 的纠纷,我们建议您使用Space Disagreement Resolver，请问是否需要向您介绍如何使用？（是/否）")
        elif robot_state == 2 and msg.text() == '是':
            # 构建一个FileBox
            robot_state = 3
            file_box_video = FileBox.from_file('/root/paddlejob/workspace/code/video/GameIntroduction.mp4')
            await msg.say("规则介绍载入中，请稍侯...")
            await msg.say(file_box_video)
            await msg.say("该视频将向您介绍Space Disagreement Resolver的使用规则，您可以决定是否使用此方法？（是/否）")
        elif (robot_state == 3 and msg.text() == '否'):
            await msg.say("请多看几遍视频：）")
        elif (robot_state == 2 and msg.text() == '否') or (robot_state == 3 and msg.text() == '是'):
            robot_state = 4
            await msg.say("Space Disagreement Resolver将百分百确保双方出拳结果的保密性")
            await msg.say("正在连线 " + target + "...")
            await msg.say(target + " 同意通过Space Disagreement Resolver解决与您的纠纷")
            candidate_output = ['rock','scissors','paper','lizard',"spock"]
            target_output = candidate_output[random.randint(0,4)]
            await msg.say(target + " 已将其出拳结果存入Space Disagreement Resolver")
            await msg.say("请您进行出拳...")
        else:
            robot_state = 0

    #如果收到的message是一张图片且进入了纠纷解决状态
    if robot_state == 4 and msg.type() == Message.Type.MESSAGE_TYPE_IMAGE:
    
        # 将Message转换为FileBox
        file_box_user_image = await msg.to_file_box()
    
        # 获取图片名
        img_name = file_box_user_image.name
    
        # 图片保存的路径
        img_path = './image/' + img_name
    
        # 将图片保存为本地文件
        await file_box_user_image.to_file(file_path=img_path)
    
        await msg.say("正在将您的出拳存入Space Disagreement Resolver")
        
        # 调用出拳识别函数
        user_input = getUserInputJson(img_path)
        prediction = user_input['results'][0]['name']
        confidence = float(user_input['results'][0]['score'])
            
        print(user_input)
        print(prediction)
        print(confidence)
        # 从新的路径获取图片
        if confidence > 0.3:
            feedback = prediction
        else:
            feedback = 'Unknown'
        
        if feedback == 'Unknown':
            await msg.say('无法识别您的出拳，请再次出拳')
            robot_state = 4
        else:
            final_result = who_win(target_output,feedback)
            
            image_1 = Image.open('/root/paddlejob/workspace/code/image_get/'+target_output+'.png')
            image_2 = Image.open(img_path)
            image_compose(image_1, image_2, target_output, feedback, final_result)
            
            file_box_final_result = FileBox.from_file('./image/final_image.png')

            await msg.say(file_box_final_result)
            
            if final_result == 1:
                await msg.say(target + ' ，请按 ' + target+ ' 的意见解决纠纷')
                await msg.say('感谢您使用Space Disagreement Resolver，再见')
                robot_state = 0
            elif final_result == 2:
                await msg.say('您 胜出了，您 与 ' + target+ ' 之间的纠纷，将按您的意见解决')
                await msg.say('感谢您使用Space Disagreement Resolver，再见')
                robot_state = 0
            else:
                await msg.say('尚未解决争端，请问是否继续？（是/否）')
                robot_state = 4
    
async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    print(user)


async def main():
    # 确保我们在环境变量中设置了WECHATY_PUPPET_SERVICE_TOKEN
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()

    bot.on('scan',      on_scan)
    bot.on('login',     on_login)
    bot.on('message',   on_message)

    await bot.start()

    print('[Python Wechaty] Ding Dong Bot started.')


asyncio.run(main())