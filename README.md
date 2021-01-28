# Mask-recognition
基于Paddle-Lite的口罩识别项目

## 使用
### 获取源代码

```shell
git clone https://github.com/zzh-blog/Mask-recognition.git
```

### 更改预编译库
> **armlinux.armv7hf.gcc 编译平台可跳过此步骤**

>本项目不提供预编译文件，内部携带的Paddle-Lite预编译库为官方发布的armv7预编译库，所以如果在其他Paddle-Lite支持的微型运算单元上运行本程序，你需要额外更改预编译库

1. 下载[Paddle-Lite预编译库](https://github.com/PaddlePaddle/Paddle-Lite/releases/)

2. 解压后用Paddle-Lite同名文件夹替换工程目录下文件夹
    + 先删除目录下Paddle-Lite
    + 后拷贝新的支持库

### 安装 Opencv3

> 这个过程依然是已安装Opencv3的编译环境可以跳过

具体步骤仁者见仁，智者见智，不予多说，可以自行上baidu或bing搜索关于目标平台的安装方法

### 编译

1. 切换到工程目录 (依据clone下的工程位置确定<Project>的内容)
```shell
cd <Project>
```

2. 创建编译后文件目录

```shell
mkdir build
```
3. 进入目录生成makefile并且退出目录

```shell
cd build
cmake ..
cd ..
```

4. 开始编译 (\<n\>代表编译线程，请改成一个用户设定值)
```shell
cmake --build build --config Release --target Mask -- -j <n>
```

**完整编译命令** (方便不需要修改编译参数的人)
```shell
rm -r ./build && mkdir ./build && cd build && cmake .. && cd .. && cmake --build build --config Release --target Mask -- -j 1
```

### 测试
在项目目录下执行命令进行测试  
**前期准备** :  
+ Python环境为python3.x
+ pip安装opencv: **pip install opencv-python**

执行命令：
```shell
python CycleTest.py
```

### [拓展]
CycleTest.py为基础识别的demo，可以参考其实现调用 MaskApi.py这个包装

## MaskApi.py的返回数据

|条目|内容|

使用准备 : 需要实例化Mask_Api类 (程序自动引入编译后的支持库)
```python
[Object] = Mask_Api()
[Object].Load()
```
**注意** : Load() 方法不填入任何数据默认载入内置识别模型

进入检测 :

```python
[Object].Check(Img)
```
|条目|内容|
|-|-|
|返回类型|str (字符串)|
|内容类型|dict (字典)|

结构 (以 Num = 1 举例子): 
```python
Content = {
    'Num': 1,
    'Data': [
        {
            'x': 239,
            'y': 252,
            'width': 56,
            'height': 70,
            'prob': 0.005112
        }
    ]}

```
解释 :
|字段|type|含义|
|-|-|-|
|Num|int|画面中人脸数|
|Data|dict|人脸的详细数据|
|x|int|单个人脸在画面中的横坐标|
|y|int|单个人脸在画面中的纵坐标|
|width|int|单个人脸在画面中的宽度|
|height|int|单个人脸在画面中的高度|
|prob|double|单个人脸戴口罩的可能性|

**Enjoy it !**