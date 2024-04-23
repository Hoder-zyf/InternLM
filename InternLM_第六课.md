# InternLM 第六课 Lagent&AgentLego智能体应用搭建

## 零.笔记

1. **为什么要有智能体**：幻觉；时效性；可靠性
  
2. **智能体概述：**
  
  ![屏幕截图 2024-04-23 221849](https://github.com/Hoder-zyf/InternLM/assets/73508057/217b794f-aa47-4ff3-936b-d58b73806967)



  ![屏幕截图 2024-04-23 221906](https://github.com/Hoder-zyf/InternLM/assets/73508057/c777cab7-ea58-4b30-87e0-35a1fe741d2a)

  
  
3. **智能体范式(AutoGPT)**：
  
  ![屏幕截图 2024-04-23 222236](https://github.com/Hoder-zyf/InternLM/assets/73508057/a60a394e-0fd5-4552-9ab1-cf5996400a58)



  
4. **Lagent**:一个轻量开源智能体框架，也提供了一些工具增强大语言模型能力。
  
![屏幕截图 2024-04-23 222550](https://github.com/Hoder-zyf/InternLM/assets/73508057/a809c533-a1cc-4243-89b7-3b89dfc6c19f)




  
5. **AgentLego**:一个提供了多种开源工具 API 的多模态工具包，旨在像是乐高积木一样，让用户可以快速简便地拓展自定义工具，从而组装出自己的智能体。
  
  ![屏幕截图 2024-04-23 222624](https://github.com/Hoder-zyf/InternLM/assets/73508057/30726064-6239-4d68-bc8b-6539ee65c051)



 
  
6. **AgentLego和Lagent的关系**：Lagent 是一个智能体框架，而 AgentLego 与大模型智能体并不直接相关，而是作为工具包，在相关智能体的功能支持模块发挥作用。
  
  ![屏幕截图 2024-04-23 222653](https://github.com/Hoder-zyf/InternLM/assets/73508057/da6ace5c-e50a-40dc-a8d0-07a067a3cce9)

  ![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/276cbf66-2fef-4715-b521-8c8bd2bb525e)


  

## 一.基础作业

**预备工作**：配置环境:50% A100 cuda12.2+创建环境+安装lagent、AgentLego和Tutorial+安装其他依赖

```powershell
mkdir -p /root/agent
studio-conda -t agent -o pytorch-2.1.2

cd /root/agent
git clone https://gitee.com/internlm/lagent.git
git clone https://gitee.com/internlm/agentlego.git
git clone -b camp2 https://gitee.com/internlm/Tutorial.git

conda activate agent
cd lagent && git checkout 581d9fb && pip install -e . && cd ..
cd agentlego && git checkout 7769e0d && pip install -e . && cd ..
pip install lmdeploy==0.3.0
```

![屏幕截图 2024-04-23 002411](https://github.com/Hoder-zyf/InternLM/assets/73508057/a3d656a2-5600-4f3b-94ed-5c46695ed896)


---

### 1. 完成Lagent Web Demo的使用

1. **使用LMDeploy部署：**
  
  ```powershell
  lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                              --server-name 127.0.0.1 \
                              --model-name internlm2-chat-7b \
                              --cache-max-entry-count 0.1
  ```
  
2. **新建一个terminal启动Lagent Web Demo:**
  
  ```powershell
  conda activate agent
  cd /root/agent/lagent/examples
  streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860
  ```
  
3. **全部加载完成后进行本地端口映射：**
  
  ```powershell
  ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 43767
  ```
  
4. **在本地打开[lagent-web](http://localhost:7860/):**
  
  首先把模型IP设置为127.0.0.1:23333,然后敲击回车后稍等几十秒，选择ArixivSearch插件，就已经加载好插件了。
  
  - 我们输入`帮我搜索InternLM2 Technical Report`后，得到如下输出：
    
    ![屏幕截图 2024-04-23 004553](https://github.com/Hoder-zyf/InternLM/assets/73508057/4bf8bf5f-dfcb-47e8-b046-a14321d423a7)

    
  - 我们输入`帮我搜索Finbert的相关论文`,得到如下结果：
    
    ![屏幕截图 2024-04-23 005054](https://github.com/Hoder-zyf/InternLM/assets/73508057/09f57931-11c3-42dc-84ab-7e2a01b2e140)


    
  
  ---
  
  **综合感觉效果不错，速度也不错，如果可以设置搜索几篇特定文章就好了**
  

### 2.直接使用 AgentLego

1. 下载demo文件+安装目标检测工具（基于mmdet算法库中的RTMDet-Large模型）
  
  ```powershell
  cd /root/agent
  wget http://download.openmmlab.com/agentlego/road.jpg
  
  # 先安装mim,再通过mim安装mmdet
  conda activate agent
  pip install openmim==0.3.9
  mim install mmdet==3.3.0g
  ```
  
  ---
  
  RTMDet-Large是mmdetection算法库中的一个大型实时目标检测模型。mmdetection是一个基于PyTorch的开源目标检测工具箱，它是OpenMMLab项目的一部分。RTMDet是该工具箱中最新发布的一系列全卷积单阶段检测模型，专为实时目标识别任务设计。
  
  根据搜索结果中的技术报告摘要，RTMDet模型的主要特点包括：
  
  1. **高效实时检测**：RTMDet旨在设计一个高效的实时目标检测器，超越了YOLO系列的性能，并且易于扩展，适用于多种目标识别任务，如实例分割和旋转目标检测。
    
  2. **大卷积核**：通过在模型的backbone（主干网络）和neck（颈部网络）中使用大卷积核的depth-wise convolution（深度卷积），增强了模型捕获全局上下文的能力。
    
  3. **软标签动态标签分配**：在训练策略中，RTMDet引入了软标签来计算匹配成本，以改进动态标签分配策略，从而提高模型的准确性。
    
  4. **参数-精度权衡**：RTMDet在不同的模型尺寸（tiny、small、medium、large、extra-large）上都实现了参数数量和准确性之间的最佳权衡。
    
  5. **性能**：RTMDet-Large模型在COCO数据集上达到了52.8%的AP（平均精度）得分，并在NVIDIA 3090 GPU上以300+ FPS（每秒帧数）的速度运行，超越了当前主流的工业级目标检测器。
    
  6. **扩展性**：RTMDet可以轻松扩展到实例分割和旋转目标检测任务，只需进行少量修改。
    
  7. **代码和模型开源**：RTMDet的代码和预训练模型已经开源，可以在GitHub的mmdetection相关仓库中找到。
    
  
  RTMDet-Large作为系列中的一个型号，专为需要较高准确率和计算资源充足的应用场景设计。它在保持实时性的同时，提供了更高的检测精度，适用于对检测性能要求较高的工业应用。
  
  ---
  
2. **在 /root/agent 目录下新建 direct_use.py 以直接使用目标检测工具**:
  
  ```powershell
  touch /root/agent/direct_use.py
  ```
  
  写入如下代码：
  
  ```python
  import re
  
  import cv2
  from agentlego.apis import load_tool
  
  # load tool
  tool = load_tool('ObjectDetection', device='cuda')
  
  # apply tool
  visualization = tool('/root/agent/road.jpg')
  print(visualization)
  
  # visualize
  image = cv2.imread('/root/agent/road.jpg')
  
  preds = visualization.split('\n')
  pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'
  
  for pred in preds:
      name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
      x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
      cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
  
  cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
  ```
  
3. **执行代码：**`python /root/agent/direct_use.py`
  
4. **效果展示：** **比较好的完成了目标检测的任务**
  
  ![屏幕截图 2024-04-23 011629](https://github.com/Hoder-zyf/InternLM/assets/73508057/e19f8250-2fd7-4ebb-a515-2ecd680b4889)


  
  ![road_detection_direct](https://github.com/Hoder-zyf/InternLM/assets/73508057/27139436-f54a-4ebc-9e5d-a364864c94c7)

  

## 二.进阶作业

### 1.将AgentLego作为智能体工具使用

1. **修改配置文件:** `vim /root/agent/agentlego/webui/modules/agents/lagent_agent.py` 把model_name从20b改为7b(105行)
  
2. **使用LMDeploy部署**：
  
  ```powershell
  conda activate agent
  lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                              --server-name 127.0.0.1 \
                              --model-name internlm2-chat-7b \
                              --cache-max-entry-count 0.1
  ```
  
3. **新建一个terminal启动AgentLego Webui:**
  
  ```powershell
  conda activate agent
  cd /root/agent/agentlego/webui
  python one_click.py
  ```
  
4. **全部加载完成后进行本地端口映射:**
  
  ```powershell
  ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 43767
  ```
  
5. **在本地打开[AgentLego Web UI](http://localhost:7860/):**
  
  **按照教程在agent部分进行以下设置（最后点击save to以保存配置，这样以后使用只要选择internlm2后加载就行）**
  
  **！！！！！记得点击Load以加载配置**
  
  ![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/d600d545-29a1-4c43-82ad-d4caaf4098aa)

  
  **按照教程在tools部分进行以下设置（Tool选择new tool,选择ObjectDetection后点击save to以保存配置**
  
  ![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/add90e4a-012e-4c3b-b253-9d04b38fd3fd)

  
  **进入Chat界面，在页面下方只选择ObjectDetection工具这一个工具**
  
6. **实践结果：**
  
  - 输入`road.jpg`和`请检测图中物体`，输出如下：
    
    ![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/f16f46e5-d70b-400e-abf7-c719595c2c0f)

    
  - 输入`cat.jpg`和`请检测图中物体`，输出如下：
    
    ![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/6024fa37-153f-404c-9ff6-551de03cbe72)

    
    ---
    
    **感觉效果还可以，但是一旦多几轮对话，直接就不答复了？！**
    
  
  ### 2.使用 Lagent自定义工具并调用(天气查询)
  
  ---
  
  Lagent 中关于工具部分的介绍文档位于 https://lagent.readthedocs.io/zh-cn/latest/tutorials/action.html 。使用 Lagent 自定义工具主要分为以下几步：
  
  1. 继承 BaseAction 类
  2. 实现简单工具的 run 方法；或者实现工具包内每个子工具的功能
  3. 简单工具的 run 方法可选被 tool_api 装饰；工具包内每个子工具的功能都需要被 tool_api 装饰
  
  ---
  
  1. 创建工具文件：`touch /root/agent/lagent/lagent/actions/weather.py`
    
    **大概的思路就是通过用户输入的城市匹配到city_code，再通过city_code匹配天气**
    
    ```python
    import json
    import os
    import requests
    from typing import Optional, Type
    
    from lagent.actions.base_action import BaseAction, tool_api
    from lagent.actions.parser import BaseParser, JsonParser
    from lagent.schema import ActionReturn, ActionStatusCode
    
    class WeatherQuery(BaseAction):
        """Weather plugin for querying weather information."""
    
        def __init__(self,
                     key: Optional[str] = None,
                     description: Optional[dict] = None,
                     parser: Type[BaseParser] = JsonParser,
                     enable: bool = True) -> None:
            super().__init__(description, parser, enable)
            key = os.environ.get('WEATHER_API_KEY', key)
            if key is None:
                raise ValueError(
                    'Please set Weather API key either in the environment '
                    'as WEATHER_API_KEY or pass it as `key`')
            self.key = key
            self.location_query_url = 'https://geoapi.qweather.com/v2/city/lookup'
            self.weather_query_url = 'https://devapi.qweather.com/v7/weather/now'
    
        @tool_api
        def run(self, query: str) -> ActionReturn:
            """一个天气查询API。可以根据城市名查询天气信息。
    
            Args:
                query (:class:`str`): The city name to query.
            """
            tool_return = ActionReturn(type=self.name)
            status_code, response = self._search(query)
            if status_code == -1:
                tool_return.errmsg = response
                tool_return.state = ActionStatusCode.HTTP_ERROR
            elif status_code == 200:
                parsed_res = self._parse_results(response)
                tool_return.result = [dict(type='text', content=str(parsed_res))]
                tool_return.state = ActionStatusCode.SUCCESS
            else:
                tool_return.errmsg = str(status_code)
                tool_return.state = ActionStatusCode.API_ERROR
            return tool_return
    
        def _parse_results(self, results: dict) -> str:
            """Parse the weather results from QWeather API.
    
            Args:
                results (dict): The weather content from QWeather API
                    in json format.
    
            Returns:
                str: The parsed weather results.
            """
            now = results['now']
            data = [
                f'数据观测时间: {now["obsTime"]}',
                f'温度: {now["temp"]}°C',
                f'体感温度: {now["feelsLike"]}°C',
                f'天气: {now["text"]}',
                f'风向: {now["windDir"]}，角度为 {now["wind360"]}°',
                f'风力等级: {now["windScale"]}，风速为 {now["windSpeed"]} km/h',
                f'相对湿度: {now["humidity"]}',
                f'当前小时累计降水量: {now["precip"]} mm',
                f'大气压强: {now["pressure"]} 百帕',
                f'能见度: {now["vis"]} km',
            ]
            return '\n'.join(data)
    
        def _search(self, query: str):
            # get city_code
            try:
                city_code_response = requests.get(
                    self.location_query_url,
                    params={'key': self.key, 'location': query}
                )
            except Exception as e:
                return -1, str(e)
            if city_code_response.status_code != 200:
                return city_code_response.status_code, city_code_response.json()
            city_code_response = city_code_response.json()
            if len(city_code_response['location']) == 0:
                return -1, '未查询到城市'
            city_code = city_code_response['location'][0]['id']
            # get weather
            try:
                weather_response = requests.get(
                    self.weather_query_url,
                    params={'key': self.key, 'location': city_code}
                )
            except Exception as e:
                return -1, str(e)
            return weather_response.status_code, weather_response.json()
    ```
    
  2. **进入和风开发，注册账号。然后创建项目，选择`免费订阅`和`Web API`,创建我们的API KEY.**
   
    
  3. **使用LMDeploy部署**
    
    ```powershell
    conda activate agent
    lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                                --server-name 127.0.0.1 \
                                --model-name internlm2-chat-7b \
                                --cache-max-entry-count 0.1
    ```
    
  4. **在一个新的terminal启动Lagent Web Demo**:
    
    ```powershell
    export WEATHER_API_KEY={API KEY}
    conda activate agent
    cd /root/agent/Tutorial/agent
    streamlit run internlm2_weather_web_demo.py --server.address 127.0.0.1 --server.port 7860
    ```
    
  5. **在本地进行端口映射**
    
    ```powershell
    ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 43767
    ```
    
  6. **本地打开[lagent-web](http://localhost:7860/):**
    
    在模型IP处输入`127.0.0.1:23333`,在插件选择处选择刚刚创建的`WeatherQuery`插件（就是刚刚weather中的`class`名称）
    
    ---
    
    **Q:为什么之前调用的ArxivSearch不见了呢?**
    
    **A:之前我们运行的是internlm2_agent_web_demo.py,现在运行的是internlm2_weather_web_demo.py**
    
    如果我们想把ArxivSearch加上，应该只要加上`from lagent.actions import ArxivSearch`，然后在第26行action_list部分加上ArxivSearch()即可。
    
    **效果如图（效果挺好）：** **同时调用ArxivSearch和WeatherQuery**
    
    ![屏幕截图 2024-04-23 110233](https://github.com/Hoder-zyf/InternLM/assets/73508057/d1bbd70f-c52d-4e31-b7ac-7a58a0f4863f)


    
    ![屏幕截图 2024-04-23 110309](https://github.com/Hoder-zyf/InternLM/assets/73508057/9299d411-e7ee-4a39-8931-ef16fb0dd48e)

    
    ---
    
    **天气查询：**`查询常州的天气`
    
    ![屏幕截图 2024-04-23 105601](https://github.com/Hoder-zyf/InternLM/assets/73508057/4e405f05-4c9b-4b75-9290-64593e24a709)

    

### 3.使用 AgentLego自定义工具并调用

---

在本节中，我们将基于 AgentLego 构建自己的自定义工具。AgentLego 在这方面提供了较为详尽的文档，文档地址为 [自定义工具 &mdash; AgentLego 0.2.0 文档](https://agentlego.readthedocs.io/zh-cn/latest/modules/tool.html) 。自定义工具主要分为以下几步：

1. 继承 BaseTool 类
2. 修改 default_desc 属性（工具功能描述）
3. 如有需要，重载 setup 方法（重型模块延迟加载）
4. 重载 apply 方法（工具功能实现）

其中第一二四步是必须的步骤。下面我们将实现一个调用 MagicMaker 的 API 以实现图像生成的工具。MagicMaker 是汇聚了优秀 AI 算法成果的免费 AI 视觉素材生成与创作平台。主要提供图像生成、图像编辑和视频生成三大核心功能，全面满足用户在各种应用场景下的视觉素材创作需求。体验更多功能可以访问 [Magic Maker](https://magicmaker.openxlab.org.cn/home) 。

---

1. **创建工具文件：**`touch /root/agent/agentlego/agentlego/tools/magicmaker_image_generation.py`
  
  ```python
  import json
  import requests
  
  import numpy as np
  
  from agentlego.types import Annotated, ImageIO, Info
  from agentlego.utils import require
  from .base import BaseTool
  ```
  
  class MagicMakerImageGeneration(BaseTool):
  
  ```
  default_desc = ('This tool can call the api of magicmaker to '
                  'generate an image according to the given keywords.')
  
  styles_option = [
      'dongman',  # 动漫
      'guofeng',  # 国风
      'xieshi',   # 写实
      'youhua',   # 油画
      'manghe',   # 盲盒
  ]
  aspect_ratio_options = [
      '16:9', '4:3', '3:2', '1:1',
      '2:3', '3:4', '9:16'
  ]
  
  @require('opencv-python')
  def __init__(self,
               style='guofeng',
               aspect_ratio='4:3'):
      super().__init__()
      if style in self.styles_option:
          self.style = style
      else:
          raise ValueError(f'The style must be one of {self.styles_option}')
  
      if aspect_ratio in self.aspect_ratio_options:
          self.aspect_ratio = aspect_ratio
      else:
          raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')
  
  def apply(self,
            keywords: Annotated[str,
                                Info('A series of Chinese keywords separated by comma.')]
      ) -> ImageIO:
      import cv2
      response = requests.post(
          url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
          data=json.dumps({
              "official": True,
              "prompt": keywords,
              "style": self.style,
              "poseT": False,
              "aspectRatio": self.aspect_ratio
          }),
          headers={'content-type': 'application/json'}
      )
      image_url = response.json()['data']['imgUrl']
      image_response = requests.get(image_url)
      image = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_response.content, np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
      return ImageIO(image)
  ```
  

```
---

**代码解释：**

最后4行代码是用于从网络请求图片，并对其进行处理以便于显示或进一步处理的Python代码。下面是对每一行代码的解释：

1. `image_url = response.json()['data']['imgUrl']` 这行代码从一个名为 `response` 的响应对象中提取图片的URL。`response` 可能是通过发送网络请求（例如使用 `requests.get`）获得的，它包含了一个JSON格式的响应体。这里使用 `response.json()` 来解析JSON格式的响应体为Python字典，然后通过键 `data` 与 `imgUrl` 来获取图片的URL。

2. `image_response = requests.get(image_url)` 这行代码使用 `requests` 库发起一个GET请求到上一步获取的图片URL，目的是下载图片。`image_response` 是这次请求的响应对象。

3. `image = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_response.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)` 这行代码执行了以下操作：

   - `image_response.content` 获取响应内容，即图片的字节流。
   - `np.frombuffer(image_response.content, np.uint8)` 将字节流转换为一个 `numpy` 数组。
   - `cv2.imdecode(..., cv2.IMREAD_COLOR)` 使用 `cv2` (OpenCV库) 将图片的字节流解码为一个颜色的图像（默认是BGR格式）。
   - `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` 将BGR格式的图像转换为RGB格式，因为大多数图像处理库和显示设备使用RGB色彩空间。

4. `return ImageIO(image)` 这行代码返回一个 `ImageIO` 对象，它可能是一个用于处理或显示图像的自定义对象或函数。这个对象或函数接收一个图像（在这里是经过处理的RGB格式的图像）作为输入。注意，`ImageIO` 并不是Python标准库或常用第三方库中的一部分，它可能是用户自定义的一个类或函数。

**总结来说，这几行代码的作用是从一个网络响应中提取图片URL，下载图片，将其转换为适用于进一步处理的RGB格式图像，并最终返回一个包含该图像的对象或用于后续操作的图像数据。**

---

2. **注册新工具**：修改 `/root/agent/agentlego/agentlego/tools/__init__.py` 文件，将我们的工具注册在工具列表中。(将 MagicMakerImageGeneration 通过 from .magicmaker_image_generation import MagicMakerImageGeneration 导入到了文件中，并且将其加入了 __all__ 列表中。)
```

# 加入

from .magicmaker_image_generation import MagicMakerImageGeneration

# 将__all__中'BaseTool'增加MagicMakerImageGeneration

['BaseTool', 'make_tool', 'BingSearch', 'MagicMakerImageGeneration']

````
3. **使用LMDeploy部署**

```powershell
conda activate agent
lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                            --server-name 127.0.0.1 \
                            --model-name internlm2-chat-7b \
                            --cache-max-entry-count 0.1
````

4. **在一个新的terminal启动AgentLego WebUI**:
  
  ```powershell
  conda activate agent
  cd /root/agent/agentlego/webui
  python one_click.py
  ```
  
5. **在本地进行端口映射**:
  
  ```powershell
  ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 43767
  ```
  
6. **在本地打开[AgentLego Web UI](http://localhost:7860/):**
  
  - 在Agent部分选择之前的internlm2,然后Load
    
  - Tools中选择new tool,然后选择我们刚刚注册的`MagicMakerImageGeneration`,然后点击save保存配置
    
  - 返回Chat部分，我们此处只选择`MagicMakerImageGeneration`这一个插件。
    
7. **结果展示**
  
  `画一张小猪佩奇大战变形金刚的图片`：**小猪佩奇画的不错，变形金刚一言难尽**
  
  ![屏幕截图 2024-04-23 120354](https://github.com/Hoder-zyf/InternLM/assets/73508057/755bd0fe-6487-425e-97a4-2d9b1f2db7f3)

  
  `画一个女生，身姿曼妙，戴着面纱，眼睛是蓝色的`
  
![屏幕截图 2024-04-23 121049](https://github.com/Hoder-zyf/InternLM/assets/73508057/b240b85c-151f-4e87-8f54-83fc317fe2f0)


  **应该和默认国风有关系，要求都满足了，但是感觉看上去有点让人害怕，而且不知道为什么会连续生成两张一模一样的照片**
