# InternLM 第五课

## 零.视频课程

1. **大模型部署面临的挑战——计算量大**：
  ![屏幕截图 2024-04-16 212537](https://github.com/Hoder-zyf/InternLM/assets/73508057/948eb7e0-98f7-45ad-8a13-5aa389c65cab)
  
  
2. **大模型部署面临的挑战——内存开销大**：
  ![屏幕截图 2024-04-16 212740](https://github.com/Hoder-zyf/InternLM/assets/73508057/13de3c3b-6977-43af-b38b-b3d4a3cea39b)
 
3. **大模型部署面临的挑战——访存瓶颈**：
  
  **访存瓶颈是指在计算机系统中，由于处理器（如CPU或GPU）与内存之间的数据交换速度不足以满足计算需求，导致整体性能受限的现象。这种情况通常发生在处理器需要频繁地从内存中读取或写入大量数据时，而内存的带宽和延迟无法跟上处理器的速度，从而成为系统性能提升的制约因素。**
  
  ---
  
  **计算量**：**计算量通常指的是完成一个计算任务所需的基本运算次数，例如浮点运算（FLOPs）或整数运算（INTOPS）。在深度学习中，计算量通常与神经网络模型的大小、复杂度以及输入数据的批量大小（batch size）有关。计算量是衡量模型训练或推理过程中所需执行的总计算工作量的一个指标。计算量越大，意味着需要更多的计算资源和时间来完成模型的训练或推理。**
  
  **访存量：访存量是指在执行计算任务过程中，处理器需要从内存中读取或写入的数据量。在深度学习中，访存量与模型的参数数量、激活值（即网络中每个神经元的输出）以及输入数据的大小有关。访存量反映了模型在前向传播或反向传播过程中与内存交互的频率和数据量。如果访存量很大，可能会导致内存带宽成为瓶颈，从而影响计算效率。**
  
  **显存带宽：显存带宽是指图形处理单元（GPU）的显存与处理器核心之间的数据传输速率，通常以GB/s（千兆字节每秒）为单位。显存带宽决定了GPU能够多快地读写存储在其显存中的数据。在深度学习和其他图形密集型任务中，显存带宽对于整体性能至关重要。如果显存带宽不足，即使GPU的计算能力很强，也无法充分发挥其性能，因为数据传输的瓶颈会限制计算任务的执行速度。**
   ![屏幕截图 2024-04-16 212959](https://github.com/Hoder-zyf/InternLM/assets/73508057/f14a8f90-e9e4-4e38-a492-0c1714177e97)
  
 
  
4. **大模型部署方法：**
  
  1）模型剪枝：减少参数量，在保证性能最低下降的同时，减少存储需求，提高计算效率。
  
  2）知识蒸馏：模型压缩，通过引导轻量化的学生模型“模仿”性能更好、结构更复杂的教师模型，在不改变学生模型结果的基础上提高性能。（比如上下文学习、CoT、指令跟随）
  
  3）量化：把浮点数改成整数或者其他的离散形式
  
5. **LMDeploy核心功能：**
  
   ![屏幕截图 2024-04-16 214328](https://github.com/Hoder-zyf/InternLM/assets/73508057/6cf9a662-4c54-4b11-ac26-34f79aa077ad)

## 一.基础作业

1. **选择开发机**： 30% A100 **cuda12.2**
  
2. **激活环境并安装LMDeploy**: `conda activate lmdeploy pip install lmdeploy[all]==0.3.0`
  
![屏幕截图 2024-04-16 225025](https://github.com/Hoder-zyf/InternLM/assets/73508057/4a0d00d2-8861-4c7c-86b3-37799c64ddbd)

3. **使用LMDeploy与1.8B模型对话**:
  
  ```powershell
  conda activate lmdeploy
  # 使用LMDeploy与模型进行对话的通用命令格式为：
  # lmdeploy chat [HF格式模型路径/TurboMind格式模型路径]
  # 与1.8b模型对话：
  lmdeploy chat /root/internlm2-chat-1_8b
  ```
  
   ![屏幕截图 2024-04-16 231219](https://github.com/Hoder-zyf/InternLM/assets/73508057/db3712a8-1681-4cb0-862a-0432d9f062cb)
  
  **输出的速度相比hugging face真的快了很多，但是毕竟1.8B模型，这个知识库还是比较有限的，比如我问周志华，他不认识这个人。**
  
  **随后我尝试了7B模型，知道周志华是机器学习专家了，但是我问他是哪里毕业的，这个回答是错的......，幻觉还是挺严重的**
  

## 二.进阶作业

**前置知识：**

- 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速度。
- 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。

**常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。**

**那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用KV8量化和W4A16量化。KV8量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。W4A16 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。**

---

1. **设置最大KV Cache缓存大小**：不加`--cache-max-entry-count`参数，**显存为20944 / 24566 MiB**,加入**0.4**的限制后，**显存为12776 / 24566 MiB**，显著下降。
  
  ```powershell
  lmdeploy chat /root/internlm2-chat-1_8b
  lmdeploy chat /root/internlm2-chat-1_8b --cache-max-entry-count 0.4
  ```
  
2. **使用W4A16量化+KV Cache缓存大小**：
  
  ---
  
  **LMDeploy使用AWQ算法，实现模型4bit权重量化。推理引擎TurboMind提供了非常高效的4bit推理cuda kernel，性能是FP16的2.4倍以上。它支持以下NVIDIA显卡：**
  
  - **图灵架构（sm75）：20系列、T4**
  - **安培架构（sm80,sm86）：30系列、A10、A16、A30、A100**
  - **Ada Lovelace架构（sm90）：40 系列**
  
  ```powershell
  pip install einops==0.7.0
  
  # 调用量化功能，使用自动awq算法；ptb数据集，采样128个数据对；上下文长度是1024
  lmdeploy lite auto_awq \
     /root/internlm2-chat-1_8b \
    --calib-dataset 'ptb' \
    --calib-samples 128 \
    --calib-seqlen 1024 \
    --w-bits 4 \
    --w-group-size 128 \
    --work-dir /root/internlm2-chat-1_8b-4bit
  ```
  
  这个命令是在部署一个语言模型，其中使用了一些参数来配置模型的一些设置。**其中包括模型的校准数据集为'ptb'，校准样本为128个，校准序列长度为1024。另外，模型还使用了4位的权重精度和128个权重分组大小。工作目录是在'/root/internlm2-chat-1_'8b-4bit下。**
  
  **对话下来感觉和原始模型差别不大。**
  
  ```powerquery
  # 显存为20520 / 24566 MiB
  lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq
  
  # 显存为11496 / 24566 MiB
  lmdeploy chat /root/internlm2-chat-1_8b-4bit --model-format awq --cache-max-entry-count 0.4
  ```
  
  
 ![屏幕截图 2024-04-17 010224](https://github.com/Hoder-zyf/InternLM/assets/73508057/7a32ac09-7f91-45d5-945a-cad73b6998cb)

  

3. **API Server启动Imdeploy，开启W4A16量化+KV Cache=0.4**：
  
  ---
  
  **前置知识：**
  
  在生产环境下，我们有时会将大模型封装为API接口服务，供客户端访问。
  
  我们来看下面一张架构图：
  
  ![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/437544c3-c4b8-4a03-aaea-770d430aa130)

  
  我们把从架构上把整个服务流程分成下面几个模块。
  
  - **模型推理/服务。主要提供模型本身的推理，一般来说可以和具体业务解耦，专注模型推理本身性能的优化。可以以模块、API等多种方式提供。**
  - **API Server。中间协议层，把后端推理/服务通过HTTP，gRPC或其他形式的接口，供前端调用。**
  - **Client。可以理解为前端，与用户交互的地方。通过通过网页端/命令行去调用API接口，获取模型推理/服务。**
  
  **值得说明的是，以上的划分是一个相对完整的模型，但在实际中这并不是绝对的。比如可以把“模型推理”和“API Server”合并，有的甚至是三个流程打包在一起提供服务。**
  
  ---
  
  ```powershell
  # tp参数表示并行数量（GPU数量）
  lmdeploy serve api_server \
      /root/internlm2-chat-1_8b-4bit \
      --model-format awq \
      --quant-policy 0 \
      --server-name 0.0.0.0 \
      --server-port 23333 \
      --cache-max-entry-count 0.4 \
      --tp 1
  ```
  
  然后我们可以在本地进行端口映射`ssh -CNg -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 43767`，打开`[FastAPI - Swagger UI](http://127.0.0.1:23333/)`,**这里面有详细的接口使用说明**：
  
 ![屏幕截图 2024-04-17 004105](https://github.com/Hoder-zyf/InternLM/assets/73508057/5a0b8188-8f1e-4ceb-a05a-baf2db4613a2)

  
4. **使用命令行连接API与模型对话**： 打开vscode，新建一个terminal，先激活环境，然后`lmdeploy serve api_client http://localhost:23333`即可
  
  ![屏幕截图 2024-04-17 004556](https://github.com/Hoder-zyf/InternLM/assets/73508057/e04f1fc0-6ef2-4d9e-9e15-f980e8e46e81)

  
  ---
  
  **现在框架如下：**
  
  ![](https://github.com/InternLM/Tutorial/raw/camp2/lmdeploy/imgs/4.2_4.jpg)
  
5. **使用Gradio网页客户端连接API与模型对话**：打开vscode，新建一个terminal，先激活环境，然后`lmdeploy serve gradio http://localhost:23333 \
   --server-name 0.0.0.0 \
   --server-port 6006`
  
  接着在**本地**powershell中进行窗口映射`ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 43767`，访问地址`[Gradio](http://127.0.0.1:6006/)`
  
  ![屏幕截图 2024-04-17 005159](https://github.com/Hoder-zyf/InternLM/assets/73508057/0ec6dd3c-8c04-45e1-9547-4f9ff8f375d9)

  
  ---
  
  **现在框架如下：**
  
  ![](https://github.com/InternLM/Tutorial/raw/camp2/lmdeploy/imgs/4.3_3.jpg)
  
6. **使用W4A16量化，调整KV Cache的占用比例为0.4，使用Python代码集成的方式运行internlm2-chat-1.8b模型**:
  
  - 新建python代码pipeline.py:`touch /root/pipeline/py`
    
  - 参考官方文档设置`KV Cache和model_format`
    
    ![屏幕截图 2024-04-17 011936](https://github.com/Hoder-zyf/InternLM/assets/73508057/e2c9f80b-0d5c-4d3c-91eb-b1ebb8ac04e0)

    ![屏幕截图 2024-04-17 012141](https://github.com/Hoder-zyf/InternLM/assets/73508057/3deb187f-38a2-4ecf-a18e-7b6782e354a0)

    
    ```python
    from lmdeploy import pipeline, TurbomindEngineConfig
    
    # 调整KV Cache占比,使用W4A16量化
    backend_config = TurbomindEngineConfig(cache_max_entry_count=0.4,model_foramt='awq')
    
    pipe = pipeline('/root/internlm2-chat-1_8b-4bit',
                    backend_config=backend_config)
    response = pipe(['Hi, pls intro yourself', '上海是'])
    print(response)
    ```
    
  - 结果：
    
    ![屏幕截图 2024-04-17 012532](https://github.com/Hoder-zyf/InternLM/assets/73508057/32548c93-5741-4da4-acc6-fcbcae6220ce)

    
7. **使用LMDeploy运行视觉多模态大模型llava**
  
  - 安装llava依赖库：`pip install git+https://github.com/haotian-liu/LLaVA.git@4e2277a060da264c4f21b364c867cc622c945874`
    
  - 新建python文件pipeline_llava.py:`touch /root/pipeline_llava.py`
    
    ```python
    # 载入图片的load_image参数
    from lmdeploy.vl import load_image
    from lmdeploy import pipeline, TurbomindEngineConfig
    
    # 图片分辨率较高时请调高session_len
    backend_config = TurbomindEngineConfig(session_len=8192) 
    # pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
    
    pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)
    # 从github下载了一张关于老虎的图片
    image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
    
    response = pipe(('describe this image', image))
    print(response)
    ```
    
  - **示例1：tiger.jpeg**
    
    ![](https://github.com/InternLM/Tutorial/raw/camp2/lmdeploy/imgs/6.1_1.jpg)
    
    ![屏幕截图 2024-04-17 013735](https://github.com/Hoder-zyf/InternLM/assets/73508057/870dff77-a9cf-4c1c-b739-d5c21302d68a)

    
    **这幅图片显示一只老虎躺在一个长满草的区域上，它的头转向一侧，直视着摄像机。老虎的皮毛呈现出独特的条纹图案，边缘较暗，中间较浅。老虎的眼睛睁得很大，嘴巴闭合着。背景虽然模糊，但看起来是一个自然的户外环境，有绿草和一些树木或灌木，暗示着是一个野生动物保护区或类似的环境。图片的焦点是在老虎身上，背景轻柔地模糊了。**
    
  - **示例2：cat.jpg**
    
    ![cat](https://github.com/Hoder-zyf/InternLM/assets/73508057/8a8fa179-6c16-4ca5-97f1-f1ce26914d52)

    
    ```python
    from lmdeploy.vl import load_image
    from lmdeploy import pipeline, TurbomindEngineConfig
    
    
    backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
    # pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
    pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)
    
    image = load_image('/root/cat.jpg')
    response = pipe(('describe this image', image))
    print(response)
    ```
    
    ![屏幕截图 2024-04-17 014630](https://github.com/Hoder-zyf/InternLM/assets/73508057/8aa415fc-a950-40ef-b7e4-dcfbca81c7ba)

    
    **这是一幅插画，描绘了一只坐在桌子前的猫，看起来正在用电脑工作。这只猫有着黄色和白色的外套，有着独特的虎斑图案，穿着一件绿色的衬衫和领带。桌子上乱七八糟地摆放着各种物品，包括电脑显示器、键盘、鼠标和一个杯子。猫的姿势表明它在专心致志地工作，头向前倾，眼睛专注地盯着屏幕。背景有一扇窗户，阳光透过窗户射进来，营造出温馨舒适的氛围。艺术风格让人联想到手绘插图，色调柔和，给人一种迷人而奇妙的感觉。**
    
8. **使用Gradio运行llava模型**：
  
  - 新建`gradio_llava.py`:
    
    ```python
    import gradio as gr
    from lmdeploy import pipeline, TurbomindEngineConfig
    
    
    backend_config = TurbomindEngineConfig(session_len=8192) # 图片分辨率较高时请调高session_len
    # pipe = pipeline('liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config) 非开发机运行此命令
    pipe = pipeline('/share/new_models/liuhaotian/llava-v1.6-vicuna-7b', backend_config=backend_config)
    
    def model(image, text):
        if image is None:
            return [(text, "请上传一张图片。")]
        else:
            response = pipe((text, image)).text
            return [(text, response)]
    
    demo = gr.Interface(fn=model, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs=gr.Chatbot())
    demo.launch()   
    ```
    
    ---
    
    ![屏幕截图 2024-04-17 020230](https://github.com/Hoder-zyf/InternLM/assets/73508057/9b0b9f82-30e6-4ffa-b331-a385f79d45cf)

    ![屏幕截图 2024-04-17 020303](https://github.com/Hoder-zyf/InternLM/assets/73508057/f2aca937-16c8-4d84-a7fa-3d6b4720b8c7)

    ![屏幕截图 2024-04-17 020019](https://github.com/Hoder-zyf/InternLM/assets/73508057/f1b8c8fa-8ba1-4f50-bd25-492b32cf045c)

    ---
    
  - 运行gradio_llava.py,通过ssh转发端口`ssh -CNg -L 7860:127.0.0.1:7860 root@ssh.intern-ai.org.cn -p 43767`,然后打开`[Gradio](http://127.0.0.1:7860/)`
    
  - 效果：
    
    ![屏幕截图 2024-04-17 020710](https://github.com/Hoder-zyf/InternLM/assets/73508057/3f5e7b03-9da7-4130-9c00-9c821da8fe89)

    
    **翻译：这个角色看起来像是在睡觉，手里还拿着一只咖啡杯。所以，他可能是感觉很累，或者正在享受一种清爽的感觉。而且，他手里拿着咖啡，可能是觉得自己正处在忙碌却也能够得到放松的情境中。**（不知道为啥回答是日语）
