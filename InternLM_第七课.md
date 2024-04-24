# InternLM 第七课 OpenCompass评测

## 零.笔记

1. **大模型评测面临的挑战**：全面性；评测成本；数据污染；鲁棒性
  
2. **OpenCompass评测体系**：
  
  - **如何评测？**
    
    ![屏幕截图 2024-04-23 224624](https://github.com/Hoder-zyf/InternLM/assets/73508057/967be025-24d5-4da3-9ff9-54999a368c51)

    
  - **客观评测**（客观问答题与客观选择题）与主观评测（开放式的主观问答）
    
  - **提示词工程：**
    
   ![屏幕截图 2024-04-23 224752](https://github.com/Hoder-zyf/InternLM/assets/73508057/215391a1-87fc-47fe-8933-03911eacfa5d)

    
  - **长文本评测（比如大海捞针）**：
    
    ![屏幕截图 2024-04-23 224824](https://github.com/Hoder-zyf/InternLM/assets/73508057/b43ad1ac-877f-40d7-9d80-95f514dbacca)

    
3. **OpenCompass评测流水线**：任务切分，并行
  
  ![屏幕截图 2024-04-23 224940](https://github.com/Hoder-zyf/InternLM/assets/73508057/c72815bc-845c-4548-bd50-d5d604ee4d31)

4. **工具架构**：
  
  ![](https://pic1.zhimg.com/80/v2-96d46a52ad4cd61e689d5395a5bf1b3c_720w.webp)
  
  - **模型层**：大模型评测所涉及的主要模型种类，OpenCompass 以基座模型和对话模型作为重点评测对象。
  - **能力层**：OpenCompass 从本方案从通用能力和特色能力两个方面来进行评测维度设计。在模型通用能力方面，从语言、知识、理解、推理、安全等多个能力维度进行评测。在特色能力方面，从长文本、代码、工具、知识增强等维度进行评测。
  - **方法层**：OpenCompass 采用客观评测与主观评测两种评测方式。客观评测能便捷地评估模型在具有确定答案（如选择，填空，封闭式问答等）的任务上的能力，主观评测能评估用户对模型回复的真实满意度，OpenCompass 采用基于模型辅助的主观评测和基于人类反馈的主观评测两种方式。
  - **工具层**：OpenCompass 提供丰富的功能支持自动化地开展大语言模型的高效评测。包括分布式评测技术，提示词工程，对接评测数据库，评测榜单发布，评测报告生成等诸多功能。
5. **客观评测**：
  
  针对具有**标准答案**的客观问题，我们可以我们可以通过使用**定量指标**比较模型的输出与标准答案的差异，并根据结果衡量模型的性能。同时，由于大语言模型输出自由度较高，**在评测阶段，我们需要对其输入和输出作一定的规范和设计，尽可能减少噪声输出在评测阶段的影响，才能对模型的能力有更加完整和客观的评价。 **为了更好地激发出模型在题目测试领域的能力，并引导模型按照一定的模板输出答案，OpenCompass 采用提示词工程 （prompt engineering）和语境学习（in-context learning）进行客观评测**。 在客观评测的具体实践中，我们通常采用下列两种方式进行模型输出结果的评测：
  
  - **判别式评测**：该评测方式基于**将问题与候选答案组合在一起，计算模型在所有组合上的困惑度（perplexity），并选择困惑度最小的答案作为模型的最终输出**例如，若模型在 问题? 答案1 上的困惑度为 0.1，在 问题? 答案2 上的困惑度为 0.2，最终我们会选择 答案1 作为模型的输出。
  - **生成式评测**：**该评测方式主要用于生成类任务，如语言翻译、程序生成、逻辑分析题等**具体实践时，**使用问题作为模型的原始输入，并留白答案区域待模型进行后续补全我们通常还需要对其输出进行后处理，以保证输出满足数据集的要求。**
6. **主观评测**：
  
  语言表达生动精彩，变化丰富，大量的场景和能力无法凭借客观指标进行评测。针对如模型安全和模型语言能力的评测，以人的主观感受为主的评测更能体现模型的真实能力，并更符合大模型的实际使用场景。 OpenCompass 采取的主观评测方案是**指借助受试者的主观判断对具有对话能力的大语言模型进行能力评测**在具体实践中，我们**提前基于模型的能力维度构建主观测试问题集合，并将不同模型对于同一问题的不同回复展现给受试者，收集受试者基于主观感受的评分**。由于主观测试成本高昂，本方案同时也**采用使用性能优异的大语言模拟人类进行主观打分**。在实际评测中，**本文将采用真实人类专家的主观评测与基于模型打分的主观评测相结合的方式开展模型能力评估。 在具体开展主观评测时，OpenComapss 采用单模型回复满意度统计和多模型满意度比较两种方式开展具体的评测工作。**
  
7. **OpenCompass评估模型的阶段**:
  
  - **配置**：这是整个工作流的起点。您需要配置整个评估过程，**选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。**
  - 推理与评估：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。**推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度**.这两个过程会被拆分为多个同时运行的“任务”以提高效率，但请注意，如果计算资源有限，这种策略可能会使评测变得更慢。
  - 可视化：评估完成后，OpenCompass 将结果整理成易读的表格，**并将其保存为 CSV 和 TXT 文件**。你也可以**激活飞书状态上报功能**，此后可以在飞书客户端中及时获得评测状态报告。  

## 一.基础作业

1. **选择开发机:** 50% A100 cuda11.7
  
2. **安装环境：**
  
  ```powershell
  studio-conda -o internlm-base -t opencompass
  source activate opencompass
  git clone -b 0.2.4 https://github.com/open-compass/opencompass
  cd opencompass
  pip install -e .
  ```
  
3. **解压评测数据集：**
  
  ```powershell
  cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
  # 解压
  unzip OpenCompassData-core-20231110.zip
  ```
  
4. **查看支持的数据集和模型**：`python tools/list_configs.py internlm ceval`
  
  **但是少了很多库，经过群里咨询，可能是安装环境的时候`pip install -e .`没有安装成功，我们再次运行`pip install -r requirements.txt`**
  
  之后运行`python tools/list_configs.py internlm ceval``就得到了正常的结果。
  
 ![屏幕截图 2024-04-23 180439](https://github.com/Hoder-zyf/InternLM/assets/73508057/f7af200c-365c-4fd7-aad3-bac89dfa3733)

  
5. **开始评测**：
  
  我们在--debug模式下进行评估，并检查是否存在问题：
  
  ```powershell
  python run.py
  --datasets ceval_gen \
  --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace 模型路径
  --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
  --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
  --model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
  --max-seq-len 1024 \  # 模型可以接受的最大序列长度
  --max-out-len 16 \  # 生成的最大 token 数
  --batch-size 2  \  # 批量大小
  --num-gpus 1  # 运行模型所需的 GPU 数量
  --debug
  ```
  
  - 遇到如下错误：
    
    1）**Error:mkl-service + Intel(R) MKL MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library**
    
    2)**04/23 18:09:25 - OpenCompass - ERROR - /root/opencompass/opencompass/tasks/openicl_eval.py - _score - 238 - Task [opencompass.models.huggingface.HuggingFace_Shanghai_AI_Laboratory_internlm2-chat-1_8b/ceval-computer_network]: No predictions found.**
    
   ![屏幕截图 2024-04-23 181220](https://github.com/Hoder-zyf/InternLM/assets/73508057/83e8144a-6883-434d-8324-45e79ffd7b94)

    
   ---

**解决方案：**

```powershell
pip install protobuf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
```

然后再次运行之前的评测代码就OK了
    
6. **结果展示:** **用的50% A100，花了大概16min的样子**
  
 ![屏幕截图 2024-04-23 183652](https://github.com/Hoder-zyf/InternLM/assets/73508057/77de3404-9411-4941-9d89-b1079c186a84)

  

## 二.进阶作业

1. **我们参考`OpenFinData`[CompassHub (opencompass.org.cn)](https://hub.opencompass.org.cn/dataset-detail/OpenFinData)创建了我们的数据库（选取了500条雪球网上经过人工打分的带情感标签言论数据）**
  
2. **在开发机上部署类似OpenFinData教程**
  
3. **README_OPENCOMPASS.md**：[InternLM/README_OPENCOMPASS.md at main · Hoder-zyf/InternLM (github.com)](https://github.com/Hoder-zyf/InternLM/blob/main/README_OPENCOMPASS.md)
  
4. **OpenCompass官网自定义数据集地址：** [CompassHub (opencompass.org.cn)](https://hub.opencompass.org.cn/dataset-detail/social_media_sentiment_analysis)
![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/4a02b96d-07e5-4706-9278-95beabe4059e)

5. **系统的bug:在开发机环境中，无法在GPU上使用OpenCompass的推理模块，只能在CPU上运行。**
![image](https://github.com/Hoder-zyf/InternLM/assets/73508057/ff8b3c6a-fea1-491a-ad2e-ca7737c244ad)
