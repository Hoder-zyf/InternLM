# InternLM 第四课

## 0.课程笔记

1. **两种finetune范式**：
  
  ![屏幕截图 2024-04-16 105742](https://github.com/Hoder-zyf/InternLM/assets/73508057/afc8c63a-c439-4d6f-ab91-ee6c7f57f9ed)

  
2. **几种微调对比**：
  
  ![屏幕截图 2024-04-16 110237](https://github.com/Hoder-zyf/InternLM/assets/73508057/4df20061-aa6c-4503-8874-49020de266f6)

  
3. **XTuner**:轻量级：对于7B模型，微调最小显存为8GB(可以在colab上)
  
4. **XTuner**的两个优化技巧：
  
  - **Flash Attention**:将Attention的计算并行化，原始的Attention是N^2，改变后通过减少内存读写量，避免使用大规模中间矩阵，把N^2降到线性。
    
    ---
    
    **想象一下，你在一个图书馆里找书。传统注意力机制就像你在没有任何帮助的情况下，在成千上万的书架上一本一本地寻找你需要的书。这非常耗时，而且随着图书馆（序列长度）的增大，你需要花费的时间和记忆的书架数量（内存）也会急剧增加。**
    
    **现在，FlashAttention就像是有一个高效的图书管理系统。它知道哪些书在你的视野范围内（利用快速的SRAM），并且能够快速地告诉你哪些书是相关的（分块计算）。这样，你就不需要记住整个图书馆的布局，也不需要逐个检查每一个书架。当你需要找下一本书时，系统会迅速地帮你定位到相关区域，而不需要你走遍整个图书馆。**
    
    **此外，FlashAttention还像是一个智能助手，当你完成一次搜索后，它会记住你之前找到的书的位置和相关信息。所以，下次当你需要相同的信息时，它可以直接告诉你去哪里找，而不是让你重新开始搜索（重计算）。这样不仅节省了时间，也减少了你需要记住的信息量（减少了内存使用）。**
    
    **通过这种方式，FlashAttention使得在大型数据集（大型图书馆）中寻找信息（注意力计算）变得更加快速和高效，同时确保了你找到的信息是准确和相关的。**
    
  - **DeepSpeed ZeRO**:将训练过程中的参数、梯度和优化器状态切片保存，在多GPU训练时显著提升显存（需要手动开启 --deepspeed)
    
    ![屏幕截图 2024-04-16 111340](https://github.com/Hoder-zyf/InternLM/assets/73508057/e4d23ce1-8c28-40cb-b9bc-9cb2c6692475)

    
5. **多模态LLM原理**：
  
  ![屏幕截图 2024-04-16 112041](https://github.com/Hoder-zyf/InternLM/assets/73508057/1a33e0cb-6e44-4be3-b3d4-b3137d68556e)

  
6. **LLaVA**给LLM增加视觉能力的过程：分为Preetain和Finetune两个阶段：
  
  ![屏幕截图 2024-04-16 160627](https://github.com/Hoder-zyf/InternLM/assets/73508057/2ec0d10f-d045-47c6-b62b-7f5239cfbd34)

  
  Pretain类似LLM的预训练阶段（之后就已经有视觉能力，但是**无论用户问它什么，它都只会回答输入图片的标题。即，此时的模型只会给输入图像“写标题”**）；Finetune类似LLM的微调，使用`图片+复杂文本`数据对，来对Pretrain得到的结果进行进一步训练。
  
  **Finetune阶段可以借助GPT-4创建对应格式的微调json文件**

## 1.基础作业——XTuner 微调个人小助手认知

1. **开发机**：50% A100 cuda 11.7
  
2. **配置XTuner环境**：
  
  ```powershell
  
  studio-conda xtuner0.1.17
  # 如果你是在其他平台：
  # conda create --name xtuner0.1.17 python=3.10 -y
  
  # 激活环境
  conda activate xtuner0.1.17
  # 进入家目录 （~的意思是 “当前用户的home路径”）
  cd ~
  # 创建版本文件夹并进入，以跟随本教程
  mkdir -p /root/xtuner0117 && cd /root/xtuner0117
  
  # 拉取 0.1.17 的版本源码
  git clone -b v0.1.17  https://github.com/InternLM/xtuner
  # 无法访问github的用户请从 gitee 拉取:
  # git clone -b v0.1.15 https://gitee.com/Internlm/xtuner
  
  # 进入源码目录
  cd /root/xtuner0117/xtuner
  
  # 从源码安装 XTuner
  pip install -e '.[all]'
  ```
  
3. **数据集准备**：
  
  - 创建文件夹存储文件：
    
    ```powershell
    # 前半部分是创建一个文件夹，后半部分是进入该文件夹。
    mkdir -p /root/ft && cd /root/ft
    
    # 在ft这个文件夹里再创建一个存放数据的data文件夹
    mkdir -p /root/ft/data && cd /root/ft/data
    ```
    
  - 生成数据集：
    
    ```powershell
    # 创建 `generate_data.py` 文件
    touch /root/ft/data/generate_data.py
    ```
    
    然后填入我们生成数据的代码（参考Tutorial),之后运行代码，得到`personal_assistant.json`
    
4. **模型准备**：
  
  ```powershell
  # 创建目标文件夹，确保它存在。
  # -p选项意味着如果上级目录不存在也会一并创建，且如果目标文件夹已存在则不会报错。
  mkdir -p /root/ft/model
  
  # 复制内容到目标文件夹。-r选项表示递归复制整个文件夹。
  cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/ft/model/
  ```
  
5. **配置文件选择**：**配置文件（config），其实是一种用于定义和控制模型训练和测试过程中各个方面的参数和设置的工具。准备好的配置文件只要运行起来就代表着模型就开始训练或者微调了。**
  
  ```powershell
  # 列出所有内置配置文件
  # xtuner list-cfg
  
  # 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
  xtuner list-cfg -p internlm2_1_8b
  ```
  
  注：关于配置文件名的解释：`internlm2_1_8b_qlora_alpaca_e3`中internlm2_1_8b是模型名称，qlora是使用的算法，alpaca是数据集名称，e3指的是把数据集跑3次
  
  此处虽然我们自己构建了数据集，但是我们希望通过`QLoRA`进行微调，所以选择`internlm2_1_8b_qlora_alpaca_e3`这个配置文件。
  
  ```powershell
  # 创建一个存放 config 文件的文件夹
  mkdir -p /root/ft/config
  
  # 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
  xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config
  ```
  
6. **修改配置文件**：配置文件分成5个部分
  
  - **PART 1 Settings**：涵盖了模型基本设置，如预训练模型的选择、数据集信息和训练过程中的一些基本参数（如批大小、学习率等）
    
  - **PART 2 Model & Tokenizer**：指定了用于训练的模型和分词器的具体类型及其配置，包括预训练模型的路径和是否启用特定功能（如可变长度注意力），这是模型训练的核心组成部分。
    
  - **PART 3 Dataset & Dataloader**：描述了数据处理的细节，包括如何加载数据集、预处理步骤、批处理大小等，确保了模型能够接收到正确格式和质量的数据。
    
  - **PART 4 Scheduler & Optimizer**：配置了优化过程中的关键参数，如学习率调度策略和优化器的选择，这些是影响模型训练效果和速度的重要因素。
    
  - **PART 5 Runtime**：定义了训练过程中的额外设置，如日志记录、模型保存策略和自定义钩子等，以支持训练流程的监控、调试和结果的保存。
    
  
  一般只修改PART1~PART3,**修改的主要原因是我们修改了配置文件中规定的模型、数据集。**
  
  我们修改root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py`的代码：
  
  ```python
  # Copyright (c) OpenMMLab. All rights reserved.
  import torch
  from datasets import load_dataset
  from mmengine.dataset import DefaultSampler
  from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                              LoggerHook, ParamSchedulerHook)
  from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
  from peft import LoraConfig
  from torch.optim import AdamW
  from transformers import (AutoModelForCausalLM, AutoTokenizer,
                            BitsAndBytesConfig)
  
  from xtuner.dataset import process_hf_dataset
  from xtuner.dataset.collate_fns import default_collate_fn
  from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
  from xtuner.engine.hooks import (DatasetInfoHook, EvaluateChatHook,
                                   VarlenAttnArgsToMessageHubHook)
  from xtuner.engine.runner import TrainLoop
  from xtuner.model import SupervisedFinetune
  from xtuner.parallel.sequence import SequenceParallelSampler
  from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
  
  #######################################################################
  #                          PART 1  Settings                           #
  #######################################################################
  # Model
  pretrained_model_name_or_path = '/root/ft/model'
  use_varlen_attn = False
  
  # Data
  alpaca_en_path = '/root/ft/data/personal_assistant.json'
  prompt_template = PROMPT_TEMPLATE.default
  max_length = 1024
  pack_to_max_length = True
  
  # parallel
  sequence_parallel_size = 1
  
  # Scheduler & Optimizer
  batch_size = 1  # per_device
  accumulative_counts = 16
  accumulative_counts *= sequence_parallel_size
  dataloader_num_workers = 0
  max_epochs = 2
  optim_type = AdamW
  lr = 2e-4
  betas = (0.9, 0.999)
  weight_decay = 0
  max_norm = 1  # grad clip
  warmup_ratio = 0.03
  
  # Save
  save_steps = 300
  save_total_limit = 3  # Maximum checkpoints to keep (-1 means unlimited)
  
  # Evaluate the generation performance during the training
  evaluation_freq = 300
  SYSTEM = ''
  evaluation_inputs = ['请你介绍一下你自己', '你是谁', '你是我的小助手吗']
  
  #######################################################################
  #                      PART 2  Model & Tokenizer                      #
  #######################################################################
  tokenizer = dict(
      type=AutoTokenizer.from_pretrained,
      pretrained_model_name_or_path=pretrained_model_name_or_path,
      trust_remote_code=True,
      padding_side='right')
  
  model = dict(
      type=SupervisedFinetune,
      use_varlen_attn=use_varlen_attn,
      llm=dict(
          type=AutoModelForCausalLM.from_pretrained,
          pretrained_model_name_or_path=pretrained_model_name_or_path,
          trust_remote_code=True,
          torch_dtype=torch.float16,
          quantization_config=dict(
              type=BitsAndBytesConfig,
              load_in_4bit=True,
              load_in_8bit=False,
              llm_int8_threshold=6.0,
              llm_int8_has_fp16_weight=False,
              bnb_4bit_compute_dtype=torch.float16,
              bnb_4bit_use_double_quant=True,
              bnb_4bit_quant_type='nf4')),
      lora=dict(
          type=LoraConfig,
          r=64,
          lora_alpha=16,
          lora_dropout=0.1,
          bias='none',
          task_type='CAUSAL_LM'))
  
  #######################################################################
  #                      PART 3  Dataset & Dataloader                   #
  #######################################################################
  alpaca_en = dict(
      type=process_hf_dataset,
      dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
      tokenizer=tokenizer,
      max_length=max_length,
      dataset_map_fn=openai_map_fn,
      template_map_fn=dict(
          type=template_map_fn_factory, template=prompt_template),
      remove_unused_columns=True,
      shuffle_before_pack=True,
      pack_to_max_length=pack_to_max_length,
      use_varlen_attn=use_varlen_attn)
  
  sampler = SequenceParallelSampler \
      if sequence_parallel_size > 1 else DefaultSampler
  train_dataloader = dict(
      batch_size=batch_size,
      num_workers=dataloader_num_workers,
      dataset=alpaca_en,
      sampler=dict(type=sampler, shuffle=True),
      collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn))
  
  #######################################################################
  #                    PART 4  Scheduler & Optimizer                    #
  #######################################################################
  # optimizer
  optim_wrapper = dict(
      type=AmpOptimWrapper,
      optimizer=dict(
          type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
      clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
      accumulative_counts=accumulative_counts,
      loss_scale='dynamic',
      dtype='float16')
  
  # learning policy
  # More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
  param_scheduler = [
      dict(
          type=LinearLR,
          start_factor=1e-5,
          by_epoch=True,
          begin=0,
          end=warmup_ratio * max_epochs,
          convert_to_iter_based=True),
      dict(
          type=CosineAnnealingLR,
          eta_min=0.0,
          by_epoch=True,
          begin=warmup_ratio * max_epochs,
          end=max_epochs,
          convert_to_iter_based=True)
  ]
  
  # train, val, test setting
  train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
  
  #######################################################################
  #                           PART 5  Runtime                           #
  #######################################################################
  # Log the dialogue periodically during the training process, optional
  custom_hooks = [
      dict(type=DatasetInfoHook, tokenizer=tokenizer),
      dict(
          type=EvaluateChatHook,
          tokenizer=tokenizer,
          every_n_iters=evaluation_freq,
          evaluation_inputs=evaluation_inputs,
          system=SYSTEM,
          prompt_template=prompt_template)
  ]
  
  if use_varlen_attn:
      custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]
  
  # configure default hooks
  default_hooks = dict(
      # record the time of every iteration.
      timer=dict(type=IterTimerHook),
      # print log every 10 iterations.
      logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
      # enable the parameter scheduler.
      param_scheduler=dict(type=ParamSchedulerHook),
      # save checkpoint per `save_steps`.
      checkpoint=dict(
          type=CheckpointHook,
          by_epoch=False,
          interval=save_steps,
          max_keep_ckpts=save_total_limit),
      # set sampler seed in distributed evrionment.
      sampler_seed=dict(type=DistSamplerSeedHook),
  )
  
  # configure environment
  env_cfg = dict(
      # whether to enable cudnn benchmark
      cudnn_benchmark=False,
      # set multi process parameters
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
      # set distributed parameters
      dist_cfg=dict(backend='nccl'),
  )
  
  # set visualizer
  visualizer = None
  
  # set log level
  log_level = 'INFO'
  
  # load from which checkpoint
  load_from = None
  
  # whether to resume training from the loaded checkpoint
  resume = False
  
  # Defaults to use random seed and disable `deterministic`
  randomness = dict(seed=None, deterministic=False)
  
  # set log processor
  log_processor = dict(by_epoch=False)
  ```
  
7. **模型训练**：
  
  - **常规训练**：使用`xtuner train`训练，可以添加`--work -dir`指定保存路径。
    
    ```powershell
    # 指定保存路径
    xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train
    ```
    
  - **（可选）** 使用deepspeed加速训练：共有三种不同的 `deepspeed` 类型可进行选择，分别是 `deepspeed_zero1`, `deepspeed_zero2` 和 `deepspeed_zero3`
    
    `xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train_deepspeed --deepspeed deepspeed_zero2`
    
  - **训练结果:**
    
    300轮：
    
    ![屏幕截图 2024-04-13 131633](https://github.com/Hoder-zyf/InternLM/assets/73508057/53060113-fb8d-4739-b253-8c8b1aeee3f4)


    
    600轮：
    
    ![屏幕截图 2024-04-13 131909](https://github.com/Hoder-zyf/InternLM/assets/73508057/2e693196-1708-4b30-a7b6-e0e5810dbf4f)

    
    **很明显，出现了过拟合，应对方法：**
    
    1. **减少保存权重文件的间隔并增加权重文件保存的上限**：这个方法实际上就是通过降低间隔结合评估问题的结果，从而找到最优的权重文。我们可以每隔100个批次来看什么时候模型已经学到了这部分知识但是还保留着基本的常识，什么时候已经过拟合严重只会说一句话了。但是由于再配置文件有设置权重文件保存数量的上限，因此同时将这个上限加大也是非常必要的。
    2. **增加常规的对话数据集从而稀释原本数据的占比**：这个方法其实就是希望我们正常用对话数据集做指令微调的同时还加上一部分的数据集来让模型既能够学到正常对话，但是在遇到特定问题时进行特殊化处理。比如说我在一万条正常的对话数据里混入两千条和小助手相关的数据集，这样模型同样可以在不丢失对话能力的前提下学到剑锋大佬的小助手这句话。这种其实是比较常见的处理方式，大家可以自己动手尝试实践一下。
  - **模型续航指南：**（训练中断，在原有指令下添加`--resume{checkpoint_path}`实现继续训练。
    
8. **模型转换**：将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 Huggingface 格式文件
  
  ```powershell
  # 创建一个保存转换后 Huggingface 格式的文件夹
  mkdir -p /root/ft/huggingface
  
  # 模型转换
  # xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
  xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
  ```
  
  **此时，huggingface 文件夹即为我们平时所理解的所谓 “LoRA 模型文件(相当于一个额外的adapter层）”**
  
9. **模型整合**：把adapter层和原始的基座模型组合起来用：
  
  注：**在使用前我们需要准备好三个地址，包括原模型的地址、训练好的 adapter 层的地址（转为 Huggingface 格式后保存的部分）以及最终保存的地址。**
  
  ```powershell
  # 创建一个名为 final_model 的文件夹存储整合后的模型文件
  mkdir -p /root/ft/final_model
  
  # 解决一下线程冲突的 Bug 
  export MKL_SERVICE_FORCE_INTEL=1
  
  # 进行模型整合
  # xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
  xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
  ```
  
10. **对话测试**
  
  - 与微调后的模型进行对话（final_model):`xtuner chat /root/ft/final_model --prompt-template internlm2_chat`
    
    **BUG??**:有点奇怪，我问了`你是谁`之类的，但是回答的是`我是上海AI实验室书生·浦语的1.8B大模型哦</s>`(我设置的是`是猪猪侠的小助手`，不太清楚具体原因是什么...)
    
    ---
    
    **<span style="color: red;">可能是“猪猪侠”违反了一些规则？我改成了"amstrongzyf"就正常显示</span>** `我是amstrongzyf的小助手内在是上海AI实验室书生·浦语的1.8B大模型哦</s>`
    
  - 与基准模型对话:`xtuner chat /root/ft/model --prompt-template internlm2_chat`
    
  - 与adapter层进行对话（便于选择不同的adapter)
    
    ```powershell
    # 使用 --adapter 参数与完整的模型进行对话
    xtuner chat /root/ft/model --adapter /root/ft/huggingface --prompt-template internlm2_chat
    ```
    
11. **Web demo部署**
  
  - 下载streamlit库：pip install streamlit==1.24.0
    
  - 下载InternLM项目代码：
    
    ```powershell
    # 创建存放 InternLM 文件的代码
    mkdir -p /root/ft/web_demo && cd /root/ft/web_demo
    
    # 拉取 InternLM 源文件
    git clone https://github.com/InternLM/InternLM.git
    
    # 进入该库中
    cd /root/ft/web_demo/InternLM
    ```
    
  -  修改`/root/ft/web_demo/InternLM/chat/web_demo.py`中的代码
    
  -  在本地机器上进行端口映射：`ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38951`
    
  -  在开发机中：`streamlit run /root/ft/web_demo/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006`
    
  -  打开本地机器的`http://127.0.0.1:6006`
    
  -  截图如下：
    
   ![屏幕截图 2024-04-13 143826](https://github.com/Hoder-zyf/InternLM/assets/73508057/1d8760b0-c5af-409c-a13d-f60b8c25e488)

    
   **非常神奇，我这里用的是“猪猪侠”，在前面不管是用adapter拼接还是直接用final_model都不会有标准输出，但是映射到网页之后就直接标准输出了。**

    
  
  ## 二.进阶作业
  
 1.**将自我认知模型上传到OpenXLab**:[模型中心-OpenXLab](https://openxlab.org.cn/models/detail/amstrongzyf/HW4)
  
 2.**在OpenXLab上创造了相关应用**：[应用中心-OpenXLab](https://openxlab.org.cn/apps/detail/amstrongzyf/xtuner_hw4_deploy)
  
     唯一需要注意的点：是`requirements.txt`不是`requirement.txt`
  
 3.**多模态微调**：
  
  - **使用刚刚创建的xtuner0.1.17环境**：conda activate xtuner0.1.17
    
  - **Pretain阶段**：使用大量的`图片+简单文本（caption, 即图片标题）`数据对，使LLM理解图像中的**普遍特征**。即，对大量的图片进行**粗看**，但是由于Pretain阶段对硬件要求较高，我们这里直接使用Pretrain阶段的产物——`iter_2181.pth`文件。
    
  - **FineTune阶段**：
    
    1. 制作训练数据：
      
      ```powershell
      cd ~ 
      git clone https://github.com/InternLM/tutorial -b camp2 
      cd tutorial
      
      python /root/tutorial/xtuner/llava/llava_data/repeat.py \
        -i /root/tutorial/xtuner/llava/llava_data/unique_data.json \
        -o /root/tutorial/xtuner/llava/llava_data/repeated_data.json \
        -n 200
      ```
      
    2. 修改配置文件：按照教程修改配置文件
      
      关于`openai/clip-vit-large-patch14-336`：**这个模型结合了视觉和语言的能力，通过对比语言-图像预训练（CLIP）框架来实现其功能。它使用了Vision Transformer（ViT）架构中的ViT-L/14作为图像编码器，并采用了掩码自注意力Transformer作为文本编码器**。
      
      **百度贴吧的解答：是clip模型，给图片和提示词建立关联用的东西**
      
    3. 开始finetune:`cd /root/tutorial/xtuner/llava/
      xtuner train /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py --deepspeed deepspeed_zero2`
      
    4. 训练结果截图：
      
      ![屏幕截图 2024-04-16 165310](https://github.com/Hoder-zyf/InternLM/assets/73508057/2fb970e7-a7cb-42d2-b7fb-32877a486ba4)

      
  - 对比Finetune前后性能差异：
    
    1. **微调前：**
      
      - 默认图片
        
      
      ```powershell
      # 如果在 numpy 之前导入了 torch，那么这里的子进程将获得一个 GNU 线程层（即使父进程没有定义变量）
      # 但是如果 numpy 在 Torch 之前被导入，子进程将获得一个 INTEL 线程层，这种情况会导致线程之间打架
      # 以下两行代码可以解决这个小bug
      export MKL_SERVICE_FORCE_INTEL=1
      export MKL_THREADING_LAYER=GNU
      
      # pth转huggingface
      xtuner convert pth_to_hf \
        llava_internlm2_chat_1_8b_clip_vit_large_p14_336_e1_gpu8_pretrain \
        /root/share/new_models/xtuner/iter_2181.pth \
        /root/tutorial/xtuner/llava/llava_data/iter_2181_hf
      
      # 启动！
      xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
        --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
        --llava /root/tutorial/xtuner/llava/llava_data/iter_2181_hf \
        --prompt-template internlm2_chat \
        --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
      ```
      
     ![屏幕截图 2024-04-16 223325](https://github.com/Hoder-zyf/InternLM/assets/73508057/f2a389b4-8fe4-4f0e-8a08-61829cfca6f8)


      
      - cat.jpg(自己上传的图片)
        
        ![cat](https://github.com/Hoder-zyf/InternLM/assets/73508057/6552f25e-66e3-4207-babd-8fd9181725e0)

        
        ![屏幕截图 2024-04-16 175540](https://github.com/Hoder-zyf/InternLM/assets/73508057/37af7e4a-eadc-4b55-afd8-6883a681ebb1)

        
        **微调前的模型确实可以比较清楚的发现这个基础模型只能输出这个图片在讲什么，准确率还是不错的**
        
      
      ---
      
    2. **微调后：**
      
      - 默认图片
        
      
      ```powershell
      # 解决小bug
      export MKL_SERVICE_FORCE_INTEL=1
      export MKL_THREADING_LAYER=GNU
      
      # pth转huggingface
      xtuner convert pth_to_hf \
        /root/tutorial/xtuner/llava/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy.py \
        /root/tutorial/xtuner/llava/work_dirs/llava_internlm2_chat_1_8b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune_copy/iter_1200.pth \
        /root/tutorial/xtuner/llava/llava_data/iter_1200_hf
      
      # 启动！
      xtuner chat /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b \
        --visual-encoder /root/share/new_models/openai/clip-vit-large-patch14-336 \
        --llava /root/tutorial/xtuner/llava/llava_data/iter_1200_hf \
        --prompt-template internlm2_chat \
        --image /root/tutorial/xtuner/llava/llava_data/test_img/oph.jpg
      ```
     
    ![屏幕截图 2024-04-16 171620](https://github.com/Hoder-zyf/InternLM/assets/73508057/5a776289-ae66-40d1-8991-f320fe0fddc8)

      
      **我们可以发现微调之后有了明显的性能提升，但是当问及一些与图片无关的问题时，回答也不太行（过拟合了吧）**
      
      - cat.jpg
        
       ![屏幕截图 2024-04-16 174459](https://github.com/Hoder-zyf/InternLM/assets/73508057/a48ff990-7e4a-4205-8e7e-94dcb7320b81)

        
        **这个结果就很一般了，感觉还是训练数据太单一的问题**
