# InternLM 第四课

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
  
  1. **将自我认知模型上传到OpenXLab**:[模型中心-OpenXLab](https://openxlab.org.cn/models/detail/amstrongzyf/HW4)
    
  2. **多模态微调**：由于时间原因以及算力点不足，没有进行。
    
    计划参考[LLaVA/docs/Finetune_Custom_Data.md 在 main ·刘浩天/LLaVA --- LLaVA/docs/Finetune_Custom_Data.md at main · haotian-liu/LLaVA (github.com)](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)，结合XTuner进行相关多模态微调。
