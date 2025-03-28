'''
中英文语言模型训练程序 - 支持分布式训练和训练进度保存/恢复

本程序用于训练中英文双语语言模型，支持以下特性：
1. 支持分布式训练（使用PyTorch DDP，支持多GPU和多节点）
2. 支持训练进度保存和恢复
3. 支持中英文双语数据处理
4. 支持模型测试和交互式翻译

使用方法：

1. 单进程训练：
   python train_ddp.py --模式 训练

2. 分布式训练（使用torch.multiprocessing）：
   python train_ddp.py --模式 训练 --分布式

3. 使用torchrun启动分布式训练（推荐，可跨节点）：
   # 单机多GPU
   torchrun --nproc_per_node=GPU数量 train_ddp.py --模式 训练
   
   # 多机多GPU
   # 在第一台机器上:
   torchrun --nnodes=2 --node_rank=0 --master_addr=主节点IP --master_port=端口 --nproc_per_node=GPU数量 train_ddp.py --模式 训练
   
   # 在第二台机器上:
   torchrun --nnodes=2 --node_rank=1 --master_addr=主节点IP --master_port=端口 --nproc_per_node=GPU数量 train_ddp.py --模式 训练

4. 从检查点恢复训练：
   python train_ddp.py --模式 训练 --恢复训练 最新检查点.pth
   
   # 或使用torchrun恢复：
   torchrun --nproc_per_node=GPU数量 train_ddp.py --模式 训练 --恢复训练 最新检查点.pth

5. 测试模型：
   python train_ddp.py --模式 测试 --加载模型 最佳模型.pth

6. 交互式翻译测试：
   python train_ddp.py --模式 交互 --加载模型 最佳模型.pth
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, concatenate_datasets
import sentencepiece as spm
from tqdm import tqdm
import math
import argparse
import json
from collections import Counter
import time
import logging
from datetime import datetime
import torch.distributed as dist

# ==================== 配置类 ====================
class 模型配置:
    def __init__(self):
        # 数据配置
        self.数据集名称列表 = ["wmt19", "opus100", "news_commentary"]
        self.最大样本数 = 500000    # 每个数据集最大加载样本数
        self.验证集比例 = 0.01      # 验证集占总数据的比例
        
        # 词汇表配置
        self.词汇表大小 = 32000     # 词汇表总大小
        self.最大序列长度 = 128     # 输入序列最大长度
        
        # 模型配置
        self.模型维度 = 768         # 模型隐藏层维度
        self.注意力头数 = 8         # 多头注意力头数
        self.Transformer层数 = 16   # Transformer编码器层数
        self.前馈网络维度 = 2048    # 前馈网络中间层维度
        self.丢弃率 = 0.1           # Dropout率
        
        # 训练配置
        self.批大小 = 64           # 训练批大小
        self.训练轮数 = 20          # 总训练轮数
        self.学习率 = 5e-4         # 初始学习率
        self.权重衰减 = 0.01        # 优化器权重衰减
        self.梯度裁剪 = 1.0         # 梯度裁剪阈值
        self.早停耐心值 = 3        # 验证损失不改善的轮次数
        
        # 日志配置
        self.日志级别 = logging.INFO  # 日志级别
        self.日志文件 = "训练日志.log" # 日志文件名
        self.日志格式 = "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式
        self.TensorBoard目录 = "runs"  # TensorBoard日志目录
        
        # 其他配置
        self.设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备
        self.随机种子 = 42          # 随机种子
        self.模型保存目录 = "bilingual_model"  # 模型保存路径
        self.日志间隔 = 100         # 训练日志打印间隔
        self.分词模型前缀 = "bpe"   # SentencePiece模型前缀
        
        # 分布式训练配置
        self.分布式训练 = False     # 是否使用分布式训练
        self.世界大小 = 1           # 分布式训练的进程数
        self.主进程 = True          # 是否是主进程
        self.本地排名 = 0           # 本地进程排名
        self.分布式端口 = "29500"   # 分布式训练端口
    
    def __getstate__(self):
        """自定义序列化行为，将不可序列化的对象转换为可序列化的形式"""
        状态 = self.__dict__.copy()
        # 将torch.device对象转换为字符串
        if '设备' in 状态 and isinstance(状态['设备'], torch.device):
            状态['设备'] = str(状态['设备'])
        return 状态
    
    def __setstate__(self, 状态):
        """自定义反序列化行为，将可序列化的形式转换回原始对象"""
        # 将设备字符串转换回torch.device对象
        if '设备' in 状态 and isinstance(状态['设备'], str):
            状态['设备'] = torch.device(状态['设备'])
        self.__dict__.update(状态)
        
    def 保存配置(self, 路径):
        """将配置保存到JSON文件"""
        # 创建一个可序列化的配置字典
        可序列化配置 = {}
        for 键, 值 in vars(self).items():
            # 特殊处理不可序列化的类型
            if isinstance(值, torch.device):
                可序列化配置[键] = str(值)  # 将设备对象转换为字符串
            elif isinstance(值, logging.Logger):
                可序列化配置[键] = 值.name  # 将日志器对象转换为名称
            else:
                可序列化配置[键] = 值
        
        with open(路径, 'w', encoding='utf-8') as 文件:
            json.dump(可序列化配置, 文件, indent=4, ensure_ascii=False)
    
    @classmethod
    def 加载配置(cls, 路径):
        """从JSON文件加载配置"""
        配置实例 = cls()
        with open(路径, 'r', encoding='utf-8') as 文件:
            配置字典 = json.load(文件)
        for 键, 值 in 配置字典.items():
            # 特殊处理设备属性
            if 键 == '设备' and isinstance(值, str):
                值 = torch.device(值)
            setattr(配置实例, 键, 值)
        return 配置实例

# ==================== 日志系统 ====================
class 定时刷新文件处理器(logging.FileHandler):
    """自定义文件处理器，定期刷新缓冲区"""
    def __init__(self, filename, mode='a', encoding=None, delay=False, flush_interval=60):
        super().__init__(filename, mode, encoding, delay)
        self.flush_interval = flush_interval  # 刷新间隔（秒）
        self.last_flush = time.time()
        
    def emit(self, record):
        """发送日志记录时检查是否需要刷新"""
        super().emit(record)
        current_time = time.time()
        if current_time - self.last_flush >= self.flush_interval:
            self.flush()
            self.last_flush = current_time

def 初始化日志系统(配置):
    """配置日志系统"""
    # 创建日志目录
    os.makedirs(配置.模型保存目录, exist_ok=True)
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(配置.日志级别)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(配置.日志格式)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(配置.日志级别)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 只在主进程添加文件处理器
    if not 配置.分布式训练 or 配置.主进程:
        # 添加定时刷新文件处理器（每60秒刷新一次）
        file_handler = 定时刷新文件处理器(
            os.path.join(配置.模型保存目录, 配置.日志文件),
            encoding='utf-8',
            flush_interval=60  # 每60秒刷新一次
        )
        file_handler.setLevel(配置.日志级别)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 记录初始日志
        logging.info(f"日志系统初始化完成，日志文件: {配置.日志文件}，每60秒自动刷新")
        
        # 创建TensorBoard写入器
        tensorboard_dir = os.path.join(配置.模型保存目录, 配置.TensorBoard目录)
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_writer = SummaryWriter(
            log_dir=os.path.join(tensorboard_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        # 非主进程返回None作为tensorboard_writer
        tensorboard_writer = None
        
    return tensorboard_writer

# ==================== 数据处理部分 ====================
class 双语数据集(Dataset):
    def __init__(self, 数据集, 分词器, 配置, 语言对=("en", "zh")):
        """
        初始化双语数据集
        参数:
            数据集: 原始数据集
            分词器: 文本分词器
            配置: 模型配置
            语言对: (源语言, 目标语言) 元组
        """
        self.原始数据集 = 数据集
        self.分词器 = 分词器
        self.配置 = 配置
        self.源语言, self.目标语言 = 语言对
        
    def __len__(self):
        return len(self.原始数据集)
    
    def __getitem__(self, 索引):
        """获取单个训练样本"""
        样本 = self.原始数据集[索引]
        源文本 = 样本['translation'][self.源语言]
        目标文本 = 样本['translation'][self.目标语言]
        
        # 合并中英文文本用于语言模型训练，用[SEP]分隔
        合并文本 = f"{源文本} [SEP] {目标文本}"
        
        # 分词并转换为ID
        分词ID列表 = self.分词器.编码(合并文本)
        
        # 序列截断或填充
        if len(分词ID列表) > self.配置.最大序列长度:
            分词ID列表 = 分词ID列表[:self.配置.最大序列长度-1] + [self.分词器.结束标记ID]
        else:
            填充长度 = self.配置.最大序列长度 - len(分词ID列表)
            分词ID列表 = 分词ID列表 + [self.分词器.填充标记ID] * 填充长度
        
        # 输入序列和标签 (语言模型任务是预测下一个token)
        输入序列 = 分词ID列表[:-1]
        目标序列 = 分词ID列表[1:]
        
        return torch.tensor(输入序列, dtype=torch.long), torch.tensor(目标序列, dtype=torch.long)

# ==================== 分词器部分 ====================
class 双语分词器:
    def __init__(self, 配置):
        """初始化双语分词器"""
        self.配置 = 配置
        self.分词模型 = None
        self.词汇表大小 = 配置.词汇表大小
        
        # 特殊标记定义
        self.填充标记ID = 0
        self.未知标记ID = 1
        self.开始标记ID = 2
        self.结束标记ID = 3
        self.分隔标记ID = 4  # 用于分隔中英文
        
        # 特殊标记映射表
        self.特殊标记表 = {
            "<pad>": self.填充标记ID,
            "<unk>": self.未知标记ID,
            "<bos>": self.开始标记ID,
            "<eos>": self.结束标记ID,
            "[SEP]": self.分隔标记ID
        }
    
    def 训练分词器(self, 文本文件路径):
        """训练SentencePiece分词器"""
        spm.SentencePieceTrainer.train(
            input=文本文件路径,
            model_prefix=self.配置.分词模型前缀,
            vocab_size=self.配置.词汇表大小,
            user_defined_symbols=["<pad>", "<bos>", "<eos>", "[SEP]"],
            model_type="bpe",  # 使用BPE算法
            pad_id=self.填充标记ID,
            unk_id=self.未知标记ID,
            bos_id=self.开始标记ID,
            eos_id=self.结束标记ID,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>"
        )
        
        # 加载训练好的分词模型
        self.分词模型 = spm.SentencePieceProcessor()
        self.分词模型.load(f"{self.配置.分词模型前缀}.model")
        logging.info(f"分词器训练完成，词汇表大小: {self.分词模型.get_piece_size()}")
    
    def 编码(self, 文本):
        """将文本编码为token ID列表"""
        if self.分词模型 is None:
            raise ValueError("分词器尚未训练，请先调用训练分词器方法")
        return [self.开始标记ID] + self.分词模型.encode(文本) + [self.结束标记ID]
    
    def 解码(self, 分词ID列表):
        """将token ID列表解码为文本"""
        if self.分词模型 is None:
            raise ValueError("分词器尚未训练，请先调用训练分词器方法")
        return self.分词模型.decode(分词ID列表)
    
    def 保存分词器(self, 保存路径):
        """保存分词器配置"""
        if self.分词模型 is None:
            raise ValueError("分词器尚未训练，请先调用训练分词器方法")
        
        # 保存配置信息
        配置信息 = {
            "分词模型文件": f"{self.配置.分词模型前缀}.model",
            "词汇表大小": self.词汇表大小,
            "特殊标记表": self.特殊标记表
        }
        with open(保存路径, 'w', encoding='utf-8') as 文件:
            json.dump(配置信息, 文件, ensure_ascii=False, indent=4)
        logging.info(f"分词器配置已保存到 {保存路径}")
    
    @classmethod
    def 加载分词器(cls, 配置, 加载路径):
        """从文件加载分词器"""
        with open(加载路径, 'r', encoding='utf-8') as 文件:
            分词器配置 = json.load(文件)
        
        分词器实例 = cls(配置)
        分词器实例.分词模型 = spm.SentencePieceProcessor()
        分词器实例.分词模型.load(分词器配置["分词模型文件"])
        分词器实例.词汇表大小 = 分词器配置["词汇表大小"]
        分词器实例.特殊标记表 = 分词器配置["特殊标记表"]
        logging.info(f"从 {加载路径} 加载分词器成功")
        return 分词器实例

# ==================== 模型部分 ====================
class 位置编码(nn.Module):
    def __init__(self, 模型维度, 最大长度=5000):
        """位置编码层实现"""
        super().__init__()
        self.丢弃层 = nn.Dropout(p=0.1)
        
        # 计算位置编码
        位置 = torch.arange(最大长度).unsqueeze(1)
        除数项 = torch.exp(torch.arange(0, 模型维度, 2) * (-math.log(10000.0) / 模型维度))
        位置编码 = torch.zeros(最大长度, 1, 模型维度)
        位置编码[:, 0, 0::2] = torch.sin(位置 * 除数项)
        位置编码[:, 0, 1::2] = torch.cos(位置 * 除数项)
        self.register_buffer('位置编码', 位置编码)

    def forward(self, 输入张量):
        """前向传播"""
        输入张量 = 输入张量 + self.位置编码[:输入张量.size(0)]
        return self.丢弃层(输入张量)

class 双语语言模型(nn.Module):
    def __init__(self, 配置, 分词器):
        """初始化双语语言模型"""
        super().__init__()
        self.配置 = 配置
        self.分词器 = 分词器
        
        # 词嵌入层
        self.词嵌入 = nn.Embedding(配置.词汇表大小, 配置.模型维度)
        
        # 位置编码层
        self.位置编码 = 位置编码(配置.模型维度, 配置.最大序列长度)
        
        # Transformer编码器层
        编码器层 = nn.TransformerEncoderLayer(
            d_model=配置.模型维度,
            nhead=配置.注意力头数,
            dim_feedforward=配置.前馈网络维度,
            dropout=配置.丢弃率,
            batch_first=False
        )
        self.Transformer编码器 = nn.TransformerEncoder(编码器层, 配置.Transformer层数)
        
        # 输出层
        self.输出层 = nn.Linear(配置.模型维度, 配置.词汇表大小)
        
        # 初始化权重
        self.初始化权重()
        logging.info(f"模型初始化完成，总参数量: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def 初始化权重(self):
        """初始化模型权重"""
        初始化范围 = 0.1
        self.词嵌入.weight.data.uniform_(-初始化范围, 初始化范围)
        self.输出层.bias.data.zero_()
        self.输出层.weight.data.uniform_(-初始化范围, 初始化范围)
    
    def forward(self, 输入序列, 序列掩码=None):
        """模型前向传播"""
        # 词嵌入
        输入序列 = self.词嵌入(输入序列) * math.sqrt(self.配置.模型维度)
        输入序列 = self.位置编码(输入序列)
        
        # Transformer编码
        输出序列 = self.Transformer编码器(输入序列, 序列掩码)
        
        # 输出层
        输出序列 = self.输出层(输出序列)
        return 输出序列
    
    def 生成序列掩码(self, 序列长度):
        """生成自回归掩码，防止模型看到未来信息"""
        掩码 = (torch.triu(torch.ones(序列长度, 序列长度)) == 1).transpose(0, 1)
        掩码 = 掩码.float().masked_fill(掩码 == 0, float('-inf')).masked_fill(掩码 == 1, float(0.0))
        return 掩码.to(self.配置.设备)

# ==================== 训练部分 ====================
class 模型训练器:
    def __init__(self, 模型, 训练数据加载器, 验证数据加载器, 配置, tensorboard_writer):
        """初始化训练器"""
        self.模型 = 模型
        self.训练数据加载器 = 训练数据加载器
        self.验证数据加载器 = 验证数据加载器
        self.配置 = 配置
        self.tensorboard_writer = tensorboard_writer
        
        # 在分布式训练中，模型可能是DistributedDataParallel包装过的
        # 尝试从原始模型(module)或配置中获取分词器的填充标记ID
        填充标记ID = 配置.填充标记ID if hasattr(配置, "填充标记ID") else 0
        if hasattr(模型, "分词器"):
            填充标记ID = 模型.分词器.填充标记ID
        elif hasattr(模型, "module") and hasattr(模型.module, "分词器"):
            填充标记ID = 模型.module.分词器.填充标记ID
            
        # 训练组件
        self.损失函数 = nn.CrossEntropyLoss(ignore_index=填充标记ID)
        self.优化器 = optim.AdamW(
            模型.parameters(), 
            lr=配置.学习率, 
            weight_decay=配置.权重衰减
        )
        self.学习率调度器 = optim.lr_scheduler.StepLR(self.优化器, step_size=5, gamma=0.5)
        
        # 训练状态
        self.当前轮数 = 0
        self.最佳验证损失 = float('inf')
        self.早停计数器 = 0
        self.训练损失记录 = []
        self.验证损失记录 = []
        self.训练开始时间 = time.time()
        
        # 创建模型保存目录
        if not 配置.分布式训练 or 配置.主进程:
            os.makedirs(配置.模型保存目录, exist_ok=True)
            logging.info(f"训练器初始化完成，设备: {配置.设备}")
    
    def 训练单轮(self):
        """训练一个epoch"""
        self.模型.train()
        总损失 = 0
        总样本数 = 0
        总步数 = len(self.训练数据加载器)
        
        # 在分布式训练中，设置sampler的epoch
        if self.配置.分布式训练 and hasattr(self.训练数据加载器.sampler, 'set_epoch'):
            self.训练数据加载器.sampler.set_epoch(self.当前轮数)
            logging.debug(f"设置训练采样器epoch为: {self.当前轮数}")
        
        # 只在主进程显示进度条
        if not self.配置.分布式训练 or self.配置.主进程:
            进度条 = tqdm(self.训练数据加载器, desc=f"训练轮次 {self.当前轮数 + 1}/{self.配置.训练轮数}")
            进度条迭代器 = 进度条
        else:
            进度条迭代器 = self.训练数据加载器
            
        for 步数, (输入序列, 目标序列) in enumerate(进度条迭代器):
            # 数据准备
            输入序列 = 输入序列.to(self.配置.设备).transpose(0, 1)  # [序列长度, 批大小]
            目标序列 = 目标序列.to(self.配置.设备).transpose(0, 1)
            当前批大小 = 输入序列.size(1)
            
            # 梯度清零
            self.优化器.zero_grad()
            
            # 生成序列掩码 - 处理DistributedDataParallel包装过的模型
            if hasattr(self.模型, "module"):
                序列掩码 = self.模型.module.生成序列掩码(输入序列.size(0))
            else:
                序列掩码 = self.模型.生成序列掩码(输入序列.size(0))
            
            # 前向传播
            模型输出 = self.模型(输入序列, 序列掩码)
            
            # 计算损失
            损失 = self.损失函数(模型输出.view(-1, self.配置.词汇表大小), 
                              目标序列.reshape(-1))
            
            # 反向传播
            损失.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.模型.parameters(), self.配置.梯度裁剪)
            
            # 参数更新
            self.优化器.step()
            
            # 记录统计信息
            总损失 += 损失.item() * 当前批大小
            总样本数 += 当前批大小
            
            # 计算并记录当前平均损失
            当前平均损失 = 总损失 / 总样本数
            当前困惑度 = math.exp(当前平均损失)
            
            # 只在主进程更新进度条和记录日志
            if not self.配置.分布式训练 or self.配置.主进程:
                # 更新进度条
                if hasattr(进度条迭代器, 'set_postfix'):
                    进度条迭代器.set_postfix({
                        '损失': f"{当前平均损失:.4f}",
                        '困惑度': f"{当前困惑度:.2f}",
                        '学习率': f"{self.优化器.param_groups[0]['lr']:.6f}"
                    })
                
                # 记录TensorBoard数据
                if self.tensorboard_writer is not None:
                    global_step = self.当前轮数 * 总步数 + 步数
                    self.tensorboard_writer.add_scalar('训练/批次损失', 损失.item(), global_step)
                    self.tensorboard_writer.add_scalar('训练/批次困惑度', math.exp(损失.item()), global_step)
                    self.tensorboard_writer.add_scalar('训练/学习率', self.优化器.param_groups[0]['lr'], global_step)
                
                # 定期日志记录
                if 步数 % self.配置.日志间隔 == 0:
                    logging.info(
                        f"轮次 {self.当前轮数 + 1}/{self.配置.训练轮数} | "
                        f"步数 {步数}/{总步数} | "
                        f"批次损失: {损失.item():.4f} | "
                        f"批次困惑度: {math.exp(损失.item()):.2f} | "
                        f"学习率: {self.优化器.param_groups[0]['lr']:.6f}"
                    )
        
        # 在分布式训练中，收集所有进程的损失
        if self.配置.分布式训练:
            # 创建张量用于收集总损失和总样本数
            总损失张量 = torch.tensor([总损失], device=self.配置.设备)
            总样本数张量 = torch.tensor([总样本数], device=self.配置.设备)
            
            # 收集所有进程的数据
            dist.all_reduce(总损失张量, op=dist.ReduceOp.SUM)
            dist.all_reduce(总样本数张量, op=dist.ReduceOp.SUM)
            
            总损失 = 总损失张量.item()
            总样本数 = 总样本数张量.item()

            # 添加同步点，确保所有进程都完成了数据处理
            dist.barrier()
        
        # 计算并记录epoch平均损失
        epoch平均损失 = 总损失 / 总样本数
        epoch困惑度 = math.exp(epoch平均损失)
        self.训练损失记录.append(epoch平均损失)
        
        # 只在主进程记录TensorBoard数据和日志
        if not self.配置.分布式训练 or self.配置.主进程:
            # 记录TensorBoard数据
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('训练/轮次损失', epoch平均损失, self.当前轮数)
                self.tensorboard_writer.add_scalar('训练/轮次困惑度', epoch困惑度, self.当前轮数)
            
            logging.info(
                f"训练轮次 {self.当前轮数 + 1} 完成 | "
                f"平均损失: {epoch平均损失:.4f} | "
                f"困惑度: {epoch困惑度:.2f} | "
                f"耗时: {time.time() - self.训练开始时间:.2f}秒"
            )
        
        return epoch平均损失
    
    def 验证模型(self):
        """在验证集上评估模型"""
        self.模型.eval()
        总损失 = 0
        总样本数 = 0
        
        # 在分布式训练中，设置验证sampler的epoch
        if self.配置.分布式训练 and hasattr(self.验证数据加载器.sampler, 'set_epoch'):
            self.验证数据加载器.sampler.set_epoch(self.当前轮数)
            logging.debug(f"设置验证采样器epoch为: {self.当前轮数}")
        
        with torch.no_grad():
            # 只在主进程显示进度条
            if not self.配置.分布式训练 or self.配置.主进程:
                进度条 = tqdm(self.验证数据加载器, desc="验证中")
                进度条迭代器 = 进度条
            else:
                进度条迭代器 = self.验证数据加载器
            
            for 输入序列, 目标序列 in 进度条迭代器:
                输入序列 = 输入序列.to(self.配置.设备).transpose(0, 1)
                目标序列 = 目标序列.to(self.配置.设备).transpose(0, 1)
                当前批大小 = 输入序列.size(1)
                
                # 生成序列掩码 - 处理DistributedDataParallel包装过的模型
                if hasattr(self.模型, "module"):
                    序列掩码 = self.模型.module.生成序列掩码(输入序列.size(0))
                else:
                    序列掩码 = self.模型.生成序列掩码(输入序列.size(0))
                
                # 前向传播
                模型输出 = self.模型(输入序列, 序列掩码)
                
                # 计算损失
                损失 = self.损失函数(模型输出.view(-1, self.配置.词汇表大小), 
                                  目标序列.reshape(-1))
                
                # 记录统计信息
                总损失 += 损失.item() * 当前批大小
                总样本数 += 当前批大小
                
                # 只在主进程更新进度条
                if (not self.配置.分布式训练 or self.配置.主进程) and hasattr(进度条迭代器, 'set_postfix'):
                    进度条迭代器.set_postfix({
                        '验证损失': f"{总损失 / 总样本数:.4f}",
                        '验证困惑度': f"{math.exp(总损失 / 总样本数):.2f}"
                    })
        
        # 在分布式训练中，收集所有进程的损失
        if self.配置.分布式训练:
            # 创建张量用于收集总损失和总样本数
            总损失张量 = torch.tensor([总损失], device=self.配置.设备)
            总样本数张量 = torch.tensor([总样本数], device=self.配置.设备)
            
            # 收集所有进程的数据
            dist.all_reduce(总损失张量, op=dist.ReduceOp.SUM)
            dist.all_reduce(总样本数张量, op=dist.ReduceOp.SUM)
            
            总损失 = 总损失张量.item()
            总样本数 = 总样本数张量.item()
        
        # 计算验证集平均损失
        验证平均损失 = 总损失 / 总样本数
        验证困惑度 = math.exp(验证平均损失)
        self.验证损失记录.append(验证平均损失)
        
        # 只在主进程记录TensorBoard数据和日志
        if not self.配置.分布式训练 or self.配置.主进程:
            # 记录TensorBoard数据
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('验证/轮次损失', 验证平均损失, self.当前轮数)
                self.tensorboard_writer.add_scalar('验证/轮次困惑度', 验证困惑度, self.当前轮数)
            
            logging.info(
                f"验证结果 | "
                f"轮次 {self.当前轮数 + 1} | "
                f"验证损失: {验证平均损失:.4f} | "
                f"验证困惑度: {验证困惑度:.2f}"
            )
            
            # 检查是否是最佳模型
            if 验证平均损失 < self.最佳验证损失:
                self.最佳验证损失 = 验证平均损失
                self.早停计数器 = 0
                self.保存模型('最佳模型.pth')
                logging.info(f"发现新的最佳模型，验证损失: {验证平均损失:.4f}")
            else:
                self.早停计数器 += 1
                logging.info(f"验证损失未改善，早停计数器: {self.早停计数器}/{self.配置.早停耐心值}")
        
        # 在分布式训练中，确保所有进程同步早停计数器和最佳验证损失
        if self.配置.分布式训练:
            # 广播最佳验证损失和早停计数器，确保所有进程一致
            最佳损失张量 = torch.tensor([self.最佳验证损失], device=self.配置.设备)
            早停计数器张量 = torch.tensor([self.早停计数器], device=self.配置.设备)
            
            # 从主进程广播到所有进程
            dist.broadcast(最佳损失张量, src=0)
            dist.broadcast(早停计数器张量, src=0)
            
            # 更新本地值
            self.最佳验证损失 = 最佳损失张量.item()
            self.早停计数器 = int(早停计数器张量.item())
        
        return 验证平均损失
    
    def 开始训练(self):
        """执行完整训练流程"""
        if not self.配置.分布式训练 or self.配置.主进程:
            logging.info("开始训练流程")
            if self.当前轮数 > 0:
                logging.info(f"从轮次 {self.当前轮数} 继续训练")
        
        for epoch in range(self.当前轮数, self.配置.训练轮数):
            self.当前轮数 = epoch
            轮次开始时间 = time.time()
            
            # 训练一个epoch
            训练损失 = self.训练单轮()
            
            # 验证
            验证损失 = self.验证模型()
            
            # 更新学习率
            self.学习率调度器.step()
            
            # 只在主进程打印epoch摘要
            if not self.配置.分布式训练 or self.配置.主进程:
                logging.info(
                    f"轮次 {epoch + 1} 摘要 | "
                    f"耗时: {time.time() - 轮次开始时间:.2f}秒 | "
                    f"训练损失: {训练损失:.4f} | "
                    f"训练困惑度: {math.exp(训练损失):.2f} | "
                    f"验证损失: {验证损失:.4f} | "
                    f"验证困惑度: {math.exp(验证损失):.2f} | "
                    f"学习率: {self.优化器.param_groups[0]['lr']:.6f}"
                )
                
                # 定期保存模型
                if (epoch + 1) % 5 == 0 or (epoch + 1) == self.配置.训练轮数:
                    self.保存模型(f'模型_轮次_{epoch + 1}.pth')
                
                # 保存最新检查点（用于恢复训练）
                self.保存模型('最新检查点.pth')
            
            # 检查早停条件 - 确保所有进程同步早停决定
            if self.配置.分布式训练:
                早停标志 = torch.tensor([1 if self.早停计数器 >= self.配置.早停耐心值 else 0], 
                                    device=self.配置.设备)
                dist.all_reduce(早停标志, op=dist.ReduceOp.MAX)
                早停决定 = 早停标志.item() > 0
            else:
                早停决定 = self.早停计数器 >= self.配置.早停耐心值
                
            if 早停决定:
                if not self.配置.分布式训练 or self.配置.主进程:
                    logging.info(f"达到早停条件，停止训练。最佳验证损失: {self.最佳验证损失:.4f}")
                break
            
            # 在分布式训练中，确保所有进程同步
            if self.配置.分布式训练:
                import torch.distributed as dist
                dist.barrier()
        
        # 保存最终模型（只在主进程）
        if not self.配置.分布式训练 or self.配置.主进程:
            self.保存模型('最终模型.pth')
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()
            logging.info(f"训练完成，总耗时: {time.time() - self.训练开始时间:.2f}秒")
    
    def 保存模型(self, 文件名):
        """保存模型到文件"""
        # 只在主进程保存模型
        if not self.配置.分布式训练 or self.配置.主进程:
            保存路径 = os.path.join(self.配置.模型保存目录, 文件名)
            
            # 确保保存目录存在
            os.makedirs(os.path.dirname(保存路径), exist_ok=True)
            
            # 创建临时文件，避免保存中断导致文件损坏
            临时保存路径 = 保存路径 + ".tmp"
            
            try:
                # 如果是分布式模型，需要保存原始模型
                if self.配置.分布式训练 and isinstance(self.模型, torch.nn.parallel.DistributedDataParallel):
                    模型状态字典 = self.模型.module.state_dict()
                else:
                    模型状态字典 = self.模型.state_dict()
                    
                torch.save({
                    '模型状态字典': 模型状态字典,
                    '优化器状态字典': self.优化器.state_dict(),
                    '学习率调度器状态': self.学习率调度器.state_dict(),
                    '当前轮数': self.当前轮数,
                    '最佳验证损失': self.最佳验证损失,
                    '训练损失记录': self.训练损失记录,
                    '验证损失记录': self.验证损失记录,
                    '配置': vars(self.配置)
                }, 临时保存路径)
                
                # 成功保存后，重命名为最终文件名
                if os.path.exists(保存路径):
                    os.remove(保存路径)  # 删除已有文件
                os.rename(临时保存路径, 保存路径)
                
                logging.info(f"模型已保存到 {保存路径}")
                
            except Exception as e:
                logging.error(f"保存模型到 {保存路径} 失败: {str(e)}")
                if os.path.exists(临时保存路径):
                    try:
                        os.remove(临时保存路径)  # 清理临时文件
                    except:
                        pass
    
    def 加载模型(self, 文件名):
        """从文件加载模型"""
        加载路径 = os.path.join(self.配置.模型保存目录, 文件名)
        if not os.path.exists(加载路径):
            raise FileNotFoundError(f"模型文件不存在: {加载路径}")
        
        # 在分布式训练中，确保所有进程都能访问模型文件
        if self.配置.分布式训练:
            dist.barrier()
            
        检查点 = torch.load(加载路径, map_location=self.配置.设备)
        
        # 如果是分布式模型，需要加载到module
        if self.配置.分布式训练 and isinstance(self.模型, torch.nn.parallel.DistributedDataParallel):
            self.模型.module.load_state_dict(检查点['模型状态字典'])
        else:
            self.模型.load_state_dict(检查点['模型状态字典'])
            
        self.优化器.load_state_dict(检查点['优化器状态字典'])
        self.学习率调度器.load_state_dict(检查点['学习率调度器状态'])
        self.当前轮数 = 检查点['当前轮数']
        self.最佳验证损失 = 检查点['最佳验证损失']
        self.训练损失记录 = 检查点['训练损失记录']
        self.验证损失记录 = 检查点['验证损失记录']
        
        if not self.配置.分布式训练 or self.配置.主进程:
            logging.info(f"从 {加载路径} 加载模型成功，当前轮数: {self.当前轮数 + 1}")
        
        return 检查点

    def 加载检查点(self, 检查点路径):
        """从检查点恢复训练状态"""
        if not os.path.exists(检查点路径):
            raise FileNotFoundError(f"检查点文件不存在: {检查点路径}")
        
        # 在分布式训练中，确保所有进程都能访问检查点文件
        if self.配置.分布式训练:
            import torch.distributed as dist
            dist.barrier()
            
        logging.info(f"从检查点 {检查点路径} 恢复训练状态")
        检查点 = torch.load(检查点路径, map_location=self.配置.设备)
        
        # 如果是分布式模型，需要加载到module
        if self.配置.分布式训练 and isinstance(self.模型, torch.nn.parallel.DistributedDataParallel):
            self.模型.module.load_state_dict(检查点['模型状态字典'])
        else:
            self.模型.load_state_dict(检查点['模型状态字典'])
            
        self.优化器.load_state_dict(检查点['优化器状态字典'])

        # 确保优化器状态被正确加载到当前设备
        for state in self.优化器.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.配置.设备)
                    
        self.学习率调度器.load_state_dict(检查点['学习率调度器状态'])
        self.当前轮数 = 检查点['当前轮数'] + 1  # 从下一轮开始
        self.最佳验证损失 = 检查点['最佳验证损失']
        self.训练损失记录 = 检查点['训练损失记录']
        self.验证损失记录 = 检查点['验证损失记录']
        
        # 在分布式训练中，确保所有进程同步训练状态
        if self.配置.分布式训练:
            # 广播最佳验证损失和当前轮数，确保所有进程一致
            最佳损失张量 = torch.tensor([self.最佳验证损失], device=self.配置.设备)
            当前轮数张量 = torch.tensor([self.当前轮数], device=self.配置.设备)
            
            # 从主进程广播到所有进程
            dist.broadcast(最佳损失张量, src=0)
            dist.broadcast(当前轮数张量, src=0)
            
            # 更新本地值
            self.最佳验证损失 = 最佳损失张量.item()
            self.当前轮数 = int(当前轮数张量.item())
        
        logging.info(f"训练状态恢复成功，将从轮次 {self.当前轮数} 继续训练，最佳验证损失: {self.最佳验证损失:.4f}")
        return 检查点

# ==================== 测试部分 ====================
class 模型测试器:
    def __init__(self, 模型, 分词器, 配置):
        """初始化测试器"""
        self.模型 = 模型
        self.分词器 = 分词器
        self.配置 = 配置
        self.模型.eval()  # 设置为评估模式
        logging.info(f"测试器初始化完成，设备: {配置.设备}")
    
    def 加载测试数据(self, 测试数据路径=None):
        """加载测试数据"""
        if 测试数据路径 and os.path.exists(测试数据路径):
            # 从文件加载测试数据
            测试数据 = []
            with open(测试数据路径, 'r', encoding='utf-8') as 文件:
                for 行 in 文件:
                    行 = 行.strip()
                    if not 行:
                        continue
                    try:
                        数据 = json.loads(行)
                        if isinstance(数据, dict) and 'zh' in 数据 and 'en' in 数据:
                            测试数据.append({
                                'zh': 数据['zh'],
                                'en': 数据['en']
                            })
                    except:
                        # 尝试按制表符分割
                        部分 = 行.split('\t')
                        if len(部分) >= 2:
                            测试数据.append({
                                'zh': 部分[0],
                                'en': 部分[1]
                            })
            logging.info(f"从文件加载了 {len(测试数据)} 条测试数据")
        else:
            # 生成随机测试数据
            logging.info("未提供测试数据文件，将生成随机测试数据")
            测试数据 = [
                {"zh": "我喜欢编程", "en": "I like programming"},
                {"zh": "人工智能正在改变世界", "en": "Artificial intelligence is changing the world"},
                {"zh": "深度学习是机器学习的一个分支", "en": "Deep learning is a branch of machine learning"},
                {"zh": "自然语言处理是人工智能的重要领域", "en": "Natural language processing is an important field of AI"},
                {"zh": "大型语言模型可以生成连贯的文本", "en": "Large language models can generate coherent text"}
            ]
            logging.info(f"生成了 {len(测试数据)} 条随机测试数据")
        
        # 限制测试样本数
        if self.配置.测试样本数 and len(测试数据) > self.配置.测试样本数:
            测试数据 = 测试数据[:self.配置.测试样本数]
            logging.info(f"限制测试样本数为 {len(测试数据)}")
        
        return 测试数据

    def 翻译(self, 源文本, 是英译中=True):
        """通用翻译函数"""
        with torch.no_grad():
            # 准备输入
            if 是英译中:
                输入文本 = f"{源文本} [SEP] "
            else:
                输入文本 = f"[SEP] {源文本}"
                
            输入ID = self.分词器.编码(输入文本)
            输入张量 = torch.tensor(输入ID).unsqueeze(1).to(self.配置.设备)
            
            # 生成序列
            输出ID = 输入ID.copy()
            最大生成长度 = self.配置.最大序列长度 - len(输入ID)
            
            for _ in range(最大生成长度):
                # 处理DistributedDataParallel包装过的模型
                if hasattr(self.模型, "module"):
                    序列掩码 = self.模型.module.生成序列掩码(len(输出ID))
                else:
                    序列掩码 = self.模型.生成序列掩码(len(输出ID))
                    
                输出张量 = self.模型(输入张量, 序列掩码)
                下一个词预测 = 输出张量[-1, 0].argmax(dim=-1).item()
                
                if 下一个词预测 == self.分词器.结束标记ID:
                    break
                    
                输出ID.append(下一个词预测)
                输入张量 = torch.tensor(输出ID).unsqueeze(1).to(self.配置.设备)
            
            # 解码生成的序列
            分隔符位置 = 输入文本.find('[SEP]')
            输入长度 = len(self.分词器.编码(输入文本[:分隔符位置 + 5]))
            生成ID = 输出ID[输入长度:]
            
            # 移除特殊标记
            生成ID = [id for id in 生成ID if id not in [self.分词器.开始标记ID, self.分词器.结束标记ID, self.分词器.填充标记ID]]
            
            return self.分词器.解码(生成ID)
            
    def 英译中(self, 英文文本):
        """英文翻译为中文"""
        return self.翻译(英文文本, 是英译中=True)
        
    def 中译英(self, 中文文本):
        """中文翻译为英文"""
        return self.翻译(中文文本, 是英译中=False)    

    def 计算BLEU分数(self, 参考文本, 候选文本):
        """计算BLEU分数"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # 分词
            参考分词 = list(参考文本)  # 字符级分词
            候选分词 = list(候选文本)  # 字符级分词
            
            # 计算BLEU分数
            smoothie = SmoothingFunction().method1
            bleu分数 = sentence_bleu([参考分词], 候选分词, smoothing_function=smoothie)
            
            return bleu分数
        except ImportError:
            logging.warning("未安装nltk库，无法计算BLEU分数")
            return None
    
    def 批量测试(self, 测试数据, 方向="英译中"):
        """批量测试模型性能"""
        结果 = []
        bleu分数列表 = []
        
        for 样本 in tqdm(测试数据, desc=f"{方向}测试"):
            if 方向 == "英译中":
                源文本 = 样本["en"]
                参考文本 = 样本["zh"]
                生成文本 = self.英译中(源文本)
            else:  # 中译英
                源文本 = 样本["zh"]
                参考文本 = 样本["en"]
                生成文本 = self.中译英(源文本)
            
            # 计算BLEU分数
            bleu分数 = self.计算BLEU分数(参考文本, 生成文本)
            if bleu分数 is not None:
                bleu分数列表.append(bleu分数)
            
            # 记录结果
            结果.append({
                "源文本": 源文本,
                "参考文本": 参考文本,
                "生成文本": 生成文本,
                "BLEU分数": bleu分数
            })
            
            # 打印样本结果
            logging.info(f"\n源文本: {源文本}")
            logging.info(f"参考文本: {参考文本}")
            logging.info(f"生成文本: {生成文本}")
            if bleu分数 is not None:
                logging.info(f"BLEU分数: {bleu分数:.4f}")
        
        # 计算平均BLEU分数
        平均BLEU = sum(bleu分数列表) / len(bleu分数列表) if bleu分数列表 else None
        if 平均BLEU is not None:
            logging.info(f"{方向}平均BLEU分数: {平均BLEU:.4f}")
        
        return 结果, 平均BLEU
    
    def 交互式测试(self):
        """交互式测试模型"""
        logging.info("\n开始交互式测试，输入'exit'退出")
        
        while True:
            print("\n请选择翻译方向:")
            print("1. 英译中")
            print("2. 中译英")
            print("3. 退出")
            
            选择 = input("请输入选项(1/2/3): ").strip()
            
            if 选择 == '3' or 选择.lower() == 'exit':
                break
            
            if 选择 == '1':
                # 英译中
                英文 = input("\n请输入英文文本: ").strip()
                if 英文.lower() == 'exit':
                    break
                
                print("正在翻译...")
                中文 = self.英译中(英文)
                print(f"翻译结果: {中文}")
            
            elif 选择 == '2':
                # 中译英
                中文 = input("\n请输入中文文本: ").strip()
                if 中文.lower() == 'exit':
                    break
                
                print("正在翻译...")
                英文 = self.中译英(中文)
                print(f"翻译结果: {英文}")
            
            else:
                print("无效选项，请重新选择")

# ==================== 分布式训练部分 ====================
def 设置分布式环境(配置, 本地排名=None):
    """设置分布式训练环境
    
    支持两种方式初始化：
    1. 手动指定本地排名（通过torch.multiprocessing.spawn启动）
    2. 从环境变量获取（通过torchrun启动）
    """
    # 从环境变量获取分布式参数（支持torchrun）
    if 本地排名 is None:
        if 'LOCAL_RANK' in os.environ:
            本地排名 = int(os.environ['LOCAL_RANK'])
            配置.世界大小 = int(os.environ.get('WORLD_SIZE', '1'))
            全局排名 = int(os.environ.get('RANK', '0'))
            logging.info(f"从环境变量获取分布式参数：LOCAL_RANK={本地排名}, WORLD_SIZE={配置.世界大小}, RANK={全局排名}")
        else:
            # 如果没有设置环境变量，默认为单进程
            本地排名 = 0
            配置.世界大小 = 1
            全局排名 = 0
            logging.warning("未检测到分布式环境变量，将使用单进程模式")
    
    # 设置设备（在初始化进程组前完成）
    if torch.cuda.is_available():
        torch.cuda.set_device(本地排名)
        配置.设备 = torch.device(f"cuda:{本地排名}")
    else:
        配置.设备 = torch.device("cpu")
    
    # 设置环境变量（如果没有通过torchrun设置）
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(配置.分布式端口)  # 确保端口是字符串类型
    
    # 检查是否已经初始化，避免重复初始化
    if not dist.is_initialized():
        # 初始化进程组
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://'
        )
        if 'LOCAL_RANK' in os.environ:
            logging.info(f"分布式进程组初始化完成 (torchrun模式)")
        else:
            logging.info(f"分布式进程组初始化完成 (manual spawn模式)")
    
    # 更新配置
    配置.分布式训练 = True
    配置.本地排名 = 本地排名
    配置.主进程 = (dist.get_rank() == 0)  # 使用dist.get_rank()获取全局排名
    
    # 只在主进程打印信息
    if 配置.主进程:
        logging.info(f"分布式环境初始化完成，世界大小: {dist.get_world_size()}，全局排名: {dist.get_rank()}，本地排名: {本地排名}，设备: {配置.设备}")
    
    return 配置

def 清理分布式环境():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logging.info("分布式环境已清理")

def 分布式训练进程(本地排名, 配置):
    """每个分布式进程执行的训练函数"""
    # 解析命令行参数
    参数 = 解析命令行参数()
    
    # 初始化资源变量
    tensorboard_writer = None
    
    try:
        # 设置分布式环境
        配置 = 设置分布式环境(配置, 本地排名)
        
        # 初始化日志系统和TensorBoard（只在主进程）
        if 配置.主进程:
            tensorboard_writer = 初始化日志系统(配置)
            logging.info(f"进程 {本地排名} 开始分布式训练")
            
        # 设置随机种子 - 为每个进程设置不同的种子
        基础种子 = 配置.随机种子
        进程种子 = 基础种子 + 配置.本地排名
        torch.manual_seed(进程种子)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(进程种子)
        
        # 加载数据集
        if 配置.主进程:
            logging.info("开始加载数据集...")
        训练数据集, 验证数据集 = 加载并准备数据集(配置)
        
        # 准备分词器
        分词器路径 = os.path.join(配置.模型保存目录, "分词器配置.json")
        if os.path.exists(分词器路径):
            if 配置.主进程:
                logging.info(f"加载已有分词器: {分词器路径}")
            分词器 = 双语分词器.加载分词器(配置, 分词器路径)
        else:
            if 配置.主进程:
                logging.info("训练新分词器...")
            分词器 = 准备分词器(配置, 训练数据集)
        
        # 创建数据集实例
        训练数据集实例 = 双语数据集(训练数据集, 分词器, 配置)
        验证数据集实例 = 双语数据集(验证数据集, 分词器, 配置)

        # 确保所有进程在继续之前同步
        if 配置.分布式训练:
            dist.barrier()
            
        # 创建分布式采样器
        训练采样器 = torch.utils.data.distributed.DistributedSampler(
            训练数据集实例,
            num_replicas=dist.get_world_size(),  # 使用dist函数获取世界大小
            rank=dist.get_rank()  # 使用dist函数获取排名
        )
        验证采样器 = torch.utils.data.distributed.DistributedSampler(
            验证数据集实例,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )

        # 创建数据加载器
        if 配置.主进程:
            logging.info("创建分布式数据加载器...")
        训练数据加载器 = DataLoader(
            训练数据集实例,
            batch_size=配置.批大小,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=训练采样器
        )
        验证数据加载器 = DataLoader(
            验证数据集实例,
            batch_size=配置.批大小,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=验证采样器
        )
        
        if 配置.主进程:
            logging.info(f"训练批次数: {len(训练数据加载器)}，验证批次数: {len(验证数据加载器)}")
        
        # 初始化模型
        if 配置.主进程:
            logging.info("初始化模型...")
        模型 = 双语语言模型(配置, 分词器).to(配置.设备)
        
        # 保存分词器的填充标记ID到配置对象中
        配置.填充标记ID = 分词器.填充标记ID
        
        # 将模型包装为DistributedDataParallel
        模型 = torch.nn.parallel.DistributedDataParallel(
            模型, 
            device_ids=[配置.本地排名] if torch.cuda.is_available() else None,
            output_device=配置.本地排名 if torch.cuda.is_available() else None
        )
        
        # 保存配置（只在主进程）
        if 配置.主进程:
            配置保存路径 = os.path.join(配置.模型保存目录, "训练配置.json")
            配置.保存配置(配置保存路径)
            logging.info(f"训练配置已保存到 {配置保存路径}")
        
        # 确保所有进程在继续之前同步
        if 配置.分布式训练:
            dist.barrier()
            
        # 初始化训练器
        if 配置.主进程:
            logging.info("初始化训练器...")
        训练器 = 模型训练器(模型, 训练数据加载器, 验证数据加载器, 配置, tensorboard_writer)
        
        # 检查是否需要恢复训练
        if 参数.恢复训练:
            检查点路径 = 参数.恢复训练
            if not os.path.isabs(检查点路径):
                检查点路径 = os.path.join(配置.模型保存目录, 检查点路径)
            训练器.加载检查点(检查点路径)
        
        # 开始训练
        if 配置.主进程:
            logging.info("开始分布式训练流程...")
        训练器.开始训练()
        
    except Exception as e:
        logging.error(f"进程 {配置.本地排名} 发生错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 在异常情况下通知其他进程
        if dist.is_initialized():
            # 创建错误标志
            错误标志 = torch.tensor([1], device=配置.设备)
            try:
                # 广播错误标志，通知其他进程
                dist.all_reduce(错误标志, op=dist.ReduceOp.MAX)
            except:
                pass  # 忽略通信错误
    finally:
        # 关闭TensorBoard
        if tensorboard_writer is not None:
            try:
                tensorboard_writer.close()
            except:
                pass
                
        # 清理分布式环境
        try:
            清理分布式环境()
        except:
            pass  # 忽略清理错误
            
        logging.info(f"进程 {配置.本地排名} 已退出")

def 启动分布式训练(配置):
    """启动分布式训练
    
    支持两种启动方式：
    1. 通过torch.multiprocessing.spawn手动启动多个进程
    2. 通过torchrun启动（此时本函数直接调用分布式训练进程）
    """
    # 检查是否通过torchrun启动（环境变量中有LOCAL_RANK）
    if 'LOCAL_RANK' in os.environ:
        logging.info("检测到通过torchrun启动，直接执行分布式训练进程")
        
        # 设置分布式训练参数
        配置.分布式训练 = True
        try:
            # 从环境变量获取分布式参数
            配置.世界大小 = int(os.environ.get('WORLD_SIZE', '1'))
            本地排名 = int(os.environ.get('LOCAL_RANK', '0'))
            全局排名 = int(os.environ.get('RANK', '0'))
            
            logging.info(f"torchrun环境变量: WORLD_SIZE={配置.世界大小}, LOCAL_RANK={本地排名}, RANK={全局排名}")
            
            # 直接执行分布式训练进程，不需要spawn
            分布式训练进程(None, 配置)
        except Exception as e:
            logging.error(f"启动torchrun模式分布式训练失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    else:
        # 传统方式：使用torch.multiprocessing启动多个进程
        try:
            # 获取可用GPU数量
            if torch.cuda.is_available():
                配置.世界大小 = torch.cuda.device_count()
            else:
                配置.世界大小 = 1
                logging.warning("未检测到可用GPU，将使用CPU进行分布式训练")
            
            if 配置.世界大小 <= 1:
                logging.warning("检测到只有一个可用GPU或没有GPU，建议使用单进程训练模式")
            
            logging.info(f"启动多进程分布式训练，进程数: {配置.世界大小}")
            
            # 使用torch.multiprocessing启动多个进程
            import torch.multiprocessing as mp
            mp.spawn(
                分布式训练进程,
                args=(配置,),
                nprocs=配置.世界大小,
                join=True
            )
        except Exception as e:
            logging.error(f"启动多进程分布式训练失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

# ==================== 命令行参数解析 ====================
def 解析命令行参数():
    """解析命令行参数"""
    解析器 = argparse.ArgumentParser(description="中英文双语语言模型训练程序")
    
    # 训练模式选择
    解析器.add_argument('--模式', type=str, default='训练', choices=['训练', '测试', '交互'],
                      help='运行模式: 训练、测试或交互')
    
    # 分布式训练选项
    解析器.add_argument('--分布式', action='store_true', help='是否使用分布式训练')
    
    # 模型加载选项
    解析器.add_argument('--加载模型', type=str, default=None, help='加载已有模型的路径')
    
    # 恢复训练选项
    解析器.add_argument('--恢复训练', type=str, default=None, help='恢复训练的检查点路径')
    
    # 数据集选项
    解析器.add_argument('--数据集', type=str, nargs='+', 
                      default=["wmt19", "opus100", "news_commentary"],
                      help='要使用的数据集列表')
    解析器.add_argument('--最大样本数', type=int, default=500000, 
                      help='每个数据集最大加载样本数')
    
    # 训练参数
    解析器.add_argument('--批大小', type=int, default=16, help='训练批大小')
    解析器.add_argument('--学习率', type=float, default=3e-4, help='初始学习率')
    解析器.add_argument('--训练轮数', type=int, default=20, help='训练轮数')
    解析器.add_argument('--早停耐心值', type=int, default=3, help='早停耐心值')
    
    # 模型参数
    解析器.add_argument('--模型维度', type=int, default=512, help='模型隐藏层维度')
    解析器.add_argument('--注意力头数', type=int, default=2, help='多头注意力头数')
    解析器.add_argument('--Transformer层数', type=int, default=6, help='Transformer编码器层数')
    
    # 测试参数
    解析器.add_argument('--测试数据', type=str, default=None, help='测试数据文件路径')
    解析器.add_argument('--测试样本数', type=int, default=100, help='测试样本数量')
    
    # 解析参数
    参数 = 解析器.parse_args()
    return 参数

def 加载并准备数据集(配置):
    """加载并预处理数据集"""
    数据集列表 = []
    
    def 预览数据集(数据集, 数据集名称):
        """预览数据集的前两行数据"""
        if len(数据集) >= 2:
            logging.info(f"\n{数据集名称}数据集前两行预览:")
            for i in range(2):
                样本 = 数据集[i]
                logging.info(f"\n第{i+1}行:")
                logging.info(f"中文: {样本['translation']['zh']}")
                logging.info(f"英文: {样本['translation']['en']}")
        else:
            logging.info(f"\n{数据集名称}数据集样本数不足两行")

    
    def 过滤无效样本(样本):
        """过滤无效或低质量的样本"""
        # 检查数据格式
        if 'translation' not in 样本:
            return False
            
        # 获取中英文文本
        中文文本 = 样本['translation'].get('zh', '')
        英文文本 = 样本['translation'].get('en', '')
        
        # 检查空值
        if not 中文文本 or not 英文文本:
            return False
            
        # 检查长度限制 (避免过长或过短的句子)
        if len(中文文本) < 5 or len(英文文本) < 5:
            return False
        if len(中文文本) > 500 or len(英文文本) > 500:
            return False
            
        # 检查特殊字符比例
        特殊字符 = set('!@#$%^&*()_+-=[]{}|;:,.<>?')
        中文特殊字符比例 = sum(1 for c in 中文文本 if c in 特殊字符) / len(中文文本)
        英文特殊字符比例 = sum(1 for c in 英文文本 if c in 特殊字符) / len(英文文本)
        if 中文特殊字符比例 > 0.3 or 英文特殊字符比例 > 0.3:
            return False
            
        return True
    
    # 加载WMT19中英数据集
    if "wmt19" in 配置.数据集名称列表:
        logging.info("正在加载WMT19数据集...")
        try:
            wmt19数据集 = load_dataset("wmt19", "zh-en", split=f"train[:{配置.最大样本数}]")
            # 检查并打印数据集结构
            logging.info(f"WMT19数据集结构: {wmt19数据集.features}")
            
            # 统一语言对顺序
            def 统一wmt19语言顺序(样本):
                # 检查数据格式
                if 'translation' not in 样本:
                    return None
                    
                中文文本 = 样本['translation'].get('zh', '')
                英文文本 = 样本['translation'].get('en', '')
                
                if not 中文文本 or not 英文文本:
                    return None
                    
                return {
                    'translation': {
                        'zh': 中文文本,
                        'en': 英文文本
                    }
                }
            wmt19数据集 = wmt19数据集.map(统一wmt19语言顺序)
            
            # 过滤无效样本
            wmt19数据集 = wmt19数据集.filter(过滤无效样本, desc="过滤WMT19无效样本")
            logging.info(f"WMT19数据集过滤后样本数: {len(wmt19数据集)}")
            # 预览数据集的前两行
            预览数据集(wmt19数据集, "WMT19")
            数据集列表.append(wmt19数据集)
            logging.info(f"WMT19数据集加载完成，样本数: {len(wmt19数据集)}")
        except Exception as e:
            logging.error(f"加载WMT19数据集失败: {str(e)}")
    
    # 加载OPUS-100数据集
    if "opus100" in 配置.数据集名称列表:
        logging.info("正在加载OPUS-100数据集...")
        try:
            opus100数据集 = load_dataset("opus100", "en-zh", split=f"train[:{配置.最大样本数}]")
            # 检查并打印数据集结构
            logging.info(f"OPUS-100数据集结构: {opus100数据集.features}")
            
            # 统一数据格式和语言顺序
            def 统一opus100格式(样本):
                # 检查数据格式
                英文文本 = ''
                中文文本 = ''
                
                if 'translation' in 样本:
                    英文文本 = 样本['translation'].get('en', '')
                    中文文本 = 样本['translation'].get('zh', '')
                elif 'zh' in 样本 and 'en' in 样本:
                    英文文本 = 样本.get('en', '')
                    中文文本 = 样本.get('zh', '')
                else:
                    logging.warning(f"无效的OPUS-100样本格式: {样本}")
                    return None
                    
                if not 中文文本 or not 英文文本:
                    return None
                    
                return {
                    'translation': {
                        'zh': 中文文本,
                        'en': 英文文本
                    }
                }
            opus100数据集 = opus100数据集.map(统一opus100格式)
            
            # 过滤无效样本
            opus100数据集 = opus100数据集.filter(过滤无效样本, desc="过滤OPUS-100无效样本")
            logging.info(f"OPUS-100数据集过滤后样本数: {len(opus100数据集)}")
            # 预览数据集的前两行
            预览数据集(opus100数据集, "OPUS-100")
            数据集列表.append(opus100数据集)
            logging.info(f"OPUS-100数据集加载完成，样本数: {len(opus100数据集)}")
        except Exception as e:
            logging.error(f"加载OPUS-100数据集失败: {str(e)}")
    
    # 加载News Commentary数据集
    if "news_commentary" in 配置.数据集名称列表:
        logging.info("正在加载News Commentary数据集...")
        try:
            news_commentary数据集 = load_dataset("news_commentary", "en-zh", split=f"train[:{配置.最大样本数}]")
            # 检查并打印数据集结构
            logging.info(f"News Commentary数据集结构: {news_commentary数据集.features}")
            
            # 统一数据格式和语言顺序
            def 统一news_commentary格式(样本):
                # 检查数据格式
                英文文本 = ''
                中文文本 = ''
                
                if 'translation' in 样本:
                    英文文本 = 样本['translation'].get('en', '')
                    中文文本 = 样本['translation'].get('zh', '')
                elif 'zh' in 样本 and 'en' in 样本:
                    英文文本 = 样本.get('en', '')
                    中文文本 = 样本.get('zh', '')
                else:
                    logging.warning(f"无效的News Commentary样本格式: {样本}")
                    return None
                    
                if not 中文文本 or not 英文文本:
                    return None
                    
                return {
                    'translation': {
                        'zh': 中文文本,
                        'en': 英文文本
                    }
                }
            news_commentary数据集 = news_commentary数据集.map(统一news_commentary格式)
            
            # 过滤无效样本
            news_commentary数据集 = news_commentary数据集.filter(过滤无效样本, desc="过滤News Commentary无效样本")
            logging.info(f"News Commentary数据集过滤后样本数: {len(news_commentary数据集)}")
            # 预览数据集的前两行
            预览数据集(news_commentary数据集, "News Commentary")
            数据集列表.append(news_commentary数据集)
            logging.info(f"News Commentary数据集加载完成，样本数: {len(news_commentary数据集)}")
        except Exception as e:
            logging.error(f"加载News Commentary数据集失败: {str(e)}")
    
    # 确保至少有一个数据集被成功加载
    if not 数据集列表:
        raise ValueError("没有成功加载任何数据集，请检查数据集配置和网络连接")
    
    # 合并数据集前检查所有数据集的特征是否一致
    for i, 数据集 in enumerate(数据集列表):
        logging.info(f"数据集 {i} 特征: {数据集.features}")
    
    # 在合并数据集前统一所有数据集的特征结构
    统一后的数据集列表 = []
    for i, 数据集 in enumerate(数据集列表):
        # 创建一个新的数据集，确保所有样本的特征结构完全一致
        # 只保留translation字段，并确保语言顺序一致为['zh', 'en']
        def 统一特征结构(样本):
            # 提取中英文文本，不管原始顺序如何
            if 'translation' in 样本:
                中文文本 = 样本['translation'].get('zh', '')
                英文文本 = 样本['translation'].get('en', '')
            else:
                中文文本 = ''
                英文文本 = ''
                
            # 返回统一结构的样本
            return {
                'translation': {
                    'zh': 中文文本,
                    'en': 英文文本
                }
            }
        
        # 使用map函数应用统一结构，并明确指定新的特征结构
        from datasets import Features, Translation, Value
        
        # 明确定义目标特征结构
        目标特征 = Features({
            'translation': Translation(languages=['zh', 'en'])
        })
        
        # 使用map函数应用统一结构，并指定新的特征
        统一后的数据集 = 数据集.map(
            统一特征结构, 
            desc=f"统一数据集{i}特征结构", 
            remove_columns=数据集.column_names,
            features=目标特征  # 明确指定特征结构
        )
        
        统一后的数据集列表.append(统一后的数据集)
        logging.info(f"数据集 {i} 特征统一后: {统一后的数据集.features}")
    
    # 合并数据集
    logging.info("开始合并数据集...")
    合并数据集 = concatenate_datasets(统一后的数据集列表)
    
    # 去重处理
    def 计算样本哈希(样本):
        return hash(f"{样本['translation']['zh']}{样本['translation']['en']}")
    
    样本哈希集合 = set()
    def 去重过滤(样本):
        样本哈希 = 计算样本哈希(样本)
        if 样本哈希 in 样本哈希集合:
            return False
        样本哈希集合.add(样本哈希)
        return True
    
    合并数据集 = 合并数据集.filter(去重过滤, desc="数据集去重")
    logging.info(f"数据集合并完成，去重后总样本数: {len(合并数据集)}")
    
    # 分割训练集和验证集
    数据集分割 = 合并数据集.train_test_split(test_size=配置.验证集比例, seed=配置.随机种子)
    logging.info(f"数据集分割完成，训练集: {len(数据集分割['train'])}，验证集: {len(数据集分割['test'])}")
    
    return 数据集分割['train'], 数据集分割['test']

def 准备分词器(配置, 训练数据集):
    """准备并训练分词器"""
    # 确保模型保存目录存在
    os.makedirs(配置.模型保存目录, exist_ok=True)
    
    # 在分布式训练中，只在主进程训练分词器
    if 配置.分布式训练:
        # 等待主进程完成分词器训练
        if 配置.主进程:
            logging.info("主进程开始训练分词器...")
            分词器 = _训练分词器(配置, 训练数据集)
        
        # 同步所有进程
        dist.barrier()
        
        # 非主进程加载已训练好的分词器
        if not 配置.主进程:
            分词器路径 = os.path.join(配置.模型保存目录, "分词器配置.json")
            logging.info(f"非主进程从 {分词器路径} 加载分词器")
            分词器 = 双语分词器.加载分词器(配置, 分词器路径)
    else:
        # 非分布式训练，直接训练分词器
        分词器 = _训练分词器(配置, 训练数据集)
    
    return 分词器
    
def _训练分词器(配置, 训练数据集):
    """实际训练分词器的函数(内部使用)"""
    # 创建临时文本文件用于训练分词器，使用英文文件名
    临时文本文件 = os.path.abspath(os.path.join(配置.模型保存目录, "temptext.txt"))
    logging.info(f"准备分词器训练语料到 {临时文本文件}")
    
    with open(临时文本文件, 'w', encoding='utf-8') as 文件:
        for 样本 in tqdm(训练数据集, desc="准备训练语料"):
            # 合并中英文文本
            文本 = f"{样本['translation']['en']} [SEP] {样本['translation']['zh']}"
            文件.write(文本 + "\n")
    
    # 检查文件是否存在和大小
    if not os.path.exists(临时文本文件):
        raise FileNotFoundError(f"临时语料文件不存在: {临时文本文件}")
    
    logging.info(f"临时语料文件大小: {os.path.getsize(临时文本文件) / 1024 / 1024:.2f} MB")
    logging.info("开始训练分词器...")
    
    # 初始化分词器
    分词器 = 双语分词器(配置)
    
    # 训练分词器
    分词器.训练分词器(临时文本文件)
    
    # 保存分词器配置
    分词器配置路径 = os.path.join(配置.模型保存目录, "分词器配置.json")
    分词器.保存分词器(分词器配置路径)
    
    # 安全地删除临时文件 - 先检查文件是否存在
    if os.path.exists(临时文本文件):
        try:
            os.remove(临时文本文件)
            logging.info("临时语料文件已删除")
        except Exception as e:
            logging.warning(f"删除临时语料文件时出错: {str(e)}")
    else:
        logging.warning(f"临时语料文件已不存在，无需删除: {临时文本文件}")
    
    return 分词器

    
def 主函数():
    """主函数，根据命令行参数执行不同功能"""
    # 初始化基本日志配置，确保在解析参数前有日志输出能力
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    
    # 解析命令行参数
    参数 = 解析命令行参数()
    
    # 初始化配置
    配置 = 模型配置()
    
    # 根据命令行参数更新配置
    配置.数据集名称列表 = 参数.数据集
    配置.最大样本数 = 参数.最大样本数
    配置.批大小 = 参数.批大小
    配置.学习率 = 参数.学习率
    配置.训练轮数 = 参数.训练轮数
    配置.早停耐心值 = 参数.早停耐心值
    配置.模型维度 = 参数.模型维度
    配置.注意力头数 = 参数.注意力头数
    配置.Transformer层数 = 参数.Transformer层数
    配置.测试样本数 = 参数.测试样本数
    
    # 设置HF镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 检查是否通过torchrun启动（环境变量中有LOCAL_RANK）
    if 'LOCAL_RANK' in os.environ:
        参数.分布式 = True
        logging.info("检测到通过torchrun启动，自动启用分布式训练模式")
        # 确保配置与环境变量一致
        配置.分布式端口 = os.environ.get('MASTER_PORT', 配置.分布式端口)
    
    # 根据运行模式执行不同功能
    if 参数.模式 == '训练':
        # 训练模式
        if 参数.分布式:
            # 分布式训练
            启动分布式训练(配置)
        else:
            # 单进程训练
            # 初始化日志系统和TensorBoard
            tensorboard_writer = 初始化日志系统(配置)
            
            # 设置随机种子
            torch.manual_seed(配置.随机种子)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(配置.随机种子)
            logging.info(f"设置随机种子: {配置.随机种子}")
            
            # 加载数据集
            logging.info("开始加载数据集...")
            训练数据集, 验证数据集 = 加载并准备数据集(配置)
            
            # 准备分词器
            分词器路径 = os.path.join(配置.模型保存目录, "分词器配置.json")
            if os.path.exists(分词器路径):
                logging.info(f"加载已有分词器: {分词器路径}")
                分词器 = 双语分词器.加载分词器(配置, 分词器路径)
            else:
                logging.info("训练新分词器...")
                分词器 = 准备分词器(配置, 训练数据集)
            
            # 创建数据加载器
            logging.info("创建数据加载器...")
            训练数据加载器 = DataLoader(
                双语数据集(训练数据集, 分词器, 配置),
                batch_size=配置.批大小,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            验证数据加载器 = DataLoader(
                双语数据集(验证数据集, 分词器, 配置),
                batch_size=配置.批大小,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            logging.info(f"训练批次数: {len(训练数据加载器)}，验证批次数: {len(验证数据加载器)}")
            
            # 初始化模型
            logging.info("初始化模型...")
            模型 = 双语语言模型(配置, 分词器).to(配置.设备)
            
            # 保存配置
            配置保存路径 = os.path.join(配置.模型保存目录, "训练配置.json")
            配置.保存配置(配置保存路径)
            logging.info(f"训练配置已保存到 {配置保存路径}")
            
            # 初始化训练器
            logging.info("初始化训练器...")
            训练器 = 模型训练器(模型, 训练数据加载器, 验证数据加载器, 配置, tensorboard_writer)
            
            # 检查是否需要恢复训练
            if 参数.恢复训练:
                检查点路径 = 参数.恢复训练
                if not os.path.isabs(检查点路径):
                    检查点路径 = os.path.join(配置.模型保存目录, 检查点路径)
                训练器.加载检查点(检查点路径)
            
            # 开始训练
            logging.info("开始训练流程...")
            训练器.开始训练()
    
    elif 参数.模式 == '测试' or 参数.模式 == '交互':
        # 测试或交互模式
        # 初始化日志系统
        tensorboard_writer = 初始化日志系统(配置)
        
        # 加载分词器
        分词器路径 = os.path.join(配置.模型保存目录, "分词器配置.json")
        if not os.path.exists(分词器路径):
            raise FileNotFoundError(f"分词器配置文件不存在: {分词器路径}")
        分词器 = 双语分词器.加载分词器(配置, 分词器路径)
        
        # 初始化模型
        模型 = 双语语言模型(配置, 分词器).to(配置.设备)
        
        # 加载模型权重
        模型文件路径 = 参数.加载模型
        if not 模型文件路径:
            # 如果未指定模型文件，尝试加载最佳模型
            最佳模型路径 = os.path.join(配置.模型保存目录, "最佳模型.pth")
            if os.path.exists(最佳模型路径):
                模型文件路径 = 最佳模型路径
            else:
                raise FileNotFoundError("未指定模型文件，且最佳模型文件不存在")
        
        # 加载模型检查点
        检查点 = torch.load(模型文件路径, map_location=配置.设备)
        模型.load_state_dict(检查点['模型状态字典'])
        logging.info(f"从 {模型文件路径} 加载模型成功")
        
        # 初始化测试器
        测试器 = 模型测试器(模型, 分词器, 配置)
        
        if 参数.模式 == '测试':
            # 测试模式
            测试数据 = 测试器.加载测试数据(参数.测试数据)
            
            # 英译中测试
            logging.info("\n开始英译中测试...")
            英译中结果, 英译中BLEU = 测试器.批量测试(测试数据, 方向="英译中")
            
            # 中译英测试
            logging.info("\n开始中译英测试...")
            中译英结果, 中译英BLEU = 测试器.批量测试(测试数据, 方向="中译英")
            
            # 输出总结
            logging.info("\n测试结果总结:")
            if 英译中BLEU is not None:
                logging.info(f"英译中平均BLEU分数: {英译中BLEU:.4f}")
            if 中译英BLEU is not None:
                logging.info(f"中译英平均BLEU分数: {中译英BLEU:.4f}")
        
        else:  # 交互模式
            # 交互式测试
            测试器.交互式测试()

if __name__ == '__main__':
    主函数()
