# Framework Integration# 
### 2021 
### PJ 
### Hu Chengwei, Shi Yucheng, Jin Haoliang

# Overview
DL is growing rapidly. This project aims to tidy framework and speed ​​up the experiment of the paper.
在这样一套流程之中，最重要的就是数据流。由于数据形式的异构造成了编写代码的多种多样。我们期望能从数据流中总结出相应的范式，达到加快整个项目处理的目的。

# Pytorch Integration
## 参数加载
### 静态参数处理stat_param
1. 配置文件我就要放在json文件中。
2. 配置文件是一段代码。
3. 配置文件来源于configparser。
### 动态参数处理dyn_param
运行期间才生成需要的参数，如：随机数
## 数据加载器
```
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```
### 加速模型之pinned memory内存
torch.utils.data.DataLoader中的pin_memory属性
```
pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory before  returning them.
```
通常情况下，由于虚拟内存技术的存在，数据要么在内存中以锁页（“pinned”）的方式存在，要么保存在虚拟内存（磁盘）中。而cuda只接受锁页内存传入，所以在声明新的dataloader对象时，直接令其保存在锁页内存中，后续即可快速传入cuda。否则，数据需要从虚拟内存中先传入锁页内存，再传入cuda，速度将大大增加。其缺点是对于内存的大小要求较高。
![pm](./image/pm.png)
### 预处理位置
从cache 或者 dataset file导入数据
如果使用分布式训练Make sure only the first process in distributed training process the dataset, and the others will use the cache
```
    if config.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",
                    config.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(
            config.data_dir) if evaluate else processor.get_train_examples(config.data_dir)
        features = convert_examples_to_features(
            examples, label_list, config.max_seq_len, tokenizer, "classification", use_entity_indicator=config.use_entity_indicator)
        if config.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(features, cached_features_file)

    if config.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
```
### 数据再处理（转tensor）
```
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor(
        [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor(
        [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
```
## 日志
1. logging 的全局配置
```
#引入了 logging 模块
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#声明了一个 Logger 对象
logger = logging.getLogger(__name__)
```
2. 将日志写入到文件
这里我们没有再使用 basicConfig 全局配置，而是，Logger 对象添加对应的 Handler 即可，最后可以发现日志就会被输出到 Alibaba.log 中，内容如下：
```
import logging
 #先声明一个 Logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
#然后指定其对应的 Handler 为 FileHandler 对象
handler = logging.FileHandler('Alibaba.log')
#然后 Handler 对象单独指定了 Formatter 对象单独配置输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
```
3. Formatter的用法
```
import logging
 
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARN)
#指定了一个 Formatter，并传入了 fmt 和 datefmt 参数，这样就指定了日志结果的输出格式和时间格式
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
#handler 通过 setFormatter() 方法设置此 Formatter 对象即可
handler.setFormatter(formatter)
logger.addHandler(handler)
```
4. 捕获 Traceback
```
import logging
 
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
 
# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
 
# FileHandler
file_handler = logging.FileHandler('result.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
 
# StreamHandler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
 
# Log
logger.info('Start')
logger.warning('Something maybe fail.')
try:
    result = 10 / 0
except Exception:
    logger.error('Faild to get result', exc_info=True)
logger.info('Finished')
```


## 并行化DDP
1. 初始化```torch.distributed```，这是DDP的依赖项。
2. 加载模型，如```model = model()```
3. 指定本进程对应的GPU：```torch.cuda.set_device(i)``` i 是当前进程对应的GPU号，以保证当前程在单独的GPU上运行
4. 将模型放到当前设备 ```model.to(device)```
5. 模型并行化：
```
model = make_data_parallel(model, opt.device)
    
def make_data_parallel(model, device):
   
    if device.type == 'cuda' and device.index is not None:
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        model.to(device)
​
        model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
```
6. 数据处理，首先获取原始数据。
7. 根据分布式情况以及原始数据指定Sampler，作为DataLoader的参数输入
```
train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
```
8. 使用DataLoader包装原始数据，由于传入了Sampler，会使用batch_sampler 在sampler中再进行分批。由于使用了分布式，在此步之前将batch_size除以设备数，得到新的batch_size，作为每个GPU的batch_size。因此batch_sampler会根据batch_size和sampler产生​batch。
```
train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
```
9. 在epoch中进行训练。注意，在每个epoch的开端调用```sampler.set_epoch(n)``` n为epoch数。
```
for i in range(opt.begin_epoch, opt.n_epochs + 1):
    train_sampler.set_epoch(i)
    current_lr = get_lr(optimizer)
    train_epoch(i, train_loader, model, criterion, optimizer,
                        opt.device, current_lr, train_logger,
                        train_batch_logger, tb_writer, opt.distributed)
```
10. 保存模型

## 训练
### 训练过程
```
for t in range(epoch):
    for step, (x, y) in enumerate(train_loader):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
 
        # Compute and print loss
        loss = criterion(y_pred, y) # 计算损失函数

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
        loss.backward() # loss反向传播
        optimizer.step() # 反向传播后参数更新 
```
### GPU与CPU搬运
```
device = torch.device("cude" if torch.cuda.is_available() else "cpu")
model.to(device)
```
### 优化器 & LR decay
```
optimizer = optim.SGD(params=model.parameters(), lr=0.1)

# 等间隔调整学习率，每训练step_size个epoch，lr*gamma
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 多间隔调整学习率，每训练至milestones中的epoch，lr*gamma
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 80], gamma=0.1)

# 指数学习率衰减，lr*gamma**epoch
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# 余弦退火学习率衰减，T_max表示半个周期，lr的初始值作为余弦函数0处的极大值逐渐开始下降，
# 在epoch=T_max时lr降至最小值，即pi/2处，然后进入后半个周期，lr增大
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
```
#### COS
```
criterion = LossCriterion() #构造函数有自己的参数
loss = criterion(x, y) #调用标准时也有参数
```
### 结果保存与加载
####1. 简单的保存与加载方法：
```
# 保存整个网络
torch.save(net, PATH) 
# 保存网络中的参数, 速度快，占空间少
torch.save(net.state_dict(),PATH)
#--------------------------------------------------
#针对上面一般的保存方法，加载的方法分别是：
model_dict=torch.load(PATH)
model_dict=model.load_state_dict(torch.load(PATH))
```
####2. 格式以字典的格式存储
然而，在实验中往往需要保存更多的信息，比如优化器的参数，那么可以采取下面的方法保存：
```
torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
                            'optimizer': optimizer.state_dict(),'alpha': loss.alpha, 'gamma': loss.gamma},
                           checkpoint_path + '/m-' + launchTimestamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')
```

加载的方式：
```
def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer
```
### 结果画图
####通过tensorboardX可视化训练过程
1. 先通过```tensorboardX```下的```SummaryWriter```类获取一个日志编写器对象

2. 通过这个对象的一组方法往日志中添加事件，即生成相应的图片
3. 启动前端服务器，在localhost中就可以看到最终的结果了
```
 from tensorboardX import SummaryWriter
 logger = SummaryWriter(log_dir="data/log")
 ​
 # 获取优化器和损失函数
 optimizer = torch.optim.Adam(MyConvNet.parameters(), lr=3e-4)
 loss_func = nn.CrossEntropyLoss()
 log_step_interval = 100      # 记录的步数间隔
 ​
 for epoch in range(5):
     print("epoch:", epoch)
     # 每一轮都遍历一遍数据加载器
     for step, (x, y) in enumerate(train_loader):
         # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
         predict = MyConvNet(x)
         loss = loss_func(predict, y)
         optimizer.zero_grad()   # 清空梯度（可以不写）
         loss.backward()     # 反向传播计算梯度
         optimizer.step()    # 更新网络
         global_iter_num = epoch * len(train_loader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
         if global_iter_num % log_step_interval == 0:
             # 控制台输出一下
             print("global_step:{}, loss:{:.2}".format(global_iter_num, loss.item()))
             # 添加的第一条日志：损失函数-全局迭代次数
             logger.add_scalar("train loss", loss.item() ,global_step=global_iter_num)
             # 在测试集上预测并计算正确率
             test_predict = MyConvNet(test_data_x)
             _, predict_idx = torch.max(test_predict, 1)     # 计算softmax后的最大值的索引，即预测结果
             acc = accuracy_score(test_data_y, predict_idx)
             # 添加第二条日志：正确率-全局迭代次数
             logger.add_scalar("test accuary", acc.item(), global_step=global_iter_num)
             # 添加第三条日志：这个batch下的128张图像
             img = vutils.make_grid(x, nrow=12)
             logger.add_image("train image sample", img, global_step=global_iter_num)
             # 添加第三条日志：网络中的参数分布直方图
             for name, param in MyConvNet.named_parameters():
                 logger.add_histogram(name, param.data.numpy(), global_step=global_iter_num)
```
####通过HiddenLayer实时可视化训练过程
```
 import hiddenlayer as hl
 import time
 ​
 # 记录训练过程的指标
 history = hl.History()
 # 使用canvas进行可视化
 canvas = hl.Canvas()
 ​
 # 获取优化器和损失函数
 optimizer = torch.optim.Adam(MyConvNet.parameters(), lr=3e-4)
 loss_func = nn.CrossEntropyLoss()
 log_step_interval = 100      # 记录的步数间隔
 ​
 for epoch in range(5):
     print("epoch:", epoch)
     # 每一轮都遍历一遍数据加载器
     for step, (x, y) in enumerate(train_loader):
         # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
         predict = MyConvNet(x)
         loss = loss_func(predict, y)
         optimizer.zero_grad()   # 清空梯度（可以不写）
         loss.backward()     # 反向传播计算梯度
         optimizer.step()    # 更新网络
         global_iter_num = epoch * len(train_loader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
         if global_iter_num % log_step_interval == 0:
             # 控制台输出一下
             print("global_step:{}, loss:{:.2}".format(global_iter_num, loss.item()))
             # 在测试集上预测并计算正确率
             test_predict = MyConvNet(test_data_x)
             _, predict_idx = torch.max(test_predict, 1)  # 计算softmax后的最大值的索引，即预测结果
             acc = accuracy_score(test_data_y, predict_idx)
 ​
             # 以epoch和step为索引，创建日志字典
             history.log((epoch, step),
                         train_loss=loss,
                         test_acc=acc,
                         hidden_weight=MyConvNet.fc[2].weight)
 ​
             # 可视化
             with canvas:
                 canvas.draw_plot(history["train_loss"])
                 canvas.draw_plot(history["test_acc"])
                 canvas.draw_image(history["hidden_weight"])
```
# Tensorflow Integration
## Bert4keras问题

# (Optional) TensorRT\tf-serving in deploying



