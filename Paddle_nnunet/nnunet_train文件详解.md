# Paddle_nnunet详解

### github地址：https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/contrib/MedicalSeg

## train.py命令解析

运行train.py的时候我们一般输入一下命令：

!python train.py --config ~/configs/nnunet_fold2.yml 

--log_iters 20 --precision fp16 --nnunet --save_dir output/cascade_lowres/fold2 --save_interval 100 --use_vdl --resume_model output/cascade_lowres/fold2/iter_98800

其中，每一个--后面都是我们想要配置的参数，他们的作用分别如下：

**1.--config：**这是我们模型的配置文件，里边包含了batch_size(批量大小，一批训练多少数据)，model(选择的模型，我们选择nnunet，里边的plan_path，stage，cascade涉及到nnunet模型知识，这里暂时不解释)，train_dataset(里边包含你训练集数据的路径，以及他们转换格式，裁剪后的文件路径，其他参数也暂时不解释)，val_dataset(与train_dataset同理)，optimizer（优化器，这里选择sgd），lr_scheduler(学习率改变方法)，loss(选择的损失函数)。

**2.--log_iters:** 这是每训练多少个iter输出一次日志。(我这里设置20是为了观察loss的变化情况，一般应该设置的再大一些)

**3.fp16：**FP16 和 FP32 是两种不同的浮点数精度，分别表示16位浮点数和32位浮点数。

以下解释内容出自ChatGPT：

1. **FP32（单精度）**：FP32 是一种标准的单精度浮点格式，它使用 32 位来存储一个浮点数，其中 1 位用于表示符号，8 位用于表示指数，23 位用于表示尾数。FP32 提供了相当大的数值范围和高精度，适用于需要高精度计算的应用。
2. **FP16（半精度）**：FP16 是一种半精度浮点格式，它使用 16 位来存储一个浮点数，其中 1 位用于表示符号，5 位用于表示指数，10 位用于表示尾数。相比于 FP32，FP16 使用的内存更少，计算速度更快，但精度和数值范围更小。

在深度学习领域，FP16 通常用于混合精度训练。混合精度训练是一种使用 FP16 和 FP32 结合的方法，它能在保持模型性能的同时，减少内存使用并加速计算。具体来说，模型的前向传播和反向传播过程使用 FP16 计算，而权重的更新则使用 FP32 计算。这种方法能够利用 FP16 的计算效率，同时避免因 FP16 精度较低而导致的训练不稳定问题。

我们这里选择了fp16混合精度训练，是因为模型复杂度较高，训练时间较长，显卡内存有限，所以使用fp16来减少复杂度。

**4.--nnunet:**指明我们使用的模型是nnunet，这会在后续选择训练文件时发挥作用。

**5.--save_dir:**指明我们需要把训练好的模型参数保存在那个文件夹。

**6.--save_interval**:指明每过多少个iter后进行模型参数的保存。(这里设置100，是在我觉得模型接近收敛的时候改的，旨在找寻loss最小值，刚开始训练时应该设置小一点，以防存储太多参数爆内存。)

**7.--use_vdl:**指明是否进行验证操作。

**8.--resume_model:**指明从哪个地方继续训练，通过这个命令，我们可以选择最近一次保存的参数，并从此参数开始继续训练。



## train.py解析

我们先找到主函数。

```python
if __name__ == '__main__':
	args = parse_args()
	main(args) 
```

可以看到先运行的函数是`parse_args`

```python
parser = argparse.ArgumentParser(description='Model training')
```

其中`argparse` 是 Python 的一个内置库，用于编写用户友好的命令行接口。`argparse` 库的 `ArgumentParser` 类提供了创建和处理命令行参数的方法。

在这个例子中，我们创建了一个`ArgumentParser` 的实例，并且被赋值给变量 `parser`。`description='Model training'` 是一个可选参数，它提供了关于程序的简短描述，这个描述会在命令行的帮助信息中显示。

```python
parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)
```

之后就是通过`parser`的add_argument方法不断添加参数，`"--config"` 是一个命令行参数的标志，用户在命令行中使用 `--config` 来指定一个配置文件。`dest="cfg"` 指定了 `--config` 参数在 `argparse` 解析后返回的对象中的属性名将是 `cfg`。也就是说，可以通过 `args.cfg` 来访问用户指定的配置文件。`help`是用来告诉你这个参数是干什么用的。`default`就是在你不声明这个参数的时候它的默认值是什么。`type`就是你所输入参数的数据类型。

```python
return parser.parse_args()
```

这里我们将调用parser(我们一开始创建的ArgumentParser对象)的parse_args()方法。它将从命令行读取参数，将它们转换为适当的类型（根据你使用 `add_argument` 方法添加的参数的定义），然后返回一个包含这些参数值的对象。

所以说整个`parse_args`函数就是在读取我们所输入的参数，以备后续代码使用。



之后我们会将读取完的参数传入`main`函数中，并运行`main`函数

```python
if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
```

这里是用来设置随机种子的，一般初始化神经网络的权重时，通常会用到随机数。这个随机的设置只会在你设置命令行参数时把--seed设置为不为空时启用。

```python
env_info = get_sys_env()
info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
logger.info(info)

```

这段代码的目的是获取系统环境信息并记录到日志中。

以下解释出自ChatGPT：

1. `get_sys_env()` 函数用来获取系统环境信息。具体的信息可能包括操作系统版本、Python 版本、各种库的版本等。这个函数返回一个字典，键是信息的名称，值是对应的信息。
2. 然后，这个字典被转换为一个字符串列表，每个字符串都是 "键: 值" 的形式。
3. `'\n'.join([...])` 将这个字符串列表连接成一个字符串，字符串中的每个元素之间用换行符 (`'\n'`) 分隔。
4. `''.join(['', format('Environment Information', '-^48s')] + info + ['-' * 48])` 这行代码在环境信息的上方和下方添加了一行由 `-` 符号组成的分隔线，长度为 48。`'^48s'` 是字符串格式化的一种方式，`^` 表示居中对齐，`48` 是宽度，`s` 表示字符串。所以，`format('Environment Information', '-^48s')` 就是将 'Environment Information' 这个字符串居中对齐，并用 `-` 符号填充到总长度为 48。
5. 最后，`logger.info(info)` 将这个字符串记录到日志中。`logger` 是一个日志记录器对象，`info` 是它的一个方法，用于记录一般信息。这行代码就是将系统环境信息记录到日志中。

```python
place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
	'GPUs used'] else 'cpu'
paddle.set_device(place)
```

env_info['Paddle compiled with cuda']表示 PaddlePaddle 是否是用 CUDA编译的，env_info[
	'GPUs used']表示是否有GPU可用，如果都满足`place`设置为GPU，否则设置成CPU。之后设置计算时所用的设备。

```python
cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)
```

这里我们创建了一个Config对象cfg，并调用其构造函数，将之前命令行所输入的参数作为参数进行构造。

```python
    train_dataset = cfg.train_dataset
    if train_dataset is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )
    val_dataset = cfg.val_dataset if args.do_eval else None
    losses = cfg.loss
```

这里我们将命令行所输入的参数分别赋给相应的参数，方便后续使用。

```python
    msg = '\n---------------Config Information---------------\n'
    msg += str(cfg)
    msg += '------------------------------------------------'
    logger.info(msg)
```

这段用来输出config文件的内容

![image-20230724153842480](C:\Users\12777\AppData\Roaming\Typora\typora-user-images\image-20230724153842480.png)



```python
    if args.nnunet:
        nnunet.core.train(
            cfg.model,
            train_dataset,
            val_dataset=val_dataset,
            optimizer=cfg.optimizer,
            save_dir=args.save_dir,
            iters=cfg.iters,
            batch_size=cfg.batch_size,
            resume_model=args.resume_model,
            save_interval=args.save_interval,
            log_iters=args.log_iters,
            num_workers=args.num_workers,
            use_vdl=args.use_vdl,
            losses=losses,
            keep_checkpoint_max=args.keep_checkpoint_max,
            precision=args.precision,
            profiler_options=args.profiler_options,
            to_static_training=cfg.to_static_training)
```

如果我们选择了nnunet的参数，我们将会调用nnunet.core.train进行训练，参数都是我们之前命令行输入的值。











