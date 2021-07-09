## 一. 多任务的背景：

一些推荐场景中通常需要同时优化多个目标。例如在电影推荐中，我们不仅希望用户会购买并观看推荐的电影，还希望用户看完后会购买更多的电影。因此，我们需要同时预测用户“是否购买”和“电影的评分”。

经典的多任务模型有Share-Bottom model ，如图1所示：输入之后跟着一个bottom层（通常是多个全连接层），这些层是所有任务共享的，然后在bottom层之上每个任务都有一个自己的tower网络用于各自的预测。多任务的模型通常会比单任务模型要表现的更好，原因有：
	1. 每个任务都使用了其他任务的标签信息，这点类似于__迁移学习__，A任务用到了B任务的信息；每个任务下模型拿到信息更多，因此优于单任务模型。
 	2. A任务提取的特征，同时要用于B任务训练，因此提取的特征泛化性更好。模型的鲁棒性更好。

然而实际情况下，多任务模型并不总是比在每个任务上使用单任务模型表现的更好。由于所有任务的底层参数完全共享，不同任务之间的内在冲突会损坏模型的一些任务的预测。

![image-20210705202821879](image\image-20210705202821879.png)

## 二. 模型结构

MMOE没有让所有任务共享一个底层网络，而是设计了一组底层网络，每个底层网络被称为专家网络（论文中专家网络是前馈神经网络）。然后对每个任务引入了门网络，门网络将input作为输入，然后输出softmax后的权重，使用该权重对所有的expert进行加权平均。加权平均后的专家然后被输入到对应任务的tower网络中。如图2所示

![image-20210708141144900](image\image-20210708141144900.png)

#### 简述模型的结构：

1. 将输入的离散变量和连续变量分别embedding后拼接在一起，经过几次全连接到固定维的向量x。
2. x分别经过m个子网络（专家网络）得到m个专家向量，每个子网络是相同大小的全连接。对于每个任务学习一个门网络，门网络将x作为输入，输出softmax后概率向量，并使用该概率对m个专家做加权平均。
3. 专家加权平均后的结果就作为对应塔任务网络的输入。每个塔网络接受专家加权平均值作为输入，经过一次全连接映射到一维的向量，做分类或回归任务。

#### 对应的keras代码：

```python
class Mmoe_layer(tf.keras.layers.Layer):
    def __init__(self,expert_dim,n_expert,n_task):
        super(Mmoe_layer, self).__init__()
        self.n_task = n_task
        self.expert_layer = [Dense(expert_dim,activation = 'relu') for i in range(n_expert)]
        self.gate_layers = [Dense(n_expert,activation = 'softmax') for i in range(n_task)]
    
    def call(self,x):
        #多个专家网络
        E_net = [expert(x) for expert in self.expert_layer]
        E_net = Concatenate(axis = 1)([e[:,tf.newaxis,:] for e in E_net]) #(bs,n_expert,n_dims)
        #多个门网络
        gate_net = [gate(x) for gate in self.gate_layers]     #n_task个(bs,n_expert)
        
        #每个towers等于，对应的门网络乘上所有的专家网络。
        towers = []
        for i in range(self.n_task):
            g = tf.expand_dims(gate_net[i],axis = -1)  #(bs,n_expert,1)
            _tower = tf.matmul(E_net, g,transpose_a=True)
            towers.append(Flatten()(_tower))           #(bs,expert_dim)
            
        return towers
```



## 三. 模型分析

mmoe相比share-bottom结构有两个改进点：

1.  和MOE（混合专家）模型的优点一样：每个任务的输入都是多个子网络输出的平均值。这是一种集成模型，模型训练更稳定，鲁棒性更强。
2. 多个门，每个任务都使用不同的权重对多个专家求平均，这样可以捕捉不同任务之间的差异。对相关性比较差的模型，效果更好。

#### 效果分析：

论文中人工生成了多个不同相关性的数据集，将输入x进行sin/cos变化加上线性变化得到y，公式如下。控制两个任务之间的相关性，进行了多组实验。相比于OMOE和share-bottom结构，MMOE在相关性越小的时候，表现比前者更优；对比结果如图所示。

![image-20210708142307912](image\image-20210708142307912.png)

![image-20210708142402038](image\image-20210708142402038.png)

实验结果说明了多个门的结构是有效的，可以使得模型捕捉到不同任务之间的差异性。

实现代码已上传至github。
