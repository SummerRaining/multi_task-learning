## 推荐算法模型笔记：

### DeepFM模型

**Fm流程**：what：factorization machine 因子分解机，用于获取稀疏特征的交互信息。

Why：对于高基的离散变量一般使用onehot的方式编码，而对特征的进行二阶交互会更大的增加特征数量，特征数量从n维到n(n-1)维。因此产生大量的稀疏特征。

How：FM是在逻辑回归上增加了二阶特征，通过对每个特征学习一个隐向量vi，任意两个特征的交互表示为$<v_i,v_j>x_ix_j$。输出结果为：“线性部分”加上“所有二阶交互特征”，然后加上sigmoid激活函数。 

​	$$ y_{fm} = w_0+<w,x>+\sum_{i,j\in(1,n)}<v_i,v_j>x_ix_j $$

​	$$ output = sigmoid(y_{fm}) $$ 

优点分析：

1. FM的学习参数量为nd，d为隐向量的维数。相比于所有的二阶交互参数量为n(n-1）,减少很多。
2. 可以学习到训练集上不存在的交互特征，传统的二阶特征，如果训练集上的$x_ix_j$全部为0，那么$x_ix_j$的参数$w_{ij}$无法学习。而FM可以通过$<v_i,v_j>$来计算得到。每个参数学习的数据量增加了。
3. Fm时间可以分解成，先相加再相乘- 先相乘再相加形式。时间复杂度为，O(nd)。

**Deepfm**：

why：FM对于输入特征的一阶信息只做了简单的线性运算，因为在FM之外增加了dnn部分，对于输入增加更加复杂的非线性运算。（注意：DNN部分无法获得特征间的高阶交互）

how:

1. 线性部分和二阶Fm部分与FM一致。

2. 将FM学习的隐向量拼接在一起，添加多个全连接层，最后全连接到1维输出。

3. 将线性输出，二阶fm输出，dnn输出相加。使用sigmoid激活函数得到模型输出。

   $$ y_{deepfm} = y_{fm}+y_{dnn}$$
   
   $$ y_{dnn} = dense(concate([v_1,...,v_n])) $$

### MMOE模型

简述结构：

1. 将输入的离散变量和连续变量分别embedding后拼接在一起，经过几次全连接到固定维的向量x。
2. x分别经过m个子网络（专家网络）得到m个专家向量，每个子网络是相同大小的全连接。对于每个任务学习一个门网络，门网络将x作为输入，输出softmax后概率向量，并使用该概率对m个专家做加权平均。
3.  专家加权平均后的结果就作为对应塔任务网络的输入。每个塔网络接受专家加权平均值作为输入，经过一次全连接映射到一维的向量，做分类或回归任务。

### PLE模型

简述结构：

1. PLE模型的想法是，解决多任务训练中的翘翘板问题。多任务模型例如：mmoe，每个任务的损失梯度更新的时候会对所有的参数都更新。但有时不同的任务所提取的特征是冲突的，甚至是相反的；这样就会导致A任务表现上升时，B任务的表现下降。
2. PLE模型，提出每个任务都有自己的模块，然后所有任务同时共享一个模块。例如：A任务有两个expert （e1，e2），B任务有三个expert（e3,e4,e5），它们共享两个expert(e6,e7).A任务的tower等于(e1,e2,e6,e7)四个expert的加权平均，权重由门网络学习gate1(x)。同理：B任务的tower等于(e3,e4,e5,e6,e7)五个expert的加权平均，权重是gate2(x).每个任务的输出，由tower全连接到对应的维度。
3. 优势：这样在A任务的损失进行参数更新时，只影响了A自己的模块（e1，e2）和共享模块，而不影响B的模块。避免了两个任务同时学习时，产生冲突。

### Focal Loss

1. 原理：平衡易判断的样本和判断困难的样本之间的权重,将预测的很准的样本的损失减小。
2. 计算公式：$loss = (1−p_t)^{\gamma}log(p_t)\quad  p_t=\begin{cases} p,y_t = 1\\ 1-p,y_t = 0 \end{cases}$; 当判断错误时，$p_t$接近0，$1-p_t$接近1，权重等于1，对损失没有影响。判断正确时,$1-p_t$接近0，权重接近0，降低了正确分类样本的损失。
3. 在目标检测场景中，负样本非常多而且容易判断，而正样本很少且判断困难。所以focal loss可以自动降低负样本的权重。









