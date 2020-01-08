Datewhale-初级算法组队学习

Assignment One_task1 



Stu_Id:15

Stu_Wechat_Group_Name 麻酱蘸红糖



机器学习是人工智能的一个实现途径。

达特茅斯会议-人工智能元年

​	会议内容： 人工智能



一、机器学习

1.1.1 定义

从数据中自动分析获得模型，并利用模型对未知数据进行预测。



1.1.2机器学习的应用场景

包括医疗、航空、教育、物流、电商等领域。

举例

（1）从数据（大量的猫和狗的图片）中自动分析获得模型（辨别猫和狗的规律），从而使机器拥有识别猫和狗的能力。



（2）从数据（房屋的各种信息）中自动分析获得模型（判断房屋价格的规律），从而使机器拥有预测房屋价格的能力。



1.1.3 机器学习 = 数据（data） + 模型（model） + 优化方法（optimal strategy）

从大量的日常经验中归纳规律，当面临新的问题的时候，就可以利用以往总结的规律去分析现实状况，找出潜在规律，采取最佳策略。



1.2.1Data-数据分为结构化数据和非结构化数据

1）结构化数据

可以使用关系型数据库表示和存储，表现为二维形式的数据。一般特点是：数据以行为单位，一行数据表示一个实体的信息，每一行数据的属性是相同的。举一个例子：



```objectivec
id      name            age     gender
1       Liu Yi          20      male
2       Chen Er         35      female
3       Zhang San       28      male
```

所以，结构化的数据的存储和排列是很有规律的，这对查询和修改等操作很有帮助。

但是，它的扩展性不好。比如，如果字段不固定，利用关系型数据库也是比较困难的，有人会说，需要的时候加个字段就可以了，这样的方法也不是不可以，但在实际运用中每次都进行反复的表结构变更是非常痛苦的，这也容易导致后台接口从数据库取数据出错。你也可以预先设定大量的预备字段，但这样的话，时间一长很容易弄不清除字段和数据的对应状态，即哪个字段保存有哪些数据。



2） 非结构化数据

数据结构不规则或不完整，没有预定义的数据模型，不方便用数据库二维逻辑表来表现的数据。包括所有格式的办公文档、文本、图片、各类报表、图像和音频/视频信息等等。

非结构化数据其格式非常多样，标准也是多样性的，而且在技术上非结构化信息比结构化信息更难标准化和理解。所以存储、检索、发布以及利用需要更加智能化的IT技术，比如海量存储、智能检索、知识挖掘、内容保护、信息的增值开发利用等。



1.2.2 应用场景

结构化数据，简单来说就是数据库。结合到典型场景中更容易理解，比如企业ERP、财务系统；医疗HIS数据库；教育一卡通；政府行政审批；其他核心数据库等。这些应用需要哪些存储方案呢？基本包括高速存储应用需求、数据备份需求、数据共享需求以及数据容灾需求。

非结构化数据，包括视频、音频、图片、图像、文档、文本等形式。具体到典型案例中，像是医疗影像系统、教育视频点播、视频监控、国土GIS、设计院、文件服务器（PDM/FTP）、媒体资源管理等具体应用，这些行业对于存储需求包括数据存储、数据备份以及数据共享等。



二、机器学习分类

2.1有监督学习和无监督学习

监督学习(supervised learning)（预测）

定义：输入数据是由输入特征值和目标值所组成。函数的输出可以是一个连续的值(称为回归），或是输出是有限个离散值（称作分类）。

分类 k-近邻算法、贝叶斯分类、决策树与随机森林、逻辑回归、神经网络

回归 线性回归、岭回归





无监督学习(unsupervised learning)

定义：输入数据是由输入特征值所组成。

聚类 k-means



2.2 区别

监督学习： 输入数据——有特征有标签，即有标准答案

非监督学习：输入数据——有特征无标签，即无标准答案





三、机器学习开发流程

<img src="/var/folders/ff/2q8cpg353gd7915tykkbkwp00000gn/T/com.yinxiang.Mac/com.yinxiang.Mac/WebKitDnD.DgxWKo/屏幕快照 2020-01-02 下午2.50.18.png" alt="屏幕快照 2020-01-02 下午2.50.18" style="zoom:80%;" />





四、机器学习模型





数据集构成

特征值+目标值

举例

![屏幕快照 2020-01-02 下午2.44.08](/var/folders/ff/2q8cpg353gd7915tykkbkwp00000gn/T/com.yinxiang.Mac/com.yinxiang.Mac/WebKitDnD.5Nc0Kl/屏幕快照 2020-01-02 下午2.44.08.png)

五、机器学习损失函数

5.1 Loss Function、Cost Function 和 Objective Function 的区别和联系



- 损失函数 Loss Function 通常是**针对单个训练样本而言**，给定一个模型输出 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D) 和一个真实 ![[公式]](https://www.zhihu.com/equation?tex=y) ，损失函数输出一个实值损失 ![[公式]](https://www.zhihu.com/equation?tex=L%3Df%28y_i%2C+%5Chat%7By_i%7D%29)
- 代价函数 Cost Function 通常是**针对整个训练集**（或者在使用 mini-batch gradient descent 时一个 mini-batch）的总损失 ![[公式]](https://www.zhihu.com/equation?tex=J%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D+f%28y_i%2C%5Chat%7By_i%7D%29)
- 目标函数 Objective Function 是一个更通用的术语，表示任意希望被优化的函数，用于机器学习领域和非机器学习领域（比如运筹优化）

一句话总结三者的关系：

A loss function is a part of a cost function which is a type of an objective function.



### 

5.1定义

损失函数（loss function）用来构建模型得到的预测值与真实值之间的差距，是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。损失函数是经验风险函数的核心部分，也是结构风险函数重要组成部分。模型的结构风险函数包括了经验风险项和正则项，通常可以表示成如下式子：

![img](https://img2018.cnblogs.com/blog/981211/201810/981211-20181022092825177-881167066.png)

整个公式分为两个部分，前面部分是损失函数，后面部分是正则项，正则项常用的有L1和L2正则项，目的是防止过拟合。





5.2 基本的损失函数

5.2.1 0-1损失

常用于二分类问题，0-1损失不连续、非凸，优化困难，0-1损失对每个错分类点都施以相同的惩罚，这样那些“错的离谱“ 的点并不会收到大的关注，这在直觉上不是很合适，因而常使用其他的代理损失函数进行优化。

0-1损失公式：
![img](https://img2018.cnblogs.com/blog/981211/201810/981211-20181022182728777-1949041372.png)

其中yf(x) > 0 ，则样本分类正确， yf(x) < 0 则分类错误，而相应的分类决策边界即为 f(x) = 0 。



5.2.2 平均绝对误差损失Mean Absolute Error Loss

也称为 L1 Loss

![[公式]](https://www.zhihu.com/equation?tex=+J_%7BMAE%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%5Cleft+%7C+y_i+-+%5Chat%7By_i%7D+%5Cright+%7C+%5C%5C)

同样的我们可以对这个损失函数进行可视化如下图，MAE 损失的最小值为 0（当预测等于真实值时），最大值为无穷大。可以看到随着预测与真实值绝对误差 ![[公式]](https://www.zhihu.com/equation?tex=%5Clvert+y-+%5Chat%7By%7D%5Crvert) 的增加，MAE 损失呈线性增长

![img](https://pic3.zhimg.com/80/v2-fd248542b6b5aa9fadcab44340045dee_hd.jpg)

5.2.3 均方差损失 Mean Squared Error

也称为 L2 Loss

![[公式]](https://www.zhihu.com/equation?tex=J_%7BMSE%7D+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%28y_i+-+%5Chat%7By_i%7D%29%5E2+%5C%5C)

从直觉上理解均方差损失，这个损失函数的最小值为 0（当预测等于真实值时），最大值为无穷大。下图是对于真实值 ![[公式]](https://www.zhihu.com/equation?tex=y%3D0) ，不同的预测值 ![[公式]](https://www.zhihu.com/equation?tex=%5B-1.5%2C+1.5%5D) 的均方差损失的变化图。横轴是不同的预测值，纵轴是均方差损失，可以看到随着预测与真实值绝对误差 ![[公式]](https://www.zhihu.com/equation?tex=%5Clvert+y-+%5Chat%7By%7D%5Crvert) 的增加，均方差损失呈二次方地增加。

<img src="https://pic1.zhimg.com/80/v2-f13a4355c21d16cad8b3f30e8a24b5cc_hd.jpg" alt="img" style="zoom:80%;" />

5.2.4 log对数损失函数

称逻辑斯谛回归损失(Logistic Loss)或交叉熵损失(cross-entropy Loss), 是在概率估计上定义的.

![img](https://img2018.cnblogs.com/blog/981211/201810/981211-20181022121857619-1433190203.png)

其中, Y 为输出变量, X为输入变量, L 为损失函数. N为输入样本量, M为类别数, yij 是表示类别 j 是否是输入实例 xi 的真实类别. pij 为模型或分类器预测输入实例 xi 属于类别 j 的概率。

如果目标是进行二分类，则损失函数可以简化为：

![img](https://img2018.cnblogs.com/blog/981211/201810/981211-20181022122334762-693433742.png)

5.2.5 指数损失函数

指数损失是在原有的损失函数上套一层指数，在adaboost上使用的就是指数损失，在加性模型中指数损失的主要吸引点在于计算上的方便。

指数损失公式：
![img](https://img2018.cnblogs.com/blog/981211/201810/981211-20181022174604258-1922940550.png)

其中n为样本数量，yi是样本的真实值，f(xi)是第i次迭代模型的权重。



![[公式]](https://www.zhihu.com/equation?tex=J%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D+f%28y_i%2C%5Chat%7By_i%7D%29)

5.2.6 合页损失函数 Hinge Loss

属于另外一种二分类损失函数，适用于 maximum-margin 的分类，支持向量机 Support Vector Machine (SVM) 模型的损失函数本质上就是 Hinge Loss + L2 正则化。

![[公式]](https://www.zhihu.com/equation?tex=J_%7Bhinge%7D%3D%5Csum_%7Bi%3D1%7D%5EN%5Coperatorname%7Bmax%7D%5Cleft%280%2C+1-%5Cmathbb%7Bsgn%7D%28y_i%29%5Chat%7By_i%7D%5Cright%29++%5C%5C)

下图是 ![[公式]](https://www.zhihu.com/equation?tex=y) 为正类， 即 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7Bsgn%7D%28y%29+%3D+1) 时，不同输出的合页损失示意图

![img](https://pic4.zhimg.com/80/v2-678fe676ad9dfd82d943eea56da26ecf_hd.jpg)

可以看到当 ![[公式]](https://www.zhihu.com/equation?tex=y) 为正类时，模型输出负值会有较大的惩罚，当模型输出为正值且在 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 区间时还会有一个较小的惩罚。即合页损失不仅惩罚预测错的，并且对于预测对了但是置信度不高的也会给一个惩罚，只有置信度高的才会有零损失。使用合页损失直觉上理解是要**找到一个决策边界，使得所有数据点被这个边界正确地、高置信地被分类**。





六、机器学习优化算法：

6.1常用的优化算法

梯度下降法（Gradient Descent），Momentum算法，牛顿法，AdaGrad。



6.1.1 全量梯度下降

通过每次在当前梯度方向（最陡峭的方向）向前“迈”一步，来逐渐逼近函数的最小值。针对的是整个数据集，通过对所有的样本的计算来求解梯度的方向。
![这里写图片描述](https://img-blog.csdn.net/20180121151415091?)

优点：全局最优解；易于并行实现；
缺点：当样本数目很多时，训练过程会很慢。



![image](https://images.cnblogs.com/cnblogs_com/LeftNotEasy/WindowsLiveWriter/1_1270E/image_thumb_15.png)



6.1.2 小批量梯度下降Mini Batch Gradient Descent
　　在上述的批梯度的方式中每次迭代都要使用到所有的样本，对于数据量特别大的情况，如大规模的机器学习应用，每次迭代求解所有样本需要花费大量的计算成本。是否可以在每次的迭代过程中利用部分样本代替所有的样本呢？基于这样的思想，便出现了mini-batch的概念。



6.1.3 随机梯度下降（Stochastic Gradient Descent）

在更新参数时都使用**一个**样本来进行更新。每一次跟新参数都用一个样本，更新很多次。如果样本量很大的情况（例如几十万），那么可能只用其中几万条或者几千条的样本，就已经将参数迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次，这种方式计算复杂度太高。

　　**优点**：训练速度快；

　　**缺点**：准确度下降，并不是全局最优；不易于并行实现。从迭代的次数上来看，随机梯度下降法迭代的次数较多，在解空间的搜索过程看起来很盲目。噪音很多，使得它并不是每次迭代都向着整体最优化方向。 

**三种方法使用的情况**：如果样本量比较小，采用批量梯度下降算法。如果样本太大，或者在线算法，使用随机梯度下降算法。在实际的一般情况下，采用小批量梯度下降算法。



6.1.4 引入动量的梯度下降

模拟物体运动的惯性：当我们跑步时转弯，我们最终的前进方向是由我们之前的方向和转弯的方向共同决定的。Momentum在每次更新时，保留一部分上次的更新方向：
Δθn=ρΔθn−1+gn−1
Δθn=ρΔθn−1+gn−1
θn:=θn−1−αΔθn
θn:=θn−1−αΔθn

  这里ρ值决定了保留多少上次更新方向的信息，值为0~1，初始时可以取0.5，随着迭代逐渐增大；α为学习率，同SGD。
优点：一定程度上缓解了SGD收敛不稳定的问题，并且有一定的摆脱局部最优的能力（当前梯度为0时，仍可能按照上次迭代的方向冲出局部最优点），直观上理解，它可以让每次迭代的“掉头方向不是那个大“。

缺点：这里又多了另外一个超参数ρρ需要我们设置，它的选取同样会影响到结果。





6.1.5自适应梯度下降

其实是对学习率进行了一个约束。即：

![n_t=n_{t-1}+g_t^2](http://zhihu.com/equation?tex=n_t%3Dn_%7Bt-1%7D%2Bg_t%5E2)

![\Delta{\theta_t}=-\frac{\eta}{\sqrt{n_t+\epsilon}}*g_t](http://zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bn_t%2B%5Cepsilon%7D%7D%2Ag_t)

此处，对![g_t](http://zhihu.com/equation?tex=g_t)从1到![t](http://zhihu.com/equation?tex=t)进行一个递推形成一个约束项regularizer，![-\frac{1}{\sqrt{\sum_{r=1}^t(g_r)^2+\epsilon}}](http://zhihu.com/equation?tex=-%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Csum_%7Br%3D1%7D%5Et%28g_r%29%5E2%2B%5Cepsilon%7D%7D)，![\epsilon](http://zhihu.com/equation?tex=%5Cepsilon)用来保证分母非0

**特点：**

- 前期![g_t](http://zhihu.com/equation?tex=g_t)较小的时候， regularizer较大，能够放大梯度
- 后期![g_t](http://zhihu.com/equation?tex=g_t)较大的时候，regularizer较小，能够约束梯度
- 适合处理稀疏梯度

**缺点：**

- 由公式可以看出，仍依赖于人工设置一个全局学习率
- ![\eta](http://zhihu.com/equation?tex=%5Ceta)设置过大的话，会使regularizer过于敏感，对梯度的调节太大
- 中后期，分母上梯度平方的累加将会越来越大，使![gradient\to0](http://zhihu.com/equation?tex=gradient%5Cto0)，使得训练提前结束





6.1.6 牛顿法

牛顿法
  不仅使用了一阶导信息，同时还利用了二阶导来更新参数，其形式化的公式如下：
θn:=θn−1−αL′n−1L″n−1
θn:=θn−1−αLn−1′Ln−1″

  回顾之前的θn=θn−1+Δθθn=θn−1+Δθ，我们将损失函数在θn−1θn−1处进行二阶泰勒展开：
L(θn)=L(θn−1+Δθ)≈L(θn−1)+L′(θn−1)Δθ+L″(θn−1)Δθ22
L(θn)=L(θn−1+Δθ)≈L(θn−1)+L′(θn−1)Δθ+L″(θn−1)Δθ22

  要使L(θn)<L(θn−1)L(θn)<L(θn−1)，我们需要极小化L′(θn−1)Δθ+L″(θn−1)Δθ22L′(θn−1)Δθ+L″(θn−1)Δθ22，对其求导，令导数为零，可以得到：
Δθ=−L′n−1L″n−1
Δθ=−Ln−1′Ln−1″

  也即牛顿法的迭代公式，拓展到高维数据，二阶导变为Hession矩阵，上式变为：
Δθ=−H−1L′n−1
Δθ=−H−1Ln−1′
  直观上，我们可以这样理解：我们要求一个函数的极值，假设只有一个全局最优值，我们需要求得其导数为0的地方，我们把下图想成是损失函数的导数的图像f(x)f(x)，那么：
k=tanθ=f′(x0)=f(x0)x0−x1→x1=x0−f(x0)f′(x0)
k=tan⁡θ=f′(x0)=f(x0)x0−x1→x1=x0−f(x0)f′(x0)

  我们一直这样做切线，最终xnxn将逼近与f′(x)f′(x)的0点，对于原函数而言，即Δθ=−L′n−1L″n−1Δθ=−Ln−1′Ln−1″。



七、机器学习评价指标

7.1机器学习算法的好坏主要由4个因素决定：

- 模型精度
- 判别速度
- 模型占用资源情况
- 模型训练速度



后三个情况的好坏都比较直观

判别速度就是模型的吞吐量，每秒可以处理多少条数据；

模型占用资源就是模型需要占用多少内存；

训练速度就是模型训练需要花费多长时间；

而精度的评价指标却比较多，而且评价指标在一定程度上相当于损失函数，模型优化的对象。现在就来总结一下常见的模型精度的评价指标。


7.2回归问题常见的评价指标：

7.2.1均方根误差

RMSE(Root Mean Square Error)

![img](https://file.ai100.com.cn/files/sogou-articles/original/cf58d955-304a-47c2-b2cc-f9b691eb6430/640.png)

7.2.2 均方差

MSE (Mean Square Error)

![img](https://file.ai100.com.cn/files/sogou-articles/original/4c81c092-4a9d-4a0a-80ee-1963facd990e/640.png)







前面两个由于误差是平房形式的，所以对某一两个异常值特别敏感，一两个异常值会使得整个模型有所偏斜。但是他们的好处是方便求导，符合高斯分布的假设，用的是最多的。

7.2.3 平均绝对误差

MAE(mean absolute error)

![img](https://file.ai100.com.cn/files/sogou-articles/original/4a0ea756-9bfd-430a-b12d-e2da21c77344/640.png)



7.2.4Python实现
MSE
def mse(y_test, y_true):
    return sp.mean((y_test - y_true) ** 2)

RMSE

def rmse(y_test, y_true):
    return sp.sqrt(sp.mean((y_test - y_true) ** 2))

MAE
def mae(y_test, y_true):
    return np.sum(np.absolute(y_test - y_true)) / len(y_test)


from sklearn.metrics import mean_squared_error 
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

mean_squared_error(y_test,y_predict)
root_mean_squared_error(y_test,y_predict)
mean_absolute_error(y_test,y_predict)







7.3 召回率、准确率、F值
对于二分类问题，可将样例根据其真实类别和分类器预测类别划分为：

真正例（True Positive，TP）：真实类别为正例，预测类别为正例。

假正例（False Positive，FP）：真实类别为负例，预测类别为正例。

假负例（False Negative，FN）：真实类别为正例，预测类别为负例。

真负例（True Negative，TN）：真实类别为负例，预测类别为负例。

然后可以构建混淆矩阵（Confusion Matrix）如下表所示。



分类问题对错最基础的概念就是混淆矩阵：

| 混淆矩阵   | 真实值为真 | 真实值为假 |
| ---------- | ---------- | :--------- |
| 预测值为真 | TP         | FP         |
| 预测值为假 | TN         | FN         |











八、机器学习模型选择

8.1交叉验证法（cross validation）

cross validation set，即交叉验证数据集。将所有数据按6：2：2
分为training set , cross validation set , testing set三类

  交叉验证法先将数据集D划分为k个大小相似的互斥子集，每次采用k−1k-1k−1个子集的并集作为训练集，剩下的那个子集作为测试集。进行k次训练和测试，最终返回k个测试结果的均值。又称为“k折交叉验证”（k-fold cross validation）。


  为减少因样本划分带来的偏差，通常重复p次不同的划分，最终结果是p次k折交叉验证结果的均值。



8.3 误差 方差







8.4过拟合 欠拟合



8.4.1 定义

过拟合：一个假设在训练数据上能够获得比其他假设更好的拟合， 但是在测试数据集上却不能很好地拟合数据，此时认为这个假设出现了过拟合的现象。(模型过于复杂)

\* 欠拟合：一个假设在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据，此时认为这个假设出现了欠拟合的现象。(模型过于简单)

![屏幕快照 2020-01-04 下午1.36.50](/var/folders/ff/2q8cpg353gd7915tykkbkwp00000gn/T/com.yinxiang.Mac/com.yinxiang.Mac/WebKitDnD.UKXs7O/屏幕快照 2020-01-04 下午1.36.50.png)

8.4.2原因以及解决办法

\* 欠拟合原因以及解决办法

  \* 原因：学习到数据的特征过少

  \* 解决办法：增加数据的特征数量

\* 过拟合原因以及解决办法

  \* 原因：原始特征过多，存在一些嘈杂特征， 模型过于复杂是因为模型尝试去兼顾各个测试数据点

  \* 解决办法：

​    \* 正则化

![屏幕快照 2020-01-04 下午1.37.45](/var/folders/ff/2q8cpg353gd7915tykkbkwp00000gn/T/com.yinxiang.Mac/com.yinxiang.Mac/WebKitDnD.hMqiWL/屏幕快照 2020-01-04 下午1.37.45.png)