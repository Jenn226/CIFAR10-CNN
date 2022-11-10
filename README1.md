---

<a name="CJ5ht"></a>
## 基于卷积神经网络的CIFAR10图像分类训练
<a name="r1BXr"></a>
### 一、数据集介绍
CIFAR10是一个用于图像识别的10分类彩色数据集，每张图片32x32的像素，每种对应有6000张图片，共60000张的数据中50000张训练集和10000张测试集。

<a name="WPQcY"></a>
### 二、基础模型
采用简单网络结构，卷积核大小统一采用3*3，具体如下：
:::tips
第一层：32个卷积核的卷积层、dropout层(0.3)、池化层<br />第二层：64个卷积核的卷积层、dropout层(0.3)、池化层<br />全连接层及输出层
:::
`**epochs=40，batch_size=128**`，训练结果如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/32911369/1668009262595-c1d0d9b6-0d08-4234-9ce3-23bfb5dbc399.png#clientId=u488f031e-ba1d-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=88&id=u95c8c9be&margin=%5Bobject%20Object%5D&name=image.png&originHeight=176&originWidth=1760&originalType=binary&ratio=1&rotation=0&showTitle=false&size=57785&status=done&style=none&taskId=u7f0031d3-17f4-408f-ab07-2e7510fa12b&title=&width=880)

- 结论：训练集上达到97%时，测试集仅72.3%，模型明显过拟合，针对这个问题进行改进。

<a name="ofUtL"></a>
### 三、模型优化与数据处理
<a name="Squ8C"></a>
#### 1. 数据增强
```python
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=None,rotation_range=15,width_shift_range=0.1,
                                                          height_shift_range=0.1,shear_range=0.1zoom_range=0.1,
														  horizontal_flip=True,fill_mode='nearest')
datagen.fit(x_img_train)
```

<a name="tCIWU"></a>
#### 2. 网络结构优化
:::tips

- 新增64核卷积层、128核卷积层、256核卷积层各一层
- 各层网络中间增加几层dropout防止过拟合
:::
```python
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),input_shape=(32,32,3),padding='same',
                                 activation='relu',strides=(1,1)))
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.4))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.4))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu',padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.4))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.4))
model.add(tf.keras.layers.Dense(1024,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
```

<a name="bHKH5"></a>
#### 3. 模型训练

- `**epochs=40，batch_size=128**`**训练结束后训练集准确率仅89%，故提高epochs至100，训练结果如下：**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/32911369/1668090178253-28212708-589e-4eb7-a61a-06c7d2b18b7b.png#clientId=u303aea05-b56d-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=85&id=u5fa3b21d&margin=%5Bobject%20Object%5D&name=image.png&originHeight=170&originWidth=1796&originalType=binary&ratio=1&rotation=0&showTitle=false&size=172000&status=done&style=none&taskId=ub92f635b-cd73-401b-863f-8f6a868809d&title=&width=898)

- **epochs与val_accuracy的相关性如图：**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/32911369/1668090229043-854a715e-6603-4ab2-aceb-4dda22d09608.png#clientId=u303aea05-b56d-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=387&id=ufc4a9df3&margin=%5Bobject%20Object%5D&name=image.png&originHeight=774&originWidth=1006&originalType=binary&ratio=1&rotation=0&showTitle=false&size=107515&status=done&style=none&taskId=u29bd3471-a4a6-454a-881d-61d99f63d31&title=&width=503)<br />由上图可以看出，当epochs=40左右时，val_accuracy已达到饱和，再提高epochs对于模型准确率的作用不大。

<a name="Gdpo9"></a>
### 四、安装
<a name="aFQka"></a>
#### 第1步：从Github克隆代码
:::tips
git clone [https://github.com/Jenn226/CIFAR10-CNN.git](https://github.com/Jenn226/CIFAR10-CNN)
:::
<a name="Mk0Hb"></a>
#### 第2步：安装环境
tensorflow2.9.1（采用2.1.x以上版本即可）<br />matplotlib3.6.2（用于训练数据可视化，选择性安装即可）<br />python3.10.6（采用3.7.x以上版本即可）
<a name="RF8Yo"></a>
#### 第3步：运行main1.ipynb即可
