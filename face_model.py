import numpy as np
import glob
import cv2
import keras
import keras.layers as layers
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D,Activation
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras import backend as K
import tensorflow as tf
from tensorflow.python.framework import graph_util
from sklearn.cross_validation import train_test_split
import tensorflow.contrib.slim as slim

def identity_block(input_tensor, kernel_size = 3, filters = None, stage = None, block = None):
    '''
    :param input_tensor: input
    :param kernel_size: default value is 3
    :param filters: number of filter
    :param stage: Integer. generating the layer names
    :param block: generating the layer names, values in [a,b...]
    :return: output_tensor
    '''
    filters1,filters2,filters3 = filters
    channel_axis = 3
    # generateing the name
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=channel_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size = 3, filters = None, stage = None, block = None, strides=(2, 2)):
    '''
    :param input_tensor:  input
    :param kernel_size: default is 3
    :param filters: number of filter
    :param stage: Integer. generating the layer names
    :param block: generating the layer names, values in [a,b...]
    :param strides: Strides for the first conv layer in the block
    :return: output_tensor
    '''
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def resnet(input = None):
    if input is None:
        input = Input(shape=(224,224,3))
    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    return x

def conv2d_bn(x,filters,num_row,num_col,padding='same',strides=(1, 1),name=None):
    '''
    :param x: input_tensor
    :param filters: number of filter
    :param num_row: height
    :param num_col: width
    :param padding: same or valid
    :param strides: tuple or integer
    :param name: block_name
    :return: output_tensor
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(32,kernel_size=(num_row,num_col),strides=2,padding='valid',name = conv_name)(x)
    x = BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    x = Activation('relu',name = name)(x)
    return x

def inception(input = None):
    # Model of function
    if input is None:
        # input = Input(shape=(299,299,3))
        input = tf.Variable(np.zeros((1,299,299,3)),dtype = 'float32')
    # block1
    x = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid') # 149*149*32
    x = conv2d_bn(x, 32, 3, 3, padding='valid') # 147 * 147 * 32
    x = conv2d_bn(x, 64, 3, 3) # 147 * 147 * 64
    x = MaxPooling2D((3, 3), strides=(2, 2))(x) # 73 * 73 * 64

    # block2
    x = conv2d_bn(x,80,1,1,padding='valid') # 73 * 73 * 80
    x = conv2d_bn(x,192,3,3,padding='valid') # 71 * 71 * 192
    x = MaxPooling2D((3,3),strides=(2,2),padding='valid') # 35 * 35 * 192

    # branch1 35 * 35 * 256
    # b1 1 * 1
    branch1_1_conv1 = conv2d_bn(x,64,1,1) # 35 * 35 * 64

    # b1 3 * 3
    branch3_3_conv1 = conv2d_bn(x,64,1,1) # 35 * 35 * 64
    branch3_3_conv2 = conv2d_bn(branch3_3_conv1,96,3,3) # 35 * 35 * 96
    branch3_3_conv3 = conv2d_bn(branch3_3_conv2,96,3,3)# 35 * 35 * 96

    # b1 5 * 5
    branch5_5_conv1 = conv2d_bn(x,48,1,1) # 35 * 35 * 48
    branch1_5_conv2 = conv2d_bn(branch5_5_conv1,64,5,5) # 35 * 35 * 64

    # pooling
    branch1_pooling1 = AveragePooling2D((3,3),strides=(1,1),padding='same')(x) # 35 * 35 * 192
    brach1_pooling2 = conv2d_bn(branch1_pooling1,32,1,1) # 35 * 35 * 32
    # axis = 3 颜色通道那一级
    x = layers.concatenate([branch1_1_conv1,branch3_3_conv3,branch1_5_conv2,brach1_pooling2],axis=3,name='block1') # 35 * 35 * 256
    # branch2
    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))
    return input,x


'''
样本的选择非常重要,论文提供了两种选择方式:

1. 每n步保存一次模型,执行子集的argmax(positive-anchor)和argmin(negative-anchor)

2. 在线生成triplets，即在每个mini-batch中进行筛选positive/negative样本。
我们选择了大样本的mini-batch（1800样本/batch）来增加每个batch的样本数量。每个mini-batch中，我们对单个个体选择40张人脸图片作为正样本，
随机筛选其它人脸图片作为负样本。负样本选择不当也可能导致训练过早进入局部最小。为了避免，我们采用如下公式来帮助筛选负样本:
        f(anchor) - f(positive))^2 < (f(anchor) - f(negative))^2
随机选择负样本的时候 需要保证这个等式的成立.
'''

def tf_loss(predict,alpha=0.2):
    '''
    The loss function is sum((f(anchor) - f(positive))^2 - ((f(anchor) - f(negative))^2 + alpha))
    :param predicts:  [anchor,positive,negative] ,every elemnts is [1,128]
    :param alpha: default is 0.2, (f(anchor) - f(positive))^2 + alpha < (f(anchor) - f(negative))^2
    :return: loss
	'''
    with tf.Session() as sess:
        # origin image
        anchor = predict[0]
        # same person
        positive = predict[1]
        # another person
        negatove = predict[2]
        # operation
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(positive,anchor)))
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(negatove,anchor)))
        loss = tf.maximum(tf.reduce_sum(tf.add(tf.subtract(pos_dist,neg_dist),alpha)),0.0)
        return loss
    '''
    The loss function is sum((f(anchor) - f(positive))^2 - ((f(anchor) - f(negative))^2 + alpha))
    :param predicts:  [anchor,positive,negative] ,every elemnts is [1,128]
    :param alpha: default is 0.2, (f(anchor) - f(positive))^2 + alpha < (f(anchor) - f(negative))^2
    :return: loss
    # 
    anchor = predicts[0]
    positive = predicts[1]
    negative = predicts[2]
    # compute the loss according to the Facenet paper
    dis_anchor_positive = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),1)

    dis_anchor_negative = tf.reduce_sum(tf.add(tf.square(tf.subtract(anchor,negative)),alpha),1)
    # input: [n,]
    # 如果之间的距离差大于0.2,那么就可以认为相差的很远,不需要进行优化(提供loss)了,alpha是作为最小间距,
    loss = tf.maximum(tf.subtract(dis_anchor_positive,dis_anchor_negative),0.0)

    loss = tf.reduce_mean(loss,0)

    return loss
    '''

def keras_loss(y_true,y_pred,alpha=0.2):
    # origin image
    anchor = y_pred[0]
    # same person
    positive = y_pred[1]
    # another person
    negative = y_pred[2]
    # operation
    pos_dist = K.reduce_sum(K.square(K.subtract(positive, anchor)))
    neg_dist = K.reduce_sum(K.square(K.subtract(negative, anchor)))
    loss = K.maximum(K.reduce_sum(K.add(K.subtract(pos_dist, neg_dist), alpha)), 0.0)
    return loss

def build_model(input_shape,model_name = 'inception'):
    # muti input
    input_anchor = Input(shape = input_shape)
    input_positive = Input(shape = input_shape)
    input_negative = Input(shape = input_shape)
    if model_name == 'inception':
        _,base_model = inception()
    else:
        _,base_model = resnet()
    output_anchor = base_model.predict(np.expand_dims(input_anchor,1))
    output_positive = base_model.predict(np.expand_dims(input_positive,1))
    output_negative = base_model.predict(np.expand_dims(input_negative,1))
    return Model(inputs=[input_anchor,input_positive,input_negative],outputs=[output_anchor,output_positive,output_negative])


def load_dataset(directorys,image_suffix,is_need_generator = False,batch_size = 128):
    '''
    :param directory: train_set path. None or Str
    :param is_need_generator: return type. Array or generator. default value is False
    :return: according to the is_need_generator
    '''
    # default find the origin,same,other
    '''
    if is_need_generator:
        while True:
            return None
    '''
    # 需要返回每一个标签的人,思考如何传入三个参数
    if directorys is None or image_suffix is None:
        return None
    # read the data
    images = []
    names = []
    for director_index in range(len(directorys)):
        image_names = glob.glob(directorys[director_index]+"*."+image_suffix)
        for image_name in image_names:
            image = cv2.imread(image_name)[:,:,::-1]
            images.append(image)
        names.append({directorys[director_index].split('/')[-1]:len(image_names)})
    return images,names

def train_with_keras(optimizer,epoches=10,is_need_generator=False,batch_size=128):
    model = build_model((224,224,3))
    model.compile(optimizer=optimizer,loss=keras_loss) # don`t need check the accuracy
    images, names = load_dataset('train','jpg')
    # 总共的人数
    number_of_human = len(names.keys)
    # 创建Label
    y = []
    for number_index in range(number_of_human):
        # 将元素扩散
        y.append(np.repeat([number_index],names[names.keys[number_index]]))
    # 直接拆分成三个部分
    # compile
    model.compile(optimizer=optimizer,loss=keras_loss,metrics=['loss'])
    # 转换为Numpy array
    images = np.array(images)
    y = np.array(y)
    for epoch in range(1,epoches):
        # 每一代每一个人都需要训练一次
        for face in range(number_of_human):
            # 切换人脸
            # 求之前的所有人个数
            total_number = 0
            for index in range(face):
                total_number += names[names.keys[index]]
            # 当前训练的人的数量
            anchor_number = names[names.keys[number_index]]
            # 获取训练集
            # 训练人
            anchors = images[total_number:anchor_number]
            anchor_labels = y[total_number:anchor_number]
            # 自己其他训练集
            positive_indexes = np.random.permutation(len(anchor_labels))
            positives = anchors[positive_indexes]
            positive_labels = anchor_labels[positive_indexes]
            # 其他人
            all_negatives = np.concatenate([images[:total_number],images[anchor_number+total_number:]])
            all_negative_labels = np.concatenate([y[:total_number],y[anchor_number+total_number:]])
            # 随机选取同样的数据作为训练集
            # np.random.permutation(len(anchor_labels))
            selected_negative_indexes = np.random.randint(0,len(all_negatives),len(anchors))
            # 挑选出训练集
            negatives = []
            negative_labels = []
            for index in selected_negative_indexes:
                negatives.append(all_negatives[index])
                negative_labels.append(all_negative_labels[index])
            # 转换为numpy array
            negatives = np.array(negatives)
            negative_labels = np.array(negative_labels)

            # 验证集就在训练集中取出一部分就行
            train_anchors,valid_anchors,train_anchor_labels,valid_anchor_labels = \
                train_test_split(anchors, anchor_labels, test_size=0.2, random_state=42)

            train_positives,valid_positives,train_positive_labels,valid_positive_labels = \
                train_test_split(positives, positive_labels, test_size=0.2, random_state=42)

            train_negatives,valid_negatives,train_negative_labels,valid_negative_labels = \
                train_test_split(negatives, negative_labels, test_size=0.2, random_state=42)

            # fit x:[input1,input2,input3],output=[outpu1,outpu2,output3]
            model.fit([train_anchors,train_positives,train_negatives],
                      [train_anchor_labels,train_positive_labels,train_negative_labels],
                      batch_size=1, epoches=epoch,
                      validation_data=([valid_anchors,valid_positives,valid_negatives],
                                       [valid_anchor_labels,valid_positive_labels,valid_negative_labels]))
    # 返回模型
    return model

def train_with_tf(epoches = 10):
    # 获取输出tensor
    input,output_tensor = inception()
    # 获取数据集
    images, names = load_dataset('faces/train', 'jpg')
    # 总共的人数
    number_of_human = len(names.keys)
    # 创建Label
    y = []
    for number_index in range(number_of_human):
        # 将元素扩散
        y.append(np.repeat([number_index], names[names.keys[number_index]]))
    # 开始训练
    with tf.Graph().as_default():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(1, epoches):
                # 每一代每一个人都需要训练一次
                for face in range(number_of_human):
                    # 切换人脸
                    # 求之前的所有人个数
                    total_number = 0
                    for index in range(face):
                        total_number += names[names.keys[index]]
                    # 当前训练的人的数量
                    anchor_number = names[names.keys[number_index]]
                    # 获取训练集
                    # 训练人
                    anchors = images[total_number:anchor_number]
                    anchor_labels = y[total_number:anchor_number]
                    # 自己其他训练集
                    positive_indexes = np.random.permutation(len(anchor_labels))
                    positives = anchors[positive_indexes]
                    positive_labels = anchor_labels[positive_indexes]
                    # 其他人
                    all_negatives = np.concatenate([images[:total_number], images[anchor_number + total_number:]])
                    all_negative_labels = np.concatenate([y[:total_number], y[anchor_number + total_number:]])
                    # 随机选取同样的数据作为训练集
                    # np.random.permutation(len(anchor_labels))
                    selected_negative_indexes = np.random.randint(0, len(all_negatives), len(anchors))
                    # 挑选出训练集
                    negatives = []
                    negative_labels = []
                    for index in selected_negative_indexes:
                        negatives.append(all_negatives[index])
                        negative_labels.append(all_negative_labels[index])
                    # 转换为numpy array
                    negatives = np.array(negatives)
                    negative_labels = np.array(negative_labels)

                    # 验证集就在训练集中取出一部分就行
                    train_anchors, valid_anchors, train_anchor_labels, valid_anchor_labels = \
                        train_test_split(anchors, anchor_labels, test_size=0.2, random_state=42)

                    train_positives, valid_positives, train_positive_labels, valid_positive_labels = \
                        train_test_split(positives, positive_labels, test_size=0.2, random_state=42)

                    train_negatives, valid_negatives, train_negative_labels, valid_negative_labels = \
                        train_test_split(negatives, negative_labels, test_size=0.2, random_state=42)
                    # batch运行,但是最后需要reshape一下
                    predicts = []
                    for count in range(len(train_anchor_labels)):
                        input.assign(train_anchors[count])
                        predicts.append(sess.run(output_tensor))
                        input.assign(train_positives[count])
                        predicts.append(sess.run(output_tensor))
                        input.assign(train_negatives[count])
                        predicts.append(sess.run(output_tensor))
                        loss = tf_loss(predicts)
                        train_operation = tf.train.AdamOptimizer(0.01).minimize(loss)
                        output = sess.run(train_operation)

                print("epoch ",epoch," loss :",round(output,4))

# define the model
# NN1
xavier_init = tf.contrib.layers.xavier_initializer()

def conv_lnorm_block(net,num_outputs,kerner_size,stride,padding,lnormal,conv_1d=True):

    input_shape = net.get_shape().as_list()

    if conv_1d:
        net = slim.conv2d(net,num_outputs=input_shape[-1],kernel_size=1,stride=1,activation_fn=tf.nn.relu,weights_initializer=xavier_init)

    net = slim.conv2d(inputs=net, num_outputs=num_outputs, kernel_size=kerner_size, stride=stride, padding=padding,
                      activation_fn=tf.nn.relu, weights_initializer=xavier_init)

    net = slim.max_pool2d(net, kernel_size=(3, 3), stride=2)

    if lnormal:
        net = tf.nn.lrn(net)

    return net

def conv_block(net,num_outputs,kerner_size,stride,padding,pooling=True):
    input_shape = net.get_shape().as_list()
    # 1 * 1
    net = slim.conv2d(net,numb=input_shape[-1],kernel_size=1,stride=1,activation_fn=tf.nn.relu,weights_initializer=xavier_init)
    # 3 * 3
    net = slim.conv2d(inputs=net, num_outputs=num_outputs, kernel_size=kerner_size, stride=stride, padding=padding,
                      activation_fn=tf.nn.relu, weights_initializer=xavier_init)
    if pooling:
        net = slim.max_pool2d(net, kernel_size=(3, 3), stride=2)

    return net

def maxout(net,output_units,p = 2):

    input_shape = net.get_shape().as_list()
    # in: batch * 12544, output: batch * 4096
    in_count = input_shape[-1]
    out_count = output_units

    full_count = in_count + out_count

    # batch * 12544 * 4096 * 2
    W = tf.Variable(tf.truncated_normal(shape=(input_shape[0],input_shape[-1],output_units,p),mean=0.0,
                            stddev=tf.sqrt(2/full_count)))

    bias = tf.Variable(tf.zeros(shape=(output_units,p)))

    hidden = tf.add(tf.matmul(net,W),bias)

    net = tf.reduce_max(hidden,axis=1)

    return net

def build_NN1():
    input_shape = (220,220,3)
    input = tf.placeholder(dtype=tf.float16,shape=[None,input_shape[0],input_shape[1],input_shape[2]])
    # neural_in = input_shape[0] * input_shape[1] * input_shape[2]
    # neural_out = 110 * 110 * 64
    # model
    net = conv_lnorm_block(input,64,7,2,'SAME',True,conv_1d=False)
    net = conv_lnorm_block(net,192,3,1,'SAME',True,conv_1d=True)
    net = conv_block(net,384,3,1,'SAME',pooling=True)
    net = conv_block(net, 256, 3, 1, 'SAME', pooling=False)
    net = conv_block(net, 256, 3, 1, 'SAME', pooling=False)
    # output: 7 * 7 * 256
    net = conv_block(net, 256, 3, 1, 'SAME', pooling=True)
    # flatten
    net = slim.flatten(net)
    net = maxout(net,4096)
    net = maxout(net, 4096)
    net = slim.fully_connected(net,128)
    net = tf.nn.l2_normalize(net,axis=1)
    return net


def save_model(model,pb_file_path,sess):
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
    with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

def load_model(pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            # 得到所有的操作tensor返回
            operation_tensors = tf.import_graph_def(output_graph_def, name="")
        # 进行操作
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 获取操作的方式
            input_x = sess.graph.get_tensor_by_name("input:0")
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            out_label = sess.graph.get_tensor_by_name("output:0")

# 达到要求后,保存下来每个人最后提取的特征向量
def save_local_user_feature():
    pass

if __name__ == '__main__':
    # 训练模型,并且获取模型
    # model = train_with_keras(optimizer=keras.optimizers.adam)
    # 使用tf训练
    train_with_tf()
