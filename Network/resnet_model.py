import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import CIFAR_10_Initialization as CIFAR_INIT
import utils
import time
import os
import cv2
import copy
import matplotlib.pyplot as plt

plt.switch_backend('agg')


class ResNet(object):
    def __init__(self, model_name, sess, train_data, tst_data, epoch, restore_epoch, num_class, ksize,
                 out_channel1, out_channel2, out_channel3, learning_rate, weight_decay, batch_size, img_height,
                 img_width, train_phase):

        self.sess = sess
        self.training_data = train_data
        self.test_data = tst_data
        self.eps = epoch
        self.res_eps = restore_epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.k = ksize
        self.oc1 = out_channel1
        self.oc2 = out_channel2
        self.oc3 = out_channel3
        self.lr = learning_rate
        self.wd = weight_decay
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase

        self.build_model()
        self.saveConfiguration()

    def saveConfiguration(self):
        utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        utils.save2file('restore epoch : %d' % self.res_eps, self.ckptDir, self.model)
        utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        utils.save2file('ksize : %d' % self.k, self.ckptDir, self.model)
        utils.save2file('out channel 1 : %d' % self.oc1, self.ckptDir, self.model)
        utils.save2file('out channel 2 : %d' % self.oc2, self.ckptDir, self.model)
        utils.save2file('out channel 3 : %d' % self.oc3, self.ckptDir, self.model)
        utils.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        utils.save2file('weight decay : %g' % self.wd, self.ckptDir, self.model)
        utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        utils.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight', [ksize, ksize, inputMap.get_shape()[-1], out_channel],
                                          initializer=layers.variance_scaling_initializer(),
                                          regularizer=layers.l2_regularizer(self.wd))

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training, epsilon=1e-5, momentum=0.9)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            in_channel = inputMap.get_shape()[-1]
            fc_weight = tf.get_variable('fc_weight', [in_channel, out_channel],
                                        initializer=layers.variance_scaling_initializer(),
                                        regularizer=layers.l2_regularizer(self.wd))
            fc_bias = tf.get_variable('fc_bias', [out_channel], initializer=tf.zeros_initializer())

            fc_result = tf.matmul(inputMap, fc_weight) + fc_bias

            tf.summary.histogram('fc_weight', fc_weight)
            tf.summary.histogram('fc_bias', fc_bias)
            tf.summary.histogram('fc_result', fc_result)

            return fc_result

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def residualUnitLayer(self, inputMap, out_channel, ksize, unit_name, down_sampling, is_training, first_conv=False):
        with tf.variable_scope(unit_name):
            in_channel = inputMap.get_shape().as_list()[-1]
            if down_sampling:
                stride = 2
                increase_dim = True
            else:
                stride = 1
                increase_dim = False

            if first_conv:
                conv_layer1 = self.convLayer(inputMap, out_channel, ksize, stride, scope_name='conv_layer1')
            else:
                bn_layer1 = self.bnLayer(inputMap, scope_name='bn_layer1', is_training=is_training)
                relu_layer1 = self.reluLayer(bn_layer1, scope_name='relu_layer1')
                conv_layer1 = self.convLayer(relu_layer1, out_channel, ksize, stride, scope_name='conv_layer1')

            bn_layer2 = self.bnLayer(conv_layer1, scope_name='bn_layer2', is_training=is_training)
            relu_layer2 = self.reluLayer(bn_layer2, scope_name='relu_layer2')
            conv_layer2 = self.convLayer(relu_layer2, out_channel, ksize, stride=1, scope_name='conv_layer2')

            if increase_dim:
                identical_mapping = self.avgPoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping, [[0, 0], [0, 0], [0, 0],
                                                               [(out_channel - in_channel) // 2,
                                                                (out_channel - in_channel) // 2]])
            else:
                identical_mapping = inputMap

            return tf.add(conv_layer2, identical_mapping)

    def residualSectionLayer(self, inputMap, ksize, out_channel, unit_num, section_name, down_sampling, first_conv,
                             is_training):
        with tf.variable_scope(section_name):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

            return _out

    def resnet_model(self, input_x, model_name, unit_num1, unit_num2, unit_num3):
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            _conv = self.convLayer(input_x, self.oc1, self.k, stride=1, scope_name='unit1_conv')
            _bn = self.bnLayer(_conv, scope_name='unit1_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='unit1_relu')

            sec1_out = self.residualSectionLayer(inputMap=_relu,
                                                 ksize=self.k,
                                                 out_channel=self.oc1,
                                                 unit_num=unit_num1,
                                                 section_name='section1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            sec2_out = self.residualSectionLayer(inputMap=sec1_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc2,
                                                 unit_num=unit_num2,
                                                 section_name='section2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            sec3_out = self.residualSectionLayer(inputMap=sec2_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc3,
                                                 unit_num=unit_num3,
                                                 section_name='section3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            _fm_bn = self.bnLayer(sec3_out, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax, sec3_out

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.img_h, self.img_w, 3])
        self.y = tf.placeholder(tf.int32, [None])
        self.y_ont_hot = tf.one_hot(self.y, depth=self.num_class)
        self.is_training = tf.placeholder(tf.bool)

        self.y_pred, self.y_pred_softmax, self.sec3_fm = self.resnet_model(input_x=self.x, model_name='ResNet',
                                                                           unit_num1=3,
                                                                           unit_num2=3, unit_num3=3)

        with tf.variable_scope('Grad_CAM_Operators'):
            self.predicted_class_cam = tf.argmax(self.y_pred_softmax, 1)
            self.one_hot_cam = tf.one_hot(indices=self.predicted_class_cam, depth=self.num_class)
            self.signal_cam = tf.multiply(self.y_pred, self.one_hot_cam)
            self.loss_cam = tf.reduce_mean(self.signal_cam)
            self.grads_cam = tf.gradients(self.loss_cam, self.sec3_fm)[0]
            self.norm_grads_cam = tf.div(self.grads_cam,
                                         tf.sqrt(tf.reduce_mean(tf.square(self.grads_cam))) + tf.constant(1e-5))

        with tf.variable_scope('loss'):
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.y_ont_hot))
            self.l2_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = self.cost + self.l2_cost
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution = [tf.argmax(self.y_ont_hot, 1), tf.argmax(self.y_pred_softmax, 1)]
            self.correct_prediction = tf.equal(self.distribution[0], self.distribution[1])
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))

    def grad_cam(self, inputMap):
        output, grads_val = self.sess.run([self.sec3_fm, self.norm_grads_cam],
                                          feed_dict={self.x: inputMap, self.is_training: False})

        output = output[0]
        grads_val = grads_val[0]
        weights = np.mean(grads_val, axis=(0, 1))
        cam = np.ones(output.shape[0: 2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        return cam

    def generate_GradCAM_Image(self, save_dir='../Grad_CAM_Split/'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        out_counter = 1

        for i in range(len(self.test_data[0])):
            print('Start processing image {}.png'.format(str(i)))
            single_img = self.test_data[0][i]

            origin_img = copy.deepcopy(single_img)
            origin_img /= np.max(origin_img)

            imgForCal = np.expand_dims(single_img, 0)

            cam = self.grad_cam(imgForCal)
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (32, 32), interpolation=cv2.INTER_CUBIC)
            cam /= np.max(cam)

            fig, ax = plt.subplots()
            ax.imshow(origin_img)
            ax.imshow(cam, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
            plt.axis('off')

            height, width, channels = origin_img.shape

            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            plt.savefig(save_dir + str(out_counter) + '.png', dpi=300)
            plt.close()

            out_counter += 1
            print('Image' + str(out_counter) + '.png has been saved')

    def f_value(self, matrix):
        f = 0.0
        length = len(matrix[0])
        for i in range(length):
            recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(self.num_class)])
            precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(self.num_class)])
            result = (recall * precision) / (recall + precision)
            f += result
        f *= (2 / self.num_class)

        return f

    def test_procedure(self):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")
        test_loss = 0.0

        tst_batch_num = int(np.ceil(self.test_data[0].shape[0] / self.bs))
        for step in range(tst_batch_num):
            _testImg = self.test_data[0][step * self.bs:step * self.bs + self.bs]
            _testLab = self.test_data[1][step * self.bs:step * self.bs + self.bs]

            [matrix_row, matrix_col], tmp_loss = self.sess.run([self.distribution, self.loss],
                                                               feed_dict={self.x: _testImg,
                                                                          self.y: _testLab,
                                                                          self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

            test_loss += tmp_loss

        test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        test_loss = test_loss / tst_batch_num
        detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in range(self.num_class)]
        log1 = "Test Accuracy : %g" % test_accuracy
        log0 = "Test Loss : %g" % test_loss
        log2 = np.array(confusion_matrics.tolist())
        log3 = ''
        for j in range(self.num_class):
            log3 += 'category %s test accuracy : %g\n' % (utils.pulmonary_category[j], detail_test_accuracy[j])
        log4 = 'F_Value : %g' % self.f_value(confusion_matrics)

        utils.save2file(log1, self.ckptDir, self.model)
        utils.save2file(log0, self.ckptDir, self.model)
        utils.save2file(log2, self.ckptDir, self.model)
        utils.save2file(log3, self.ckptDir, self.model)
        utils.save2file(log4, self.ckptDir, self.model)

        return test_accuracy, test_loss

    def train(self):
        self.epoch_plt = []
        self.training_accuracy_plt = []
        self.test_accuracy_plt = []
        self.training_loss_plt = []
        self.test_loss_plt = []

        self.sess.run(tf.global_variables_initializer())
        self.train_itr = len(self.training_data[0]) // self.bs

        self.best_tst_accuracy = []
        self.best_tst_loss = []

        for e in range(1, self.eps + 1):
            _tr_img, _tr_lab = CIFAR_INIT.shuffle_data(self.training_data[0], self.training_data[1])

            training_acc = 0.0
            training_loss = 0.0

            for itr in range(self.train_itr):
                _tr_img_batch, _tr_lab_batch = CIFAR_INIT.next_batch(_tr_img, _tr_lab, self.bs, itr)
                _train_accuracy, _train_loss, _ = self.sess.run([self.accuracy, self.loss, self.train_op],
                                                                feed_dict={self.x: _tr_img_batch,
                                                                           self.y: _tr_lab_batch,
                                                                           self.is_training: True})
                training_acc += _train_accuracy
                training_loss += _train_loss

            summary = self.sess.run(self.merged, feed_dict={self.x: _tr_img_batch,
                                                            self.y: _tr_lab_batch,
                                                            self.is_training: False})

            training_acc = float(training_acc / self.train_itr)
            training_loss = float(training_loss / self.train_itr)

            utils.plot_acc_loss(self.ckptDir, self.eps, self.epoch_plt, self.training_accuracy_plt,
                                self.training_loss_plt, self.test_accuracy_plt, self.test_loss_plt)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e) + '.ckpt')

            self.epoch_plt.append(e)
            self.training_accuracy_plt.append(training_acc)
            self.training_loss_plt.append(training_loss)

            test_acc, test_loss = self.test_procedure()
            self.best_tst_accuracy.append(test_acc)
            self.best_tst_loss.append(test_loss)
            self.test_accuracy_plt.append(test_acc)
            self.test_loss_plt.append(test_loss)

            log1 = "Epoch: [%d], Training Accuracy: [%g], Test Accuracy: [%g], Loss Training: [%g] " \
                   "Loss Test: [%g], Time: [%s]" % \
                   (e, training_acc, test_acc, training_loss, test_loss, time.ctime(time.time()))

            utils.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)
        self.best_val_index = self.best_tst_accuracy.index(max(self.best_tst_accuracy))
        log2 = 'Highest Test Accuracy : [%g], Epoch : [%g]' % (
            self.best_tst_accuracy[self.best_val_index], self.best_val_index + 1)
        utils.save2file(log2, self.ckptDir, self.model)

        self.best_val_index_loss = self.best_tst_loss.index(min(self.best_tst_loss))
        log3 = 'Lowest Test Loss : [%g], Epoch : [%g]' % (
            self.best_tst_loss[self.best_val_index_loss], self.best_val_index_loss + 1)
        utils.save2file(log3, self.ckptDir, self.model)

    def test(self):
        print('-' * 20 + 'Start test procedure' + '-' * 20)
        self.saver.restore(self.sess, self.ckptDir + self.model + '-' + str(self.res_eps) + '.ckpt')
        self.generate_GradCAM_Image()
        # self.test_procedure()
