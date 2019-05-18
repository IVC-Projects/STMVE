import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn


def network(frame1, frame2, frame3, frame4, reuse = False, scope='netflow'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # feature extration
            c1_1 = slim.conv2d(frame1, 64, [3, 3], scope='convc1_1')
            c1_2 = slim.conv2d(frame2, 64, [3, 3], scope='convc1_2')
            c1_3 = slim.conv2d(frame3, 64, [3, 3], scope='convc1_3')
            c1_4 = slim.conv2d(frame4, 64, [3, 3], scope='convc1_4')
            # Merge
            conv = tf.concat([c1_1, c1_2, c1_3, c1_4], 3, name='c_concat')

            conv = slim.conv2d(conv, 64, [3, 3], scope='conv')
            # non-linear mapping
            # residual reconstruction
            # residual cell 1
            for i in range(7):

                c1 = slim.conv2d(conv, 64, [1, 1], scope='convB_%02d' % (i))
                convC = slim.conv2d(c1, 64, [3, 3], scope='convC_%02d' % (i))
                c2 = slim.conv2d(convC, 64, [1, 1], scope='convA_%02d' % (i))
                conv = tf.concat([conv, c2], 3, name='concat%02d' %(i))

            c5 = slim.conv2d(conv, 1, [5, 5], activation_fn=None, scope='conv5')

            # enhanced frame reconstruction
            output = tf.add(c5, frame2)

        return output
