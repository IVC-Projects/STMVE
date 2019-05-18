import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

def network(frame1, frame2, reuse = False, scope='netflow'):

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # feature extration
            c11 = slim.conv2d(frame1, 64, [3, 3], scope='conv1_1')
            c12 = slim.conv2d(frame2, 64, [3, 3], scope='conv1_2')

            concat1_12 = tf.concat([c11, c12], 3, name='concat1_412')

            #feature merging
            concat12_1x1 = slim.conv2d(concat1_12, 64, [1, 1], scope='concat12_1x1')

            # rename!
            conv = concat12_1x1


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
