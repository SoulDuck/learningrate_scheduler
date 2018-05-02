import tensorflow as tf
import matplotlib.pyplot as plt
lr_=tf.placeholder(tf.float32  , name = 'learning_rate')
global_step = tf.placeholder(dtype =tf.int32)

#learning_rate = tf.train.exponential_decay(0.01,global_step,decay_steps=100000,decay_rate=0.96,staircase=True)
#learning_rate = tf.train.inverse_time_decay(0.01,global_step,decay_steps=100000,decay_rate=0.96,staircase=False)
boundaries = [10000,50000]
values = [1.0 , 0.5 , 0.1]
learning_rate = tf.train.piecewise_constant(global_step , boundaries , values)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
ys=[]
maxiter=100000
for step in range(maxiter):
    lr=sess.run(learning_rate, {global_step: step})
    ys.append(lr)
print lr
plt.plot(range(maxiter), ys)
plt.show()

