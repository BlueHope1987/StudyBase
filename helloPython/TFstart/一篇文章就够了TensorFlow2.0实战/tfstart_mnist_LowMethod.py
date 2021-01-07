#低层级方法实战MNIST
#一篇文章就够了 TensorFlow 2.0 实战 (持续更新) https://www.jianshu.com/p/fa334fd76d2f
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def preprocess(x, y):
    # [b, 28, 28], [b]
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(60000).batch(128).map(preprocess).repeat(30)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(10000).batch(128).map(preprocess)
x,y = next(iter(train_db))
print('train sample:', x.shape, y.shape)
# print(x[0], y[0])

def main():
    # learning rate
    lr = 1e-3
    # 784 => 512
    w1, b1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1)), tf.Variable(tf.zeros([512])) # 梯度只会跟踪tf.Variable类型的变量
    '''
    如果不用tf.Variable, 在with tf.GradientTape() as tape: 中需要调用tape.watch(w)，否则不会计算梯度
    '''
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10])) # stddev在这里解决了梯度爆炸的问题

    for step, (x,y) in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))

        with tf.GradientTape() as tape:

            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y-out)
            # [b, 10] => [b]
            loss = tf.reduce_mean(loss, axis=1)
            # [b] => scalar
            loss = tf.reduce_mean(loss)

        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # for g in grads:
        #     print(tf.norm(g))
        # update w' = w - lr*grad
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g) # assign_sub 原地更新， 不会改变变量类型

        if step % 100 == 0:
            print(step, 'loss:', float(loss))

        # evaluate
        if step % 500 == 0:
            total, total_correct = 0., 0

            for step, (x, y) in enumerate(test_db):
                # layer1.
                h1 = x @ w1 + b1
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1 @ w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2 @ w3 + b3
                # [b, 10] => [b]
                pred = tf.argmax(out, axis=1)
                # convert one_hot y to number y
                y = tf.argmax(y, axis=1)
                # bool type
                correct = tf.equal(pred, y)
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += x.shape[0]

            print(step, 'Evaluate Acc:', total_correct/total)

main()
'''
x: (60000, 28, 28) y: (60000,) x test: (10000, 28, 28) y test: [7 2 1 ... 4 5 6]
train sample: (128, 784) (128, 10)
0 loss: 1.4342228174209595
78 Evaluate Acc: 0.0948
100 loss: 0.49810081720352173
200 loss: 0.46395644545555115
300 loss: 0.39029449224472046
400 loss: 0.36943376064300537
500 loss: 0.2850426435470581
78 Evaluate Acc: 0.1821
600 loss: 0.2787204682826996
700 loss: 0.26938849687576294
800 loss: 0.26144126057624817
900 loss: 0.23059533536434174
1000 loss: 0.2131488025188446
78 Evaluate Acc: 0.2721
1100 loss: 0.22047963738441467
1200 loss: 0.2091558575630188
1300 loss: 0.19935768842697144
1400 loss: 0.19793620705604553
1500 loss: 0.18132804334163666
78 Evaluate Acc: 0.3372
1600 loss: 0.1973016858100891
1700 loss: 0.18589623272418976
1800 loss: 0.17424234747886658
1900 loss: 0.16788603365421295
2000 loss: 0.15851165354251862
78 Evaluate Acc: 0.3938
2100 loss: 0.16916552186012268
2200 loss: 0.1628144085407257
2300 loss: 0.16531193256378174
2400 loss: 0.15764164924621582
2500 loss: 0.15642619132995605
78 Evaluate Acc: 0.4357
2600 loss: 0.15673449635505676
2700 loss: 0.15462885797023773
2800 loss: 0.15843485295772552
2900 loss: 0.1409197449684143
3000 loss: 0.14514827728271484
78 Evaluate Acc: 0.4695
3100 loss: 0.1369980424642563
3200 loss: 0.142078697681427
3300 loss: 0.13284818828105927
3400 loss: 0.13761326670646667
3500 loss: 0.13285937905311584
78 Evaluate Acc: 0.5001
3600 loss: 0.1421109437942505
3700 loss: 0.13279464840888977
3800 loss: 0.13741004467010498
3900 loss: 0.12303931266069412
4000 loss: 0.12461290508508682
78 Evaluate Acc: 0.5261
4100 loss: 0.127879798412323
4200 loss: 0.12261638045310974
4300 loss: 0.11832118034362793
4400 loss: 0.12194661796092987
4500 loss: 0.11610166728496552
78 Evaluate Acc: 0.5497
4600 loss: 0.10628758370876312
4700 loss: 0.11476684361696243
4800 loss: 0.11812153458595276
4900 loss: 0.11631371825933456
5000 loss: 0.10556945204734802
78 Evaluate Acc: 0.5689
5100 loss: 0.11813661456108093
5200 loss: 0.11597959697246552
5300 loss: 0.1174570620059967
5400 loss: 0.10926808416843414
5500 loss: 0.09885643422603607
78 Evaluate Acc: 0.5829
5600 loss: 0.10640953481197357
5700 loss: 0.10381044447422028
5800 loss: 0.10340328514575958
5900 loss: 0.09734424948692322
6000 loss: 0.1054535061120987
78 Evaluate Acc: 0.5994
6100 loss: 0.11303738504648209
6200 loss: 0.10504340380430222
6300 loss: 0.09805221855640411
6400 loss: 0.10354974865913391
6500 loss: 0.10090135782957077
78 Evaluate Acc: 0.6142
6600 loss: 0.10521630942821503
6700 loss: 0.09500792622566223
6800 loss: 0.10087510943412781
6900 loss: 0.10147567838430405
7000 loss: 0.0912024974822998
78 Evaluate Acc: 0.6261
7100 loss: 0.1065359115600586
7200 loss: 0.10142739862203598
7300 loss: 0.10247887670993805
7400 loss: 0.09286149591207504
7500 loss: 0.09944598376750946
78 Evaluate Acc: 0.6377
7600 loss: 0.0906451866030693
7700 loss: 0.08539430797100067
7800 loss: 0.09875431656837463
7900 loss: 0.08716948330402374
8000 loss: 0.08804433047771454
78 Evaluate Acc: 0.6462
8100 loss: 0.08686503767967224
8200 loss: 0.08497065305709839
8300 loss: 0.0976509377360344
8400 loss: 0.09084402024745941
8500 loss: 0.07911895215511322
78 Evaluate Acc: 0.6544
8600 loss: 0.0872725173830986
8700 loss: 0.08695673942565918
8800 loss: 0.08802779018878937
8900 loss: 0.09346498548984528
9000 loss: 0.084139883518219
78 Evaluate Acc: 0.6636
9100 loss: 0.08093613386154175
9200 loss: 0.09089435636997223
9300 loss: 0.08152323216199875
9400 loss: 0.08926413953304291
9500 loss: 0.0910903662443161
78 Evaluate Acc: 0.6699
9600 loss: 0.08344104886054993
9700 loss: 0.07821619510650635
9800 loss: 0.07896409183740616
9900 loss: 0.08899035304784775
10000 loss: 0.07887225598096848
78 Evaluate Acc: 0.6791
'''