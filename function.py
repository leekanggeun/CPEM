from utils import *

def lrelu(x, leak=0.2):
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1-leak)
    return f1*x+f2*abs(x)

def fnn(x, input_size, output_size, keep_prob, stddev=0.01, constant=0.0001, dropout=True, end=False):
    fc_w = tf.Variable(tf.truncated_normal([input_size,output_size], stddev=stddev,seed=seed))
    fc_b = tf.Variable(tf.constant(constant,shape=[output_size]), dtype=tf.float32)
    fc_h = tf.nn.relu(tf.matmul(x,fc_w)+fc_b) if not end else tf.matmul(x,fc_w)+fc_b
    return tf.nn.dropout(fc_h, keep_prob,seed=seed) if dropout else fc_h

def fcn(x, input_size, output_size, nlayers, nparameters, keep_prob):
    if nlayers == 1:
        h1 = fnn(x, input_size, output_size, keep_prob, end=True)
    elif nlayers == 2:
        h1 = fnn(fnn(x, input_size, nparameters, keep_prob, end=False), nparameters, output_size, keep_prob, end=True)
    elif nlayers >= 3:
        h0 = fnn(x, input_size, nparameters, keep_prob, end=False)
        for j in range(0,nlayers-2):
            if j == 0:
                h1 = fnn(h0, nparameters, nparameters, keep_prob, end=False)
            else:
                h1 = fnn(h1, nparameters, nparameters, keep_prob, end=False)
        h1 = fnn(h1, nparameters, output_size, keep_prob, end=True)
    else:
        print("# of layers can't be smaller than 0")
    return h1

def rfc(train_data, train_label, test_data, test_label):
    rf = RandomForestClassifier(n_estimators=150,
                                    criterion='gini',
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0,
                                    max_features=None,
                                    max_leaf_nodes=None,
                                    bootstrap=True,
                                    oob_score=False,
                                    n_jobs=10,
                                    random_state=123,
                                    verbose=0,
                                    warm_start=False,
                                    class_weight=None)
    rf.fit(train_data, train_label.ravel())
    result = rf.predict_proba(test_data)
    acc = 0.0
    for i in range(np.shape(test_data)[0]):
        r = np.argmax(result[i])
        if r == test_label[i]:
            acc += 1
    acc /= np.shape(test_data)[0]
    acc *= 100
    return acc, result

def dnn(train_data, train_label, test_data, test_label):
    g = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    batch_size = 10
    input_size = np.shape(train_data)[1]
    output_size = 48

    with g.as_default():
        p_x = tf.placeholder(tf.float32, [batch_size, 1, input_size, 1])
        p_y = tf.placeholder(tf.float32, [batch_size, output_size])
        keep_prob = tf.placeholder(tf.float32)
        h10_flat = tf.reshape(p_x, [batch_size,-1])
        h1 = fnn(h10_flat, input_size, 2048, keep_prob, end=False)
        h2 = fnn(h1, 2048, 2048, keep_prob, end=False)
        h3 = fnn(h2, 2048, 31, keep_prob, end=True)
        h4 = tf.reshape(h3, [batch_size, 31])
        h_c = tf.nn.softmax(h4)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=p_y, logits=h4))
        optim = tf.train.AdamOptimizer(1e-5)
        trainer = optim.minimize(loss)
    
    accuracy = 0.0
    result = np.zeros([np.shape(test_data)[0], 31])
    with tf.Session(graph=g, config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(0,120):
            loss_tot = 0.0
            for i in range(0,int(np.ceil(np.shape(train_data)[0]/batch_size))):
                a = np.random.randint(0,np.shape(train_data)[0],size=batch_size)
                x = train_data[a].reshape([batch_size, 1, input_size, 1])#[4,1,18181,1]
                y = np.zeros([batch_size, output_size])
                index = train_label[a]
                for u in range(0,batch_size):
                    y[u,index[u]] = 1
                _ , loss_val = sess.run([trainer, loss], feed_dict={p_x:x, p_y:y, keep_prob:0.6})
                loss_tot += loss_val
            print("%d epoch Loss: %f" % (e,(loss_tot)/np.shape(train_data)[0]))
        temp = 0
        for i in range(0,int(np.floor(np.shape(test_data)[0]/batch_size))):
            x = test_data[i*batch_size:(i+1)*batch_size].reshape([batch_size, 1, input_size, 1])
            out = sess.run(h_c, feed_dict={p_x:x, keep_prob:1})
            for j in range(0, batch_size):
                t = np.squeeze(out[j])
                result[temp] = t
                temp+=1
        remain = int(np.shape(test_data)[0]-np.floor(np.shape(test_data)[0]/batch_size)*batch_size)
        if remain > 0:
            x = test_data[-batch_size-1:-1].reshape([batch_size, 1, input_size, 1])
            out = sess.run(h_c, feed_dict={p_x:x, keep_prob:1})
            for j in range(0,int(remain)):
                t = np.squeeze(out[j+(batch_size-remain)])
                result[temp] = t
                temp+=1
        for i in range(0,np.shape(test_data)[0]):
            ind = np.argmax(np.squeeze(result[i]))
            if ind == test_label[i]:
                accuracy += 1
        accuracy /= np.shape(test_data)[0]*0.01
        sess.close()
    return accuracy, result

