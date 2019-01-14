from utils import *

def lrelu(x, leak=0.2):
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1-leak)
    return f1*x+f2*abs(x)

def fnn(x, input_size, output_size, keep_prob, stddev=0.01, constant=0.0001, dropout=True, end=False, seed=2018):
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

    ###
   
def feature_selection(data, label, n, t=0):
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(data, label)
    try:
        if t == 0:
            coef = np.squeeze(np.sum(np.square(np.array(model.coef_)), axis=0))
        else:
            coef = np.squeeze(np.square(model.coef_))
    except AttributeError:
        coef = np.squeeze(np.array(model.feature_importances_))
    ind = np.argsort(coef)
    importance_feature = np.array(data[:,ind[-n-1:-1]])
    return importance_feature 
