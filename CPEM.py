# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:57:02 2017

@author: King-carrot
"""

from function import *

if __name__ == "__main__":
#Load data
    dataID = hdf5storage.loadmat('../data/data.mat')
    data = np.array(dataID['data'], dtype=np.float32)
    gt1 = scipy.io.loadmat('../data/label.mat')
    label = np.array(gt1['label'], dtype=np.int32)
    
    #Initialize
    label -= 1
    np.random.seed(2018)

    Outer_loop = 10
    Inner_loop = 10
    
    t_index = np.random.permutation(int(np.shape(data)[0]/Outer_loop)*Outer_loop)
    t_index = np.reshape(t_index, [Outer_loop, -1])
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0" 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    box = np.array([500, 1000, 4000, 8000, 35565], dtype=np.int32)
    flag = 0
    for test_index in t_index:
        if flag == Outer_loop-1:
            test_index = np.array(np.concatenate((test_index, np.array(range(int(np.shape(data)[0]/Outer_loop)*Outer_loop,np.shape(data)[0]))), axis=0), dtype=np.int32)
        train_index = np.setdiff1d(np.array(range(0,np.shape(data)[0])), test_index)
        train_data = data[train_index]
        train_label = label[train_index]
        test_data = data[test_index]
        test_label = label[test_index]
        
        kf = np.random.permutation(int(np.shape(train_data)[0]/Inner_loop)*Inner_loop)
        kf = kf.reshape([Inner_loop]+[-1])
        val_result = np.zeros([np.shape(train_data)[0],48], dtype=np.float32)
        
        tot_acc = np.zeros([Inner_loop,5], dtype=np.float32)
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(data, label)
        coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
        coefidx = np.argsort(coef)
        for inner_fold in range(0,Inner_loop):
            val_test_ind = kf[inner_fold]
            if inner_fold == Inner_loop-1:
                val_test_ind = np.array(np.concatenate((val_test_ind,np.array(range(int(np.shape(train_data)[0]/Outer_loop)*Outer_loop,np.shape(train_data)[0]),dtype=np.int32)), axis=0),dtype=np.int32)
            
            val_train_ind = np.setdiff1d(np.array(range(0,np.shape(train_data)[0]),dtype=np.int32), val_test_ind)
            val_train = train_data[val_train_ind]
            val_test = train_data[val_test_ind]
            val_train_label = train_label[val_train_ind]
            val_test_label = train_label[val_test_ind]
            temp = 0
            for item in box:
                idx = coefidx[-item:]
                vtrain = val_train[:,idx]
                vtest = val_test[:,idx]
                nn_acc, result_nn = dnn(vtrain, val_train_label, vtest, val_test_label)
                rf_acc, result_rf = rfc(vtrain, val_train_label, vtest, val_test_label)
                en_acc = 0.0
                for i in range(0,np.shape(vtest)[0]):
                    r = np.argmax(result_nn[i]+result_rf[i])
                    if r == val_test_label[i]:
                        en_acc += 1
                en_acc /= np.shape(vtest)[0]*0.01
                tot_acc[inner_fold,temp] = en_acc
                print("Inner_fold # of features: %d, Neural network accuracy: %f, Random forests accuracy: %f, Ensemble accuracy: %f" % (item, nn_acc, rf_acc, en_acc))
                temp += 1
        
        u = np.sum(tot_acc,0)
       
        best_n = box[np.argmax(u)]
        idx = coefidx[-best_n:]
        
        tr_data = train_data[:,idx]
        te_data = test_data[:,idx]
        nn_acc, result_nn = dnn(tr_data, train_label, te_data, test_label)
        rf_acc, result_rf = rfc(tr_data, train_label, te_data, test_label)
        en_acc = 0.0
        for i in range(0,np.shape(te_data)[0]):
            r = np.argmax(result_nn[i]+result_rf[i])
            if r == test_label[i]:
                en_acc += 1
        en_acc /= np.shape(te_data)[0]*0.01
        print("Outer_fold # of features:  %d, Neural network accuracy: %f, Random forests accuracy: %f, Ensemble accuracy: %f" % (best_n, nn_acc, rf_acc, en_acc))
        flag += 1
