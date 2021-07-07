from utils import *

def DNN(train_data, train_label, test_data, args):
    batch_size = args.batch_size
    epochs = args.epoch
    learning_rate = args.lr
    strategy = tf.distribute.MirroredStrategy()

    BUFFER_SIZE = np.shape(train_data)[0]
    BATCH_SIZE = batch_size*strategy.num_replicas_in_sync

    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #train_set = tf.data.Dataset.from_tensor_slices(train_data).batch(BATCH_SIZE)
    #label_set = tf.data.Dataset.from_tensor_slices(train_label).batch(BATCH_SIZE)
    #train_dist = strategy.experimental_distribute_dataset(train_set)
    #label_dist = strategy.experimental_distribute_dataset(label_set)

    ##Keras callbacks
    def scheduler(epoch):
        return learning_rate*(1.0e-3**(epoch/epochs))
    callback = [tf.keras.callbacks.LearningRateScheduler(scheduler), LRRecorder()]

    with strategy.scope():
        DNN = CPEM_DNN()
        optimizer = tfa.optimizers.RectifiedAdam(lr=learning_rate)
        DNN.compile(optimizer, loss=keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.AUTO))
        DNN.fit(train_data, train_label, callbacks=callback, epochs=epochs, shuffle=True, batch_size=BATCH_SIZE)
        predict = DNN.predict(test_data)

    ###Calculate the accuracy, precision, recall and f1-score
    return predict

def RF(train_data, train_label, test_data):
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
                                n_jobs=30,
                                random_state=123,
                                verbose=0,
                                warm_start=False,
                                class_weight=None)
    rf.fit(train_data, np.argmax(train_label,axis=1))
    predict = rf.predict_proba(test_data)
    return predict

def LOOCV(data, label, n_features, args):
    output = np.zeros(np.shape(label), dtype=np.float32)
    alpha = 0.5
    for i in range(0,np.shape(data)[0]):
        train_data = np.concatenate((data[:i], data[i+1:]), axis=0),
        train_label = np.concatenate((label[:i], label[i+1:]), axis=0)
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(train_data, np.argmax(train_label, axis=1))
        coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
        coefidx = np.argsort(coef)
        fidx = coefidx[-n_features:]
        train_data = train_data[:,fidx]
        test_data = data[i:i+1,fidx]
        if args.ensemble:
            dnn_output = DNN(train_data, train_label, test_data, args)
            rf_output = RF(train_data, train_label, test_data)
            dnn_output = np.squeeze(dnn_output)
            rf_output = np.squeeze(rf_output)
            output[i] = alpha*dnn_output+(1-alpha)*rf_output
        else:
            output[i] = np.squeeze(DNN(train_data, train_label, test_data, args))

    return output

def K_fold_cross_validation(data, label, k, n_features, args):
    output = np.zeros(np.shape(label), dtype=np.float32)
    alpha = 0.5
    index = np.random.permutation(int(np.shape(data)[0]/k)*k)
    index = np.reshape(index, [k, -1])
    for fold, test_index in enumerate(index):
        if fold == k-1:
            test_index = np.array(np.concatenate((test_index, np.array(range(int(np.shape(data)[0]/k)*k,np.shape(data)[0]))), axis=0), dtype=np.int32)
        train_index = np.setdiff1d(np.array(range(0,np.shape(data)[0])), test_index)

        if n_features !=np.shape(data)[1]: # Feature selection or not
            lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(data[train_index], np.argmax(label[train_index], axis=1))
            coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
            coefidx = np.argsort(coef)
            fidx = coefidx[-n_features:]
            data = data[:,fidx]

        if args.ensemble:
            dnn_output = DNN(data[train_index], label[train_index], data[test_index], args)
            rf_output = RF(data[train_index], label[train_index], data[test_index])
            dnn_output = np.squeeze(dnn_output)
            rf_output = np.squeeze(rf_output)
            output[test_index] = alpha*dnn_output+(1-alpha)*rf_output
        else:
            output[test_index] = np.squeeze(DNN(data[train_index], label[train_index], data[test_index], args))
    
    return output


##Return the performance for metrics (i.e. accuracy, AUC, precision, recall, f1-score)
def Performance(predict, label):
    n,c = np.shape(label)
    accuracy = 0.0
    predict_integer = np.zeros(np.shape(predict), dtype=np.int32)
    for i in range(0,n):
        if np.argmax(predict[i]) == np.argmax(label[i]):
            accuracy += 1
        predict_integer[i,np.argmax(predict[i])] = 1
    accuracy = 100*accuracy/np.shape(lable)[0]
    
    tot_f1 = 0.0
    tot_precision = 0.0
    tot_recall = 0.0
    tot_auc = 0.0
    for i in range(0,c):
        score = precision_recall_fscore_support(label[:,i], predict_integer[:,i], pos_label=1, average='binary')
        tot_f1 += score[2]
        tot_recall += score[1]
        tot_precision += score[0]
        tot_auc += roc_auc_score(label[:,i], predict[:,i])
    
    return accuracy, tot_auc/c, tot_precision/c, tot_recall/c, tot_f1/c

