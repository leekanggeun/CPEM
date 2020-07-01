from function import *

def parse_args():
    desc = "Tensorflow 2.1 implementation of denoising by Kanggeun Lee"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=120, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--inner', type=str, default='5', help='The outer-validation method (select LOOCV or integer valud for K-fold). If search is False, inner CV will be ignored')
    parser.add_argument('--outer', type=str, default='LOOCV', help='The inner-validation method (select LOOCV or integer valud for K-fold)') 
    parser.add_argument('--ensemble', help='Ensemble of Random forests and Deep Neural Network', action='store_true')
    parser.add_argument('--search', help='Searching for the number of features through inner cross-validation', action='store_true')
    parser.add_argument('--feature_selection', help='Feature selection to increase the performance', action='store_true')
    return check_args(parser.parse_args())

def check_args(args):
    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    try:
        assert not (args.search and not args.feature_selection)
    except:
        print('Feature selection should be True if search is True')
    

    return args

def main():
    args = parse_args()
    if args is None:
        exit()
    dataID = hdf5storage.loadmat('/home/shared/leekanggeun/CPEM/data/data.mat')
    data = np.array(dataID['data'], dtype=np.float32)
    labelID = scipy.io.loadmat('/home/shared/leekanggeun/CPEM/data/label.mat')
    label_ = np.array(labelID['label'], dtype=np.int32)
    label = np.zeros([np.shape(label_)[0], np.max(label_)], dtype=np.float32)
    label_ -= 1

    n,c = np.shape(label)
    for i in range(0,n):
        label[i,label_[i]] = 1

    final_out = np.zeros(np.shape(label), dtype=np.float32)
    if args.outer == 'LOOCV':
        for i in range(0,n):
            print("Leave One Out Cross Validation (LOOCV) outer loop for %d th element" % (i+1))
            train_data = np.concatenate((data[:i], data[i+1:]), axis=0)
            train_label = np.concatenate((label[:i], label[i+1:]), axis=0)
            test_data = data[i:i+1]
            test_label = label[i:i+1]
            if args.search:
                print("Searching the number of important features")
                search_list = [500, 1000, 2000, 4000, 8000, 35565]
                acc_list = np.zeros([len(search_list),], dtype=np.float32)
                if args.inner == 'LOOCV':
                    for j, n_features in enumerate(search_list):
                        val_result = LOOCV(train_data, train_label, n_features, args)
                        acc_list[j], _, _, _, _ = Performance(val_result, train_label)
                    best_n = search_list[np.argmax(acc_list)]
                else:
                    try:
                        inner = int(args.inner)
                    except ValueError:
                        print("Please enter an integer(str type) for K-fold Inner cross validation")
                        exit()
                    for j, n_features in enumerate(search_list):
                        val_result = K_fold_cross_validation(train_data, train_label, inner, n_features, args)
                        acc_list[j], _, _, _, _ = Performance(val_result, train_label)
                    best_n = search_list[np.argmax(acc_list)]
            else:
                print("We decide the number of features")
                best_n = 4000
            if feature_selection:
                lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(train_data, np.argmax(train_label,axis=1))
                coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
                coefidx = np.argsort(coef)
                fidx = coefidx[-best_n:]
            else:
                fidx = range(0,np.shape(train_data)[1])
            if args.ensemble:
                alpha = 0.5
                final_out[i] = alpha*DNN(train_data[:,fidx], train_label, test_data[:,fidx], args)
                final_out[i] += (1-alpha)*RF(train_data[:,fidx], train_label, test_data[:,fidx])
            else:
                final_out[i] = DNN(train_data[:,fidx], train_label, test_data[:,fidx], args)
    else:
        try:
            outer = int(args.outer)
        except ValueError:
            print("Please enter an integer(str type) for K-fold Outer cross validation")
            exit()

        index = np.random.permutation(int(np.shape(data)[0]/outer)*outer)
        index = np.reshape(index, [outer, -1])
        for fold, test_index in enumerate(index):
            print("%d-fold Cross Validation (K-fold CV) %d th outer loop" % (outer,fold+1))
            if fold == outer-1:
                test_index = np.array(np.concatenate((test_index, np.array(range(int(np.shape(data)[0]/outer)*outer,np.shape(data)[0]))), axis=0), dtype=np.int32)
            train_index = np.setdiff1d(np.array(range(0,np.shape(data)[0])), test_index)
            train_data = data[train_index]
            train_label = label[train_index]
            test_data = data[test_index]

            if args.search:
                print("Searching the number of important features")
                search_list = [500, 1000, 2000, 4000, 8000, 35565]
                acc_list = np.zeros([len(search_list),], dtype=np.float32)
                if args.inner == 'LOOCV':
                    for j, n_features in enumerate(search_list):
                        val_result = LOOCV(train_data, train_label, n_features, args)
                        acc_list[j], _, _, _, _ = Performance(val_result, train_label)
                    best_n = search_list[np.argmax(acc_list)]
                else:
                    try:
                        inner = int(args.inner)
                    except ValueError:
                        print("Please enter an integer(str type) for K-fold Inner cross validation")
                        exit()
                    for j, n_features in enumerate(search_list):
                        val_result = K_fold_cross_validation(train_data, train_label, inner, n_features, args)
                        acc_list[j], _, _, _, _ = Performance(val_result, train_label)
                    best_n = search_list[np.argmax(acc_list)]
            else:
                print("We decide the number of features")
                best_n = 4000
            if args.feature_selection:
                lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(train_data, np.argmax(train_label,axis=1))
                coef = np.squeeze(np.sum(np.square(np.array(lsvc.coef_)), axis=0))
                coefidx = np.argsort(coef)
                fidx = coefidx[-best_n:]
            else:
                fidx = range(0,np.shape(train_data)[1])
            if args.ensemble:
                alpha = 0.5
                final_out[test_index] = alpha*DNN(train_data[:,fidx], train_label, test_data[:,fidx], args)
                final_out[test_index] += (1-alpha)*RF(train_data[:,fidx], train_label, test_data[:,fidx])
            else:
                final_out[test_index] = DNN(train_data[:,fidx], train_label, test_data[:,fidx], args)
    accuracy, auc, precision, recall, f1 = Performance(final_out, label)
    print("Outer validation method: " + args.outer)
    print("Inner validation method: " + args.inner)
    print("Parameter search: %s" % args.search)
    print("Ensemble: %s" % args.ensemble)
    print("Feature selection: %s" % args.feature_selection)
    print("Accuracy : %f" % accuracy)
    print("Average AUC : %f" % auc)
    print("Average Precision: %f" % precision)
    print("Average Recall: %f" % recall)
    print("Average F1-score: %f" % f1)

if __name__ == "__main__":
    main()
    

