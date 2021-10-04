import numpy as np                            # Thư viện tính toán với ma trận
import matplotlib.pyplot as plt               # Thư viện trực quan hóa dữ liệu
from sklearn import neighbors                 # Thư viện KNN
import pandas as pd                           # Thư viện đọc dữ liệu
from sklearn.model_selection import train_test_split  # Hàm tách dữ liệu
from sklearn.metrics import accuracy_score    # Hàm đánh giá độ chính xác
from sklearn import linear_model              # Linear model
from sklearn.svm import SVC                   # SVM
from sklearn.naive_bayes import MultinomialNB # Bayes
from scipy import sparse                      # Lưu ma trận dạng Sparse
from sklearn.naive_bayes import GaussianNB    # Phân phối GaussianNB
from sklearn import metrics                   # Thư viện metrics đánh giá model
import matplotlib.pyplot as plt               # Thư viện trực quan hóa
from sklearn import svm                       
# Load data 
iris_cols= ['CDCH','CRCH' ,'CDDH','CRDH','PLH']
iris_sam = pd.read_csv('iris.data', sep=',', names=iris_cols, encoding='latin-1')
X0 = iris_sam.to_numpy()                 # Chuyển dữ liệu thành matrix
N = X0.shape[0]                          # Số lượng mẫu
iris_feature = X0[:,0:4]                 # Ma trận chứa feature vector của hoa
d = iris_feature.shape[1]                # Số chiều dữ liệu
labels_count= X0[:,-1]                   # Nhãn hoa Iris
C=3   
labels_name= ['Iris-setosa','Iris-versicolor','Iris-virginica'] 
# Gán nhãn cho hoa Iris là 0, 1, 2
labels_feature = []
for lb in labels_count:
    if lb == 'Iris-setosa':
        labels_feature.append(0)
    elif lb == 'Iris-versicolor':
        labels_feature.append(1)
    else: 
        labels_feature.append(2)
labels_feature = np.array(labels_feature)
# Chuẩn hóa dữ liệu
iris_feature_nor = np.copy(iris_feature)
Min_col = np.min(iris_feature,0)                    # Max của các feature
Max_col = np.max(iris_feature,0)                    # Min của các feature
Max_Min = Max_col-Min_col                           # Max trừ Min 
for i in range(N):                                  # Chuẩn hóa theo Rescaling
    for j in range(d):
        iris_feature_nor[i,j] = (iris_feature_nor[i,j]-Min_col[j])/Max_Min[j]
iris_feature_nor=np.round(iris_feature_nor.tolist(),3) # Làm tròn dữ liệu  
# Tách dữ liệu thành 2 tập train và set tỉ lệ 8:2
X_train, X_test, y_train, y_test = train_test_split(
     iris_feature_nor, labels_feature, test_size=30)
# Hàm tạo confusion matrix 
def confusion_matrix(y_true, y_pred):
    N = np.unique(y_true).shape[0]    # Số lượng Class
    cm = np.zeros((N, N))             # Khởi tạo ma trận Confusion matrix
    for n in range(y_true.shape[0]):
        cm[y_true[n], y_pred[n]] += 1 
    return cm   
# KNN
# Chọn 3 điểm gần nhất và tính khoảng cách theo chuẩn 2
def KNN_lib(X_train,y_train):
    # Sử dụng thư viện scikit-learn KNN để phân loại 
    clf = neighbors.KNeighborsClassifier(n_neighbors = 3, p = 2, weights = 'distance')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Kết quả 30 điểm dữ liệu lấy từ tập test
    print("Print results for 30 test data points:")
    print("Predicted labels: ", y_pred[0:30])
    print("Ground truth    : ", y_test[0:30])
    # Đánh giá kết quả bằng hàm accuracy_score
    print("Accuracy of 3NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
    cnf_matrix = confusion_matrix(y_test, y_pred)    # Tạo CM
    print('Confusion matrix:')
    print(cnf_matrix)    
    print('Accuracy:', 100*(np.diagonal(cnf_matrix).sum()/cnf_matrix.sum()),'%')
    normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
    print('\nConfusion matrix (with normalization):')
    print(normalized_confusion_matrix) 
    print('Precision Recall and F1-Score')
    print(metrics.classification_report(y_test, y_pred, digits=3))
    print(metrics.classification_report(y_test, y_pred, digits=3)) 
# Linear Regression      
def LR_vec(X_train,y_train):
    # Sử dụng công thức từ công thức rút ra từ tính đạo hàm trực tiếp
    X_train_lm = np.concatenate((np.ones((X_train.shape[0],1)),X_train),axis=1)
    A = X_train_lm.T@X_train_lm                  # Nhân 2 ma trận X.T và X
    b = X_train_lm.T@y_train.reshape(120,1)      # Nhân 2 ma trận X.T và y
    w = np.linalg.pinv(A)@b                 # Tính w bằng nhân 2 ma trận A-1 và b
    # Sử dụng thư viện scikit-learn
    regr = linear_model.LinearRegression(fit_intercept=False)  # Gồm bias
    regr.fit(X_train_lm, y_train.reshape(X_train_lm.shape[0],1))
    print( 'Solution found by scikit-learn: ', regr.coef_ )
    print( 'Solution by vectorized: ', w.T)
    # Kết quả 30 điểm dữ liệu lấy từ tập test
    X_test_lm =  np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)
    y_pred =  np.int32(np.round(X_test_lm@w))      
    print("Print results for 30 test data points:")
    print("Predicted labels: ", y_pred.T[0])
    print("Ground truth    : ", y_test)
    print("Accuracy of LM_vec: %.2f %%" %(100*accuracy_score(y_test,y_pred.T[0])))
    cnf_matrix = confusion_matrix(y_test, y_pred.T[0])  
    print('Confusion matrix:')
    print(cnf_matrix)    
    print('Accuracy:', 100*(np.diagonal(cnf_matrix).sum()/cnf_matrix.sum()),'%')
    normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
    print('\nConfusion matrix (with normalization:)')
    print(normalized_confusion_matrix)
    print('Precision Recall and F1-Score')
    print(metrics.classification_report(y_test, y_pred, digits=3)) 
# MulticLass SVM dùng vectorized
def svm_loss_vectorized(W, X, y, reg):
    d, C = W.shape                 # Chiều và số Class
    _, N = X.shape                 # N là tập mẫu
    loss = 0                       # Khởi tạo giá trị hàm mất mát bằng 0
    dW = np.zeros_like(W)          # Khởi tạo ma trận đạo hàm hàm mất mát
    Z = W.T@X                      # Score matrix   
    correct_class_score = np.choose(y, Z).reshape(N,1).T      
    margins = np.maximum(0, Z - correct_class_score + 1) # Higne loss
    margins[y, np.arange(margins.shape[1])] = 0          
    loss = np.sum(margins, axis = (0, 1))
    loss /= N 
    loss += 0.5 * reg * np.sum(W * W)
    
    F = (margins > 0).astype(int)
    F[y, np.arange(F.shape[1])] = np.sum(-F, axis = 0)
    dW = X@F.T/N + reg*W
    return loss, dW
# Mini-Batch
def multiclass_svm_GD(X, y, Winit, reg, lr=.1, \
        batch_size = 100, num_iters = 100):
    W = Winit                               # Giá trị đầu tiên ma trận W
    loss_history = np.zeros((num_iters))    # Khởi tạo hàm mất mát
    for it in range(num_iters):             
        # Ngẫu nhiên chọn 1 batch của X
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]
        # Tính hàm mất mát và đạo hàm của nó bằng vectorized
        loss_history[it], dW =svm_loss_vectorized(W, X_batch, y_batch, reg)
        W -= lr*dW                          # Công thức cập nhật W
    return W, loss_history 
def multi_SVM_vec(X_test,y_test):
    C = 3                                 # Số class
    reg = .1                              # Hệ số regularization
    Winit = np.random.randn(d, C)         # Khởi tạo ma trận W ngẫu nhiên
    W, loss_history = multiclass_svm_GD(X_train.T, y_train, Winit, reg)
    Z=W.T@X_test.T                        # Score matrix
    y_pred = np.argmax(Z, axis = 0)       # Tìm chỉ số có giá trị lớn nhất 
    print("Print results for 30 test data points:")
    print("Predicted labels: ", y_pred)
    print("Ground truth    : ", y_test)
    print("Accuracy of multi_SVM_vec: %.2f %%" %(100*accuracy_score(y_test,y_pred)))
    cnf_matrix = confusion_matrix(y_test, y_pred)    
    print('Confusion matrix:')
    print(cnf_matrix)    
    print('Accuracy:', 100*(np.diagonal(cnf_matrix).sum()/cnf_matrix.sum()),'%')
    normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
    print('\nConfusion matrix (with normalization:)')
    print(normalized_confusion_matrix)   
    print('Precision Recall and F1-Score')
    print(metrics.classification_report(y_test, y_pred, digits=3)) 
    # Thư viện scikit-learn
    svc = svm.SVC(kernel='linear').fit(X_train, y_train)    
    lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
    y_pred1 = svc.predict(X_test)    
    y_pred2 = lin_svc.predict(X_test)
    print("SVC with linear kernel : ", y_pred1)  
    print("LinearSVC linear kernel: ", y_pred2)
    print("Accuracy of multi_SVM_vec: %.2f %%" %(100*accuracy_score(y_test,y_pred1)))    
    print("Accuracy of multi_SVM_vec: %.2f %%" %(100*accuracy_score(y_test,y_pred2)))
# Dùng 2 feature 3 và 4 để train
def multi_SVM_2(X_train,y_train):
    X = X_train[:,2:4]              # Chọn 2 feature cuối của X để train
    plt.figure(figsize=(10, 8))     # Tạo một figure có kích thước 10,8
    # Tạo biểu đồ phân tán với 3 màu và 3 ký hiệu khác nhau
    for i, c, s in (zip(range(C), ['b', 'g', 'r'], ['o', '^', '*'])):
        ix = y_train == i 
        plt.scatter(X[:, 0][ix], X[:, 1][ix], color=c, marker=s, s=60, label=labels_name[i])
    plt.legend(loc=2, scatterpoints=1)           # Tạo legend (nhãn)
    plt.xlabel("feature 1-" + iris_cols[2])      # Tạo label cho trục x
    plt.ylabel("feature 2-" + iris_cols[3])      # Tạo label cho trục y
    plt.show()                                   # Show tất cả
# Bayes
def Bayes_Gauss(X_train,y_train):
    gnb = GaussianNB()                                  # Thư viện Gauss
    y_pred = gnb.fit(X_train, y_train).predict(X_test)  # Dự đoán tập test
    print("Print results for 30 test data points:")
    print("Predicted labels: ", y_pred)
    print("Ground truth    : ", y_test)
    print("Accuracy of Bayes_Gauss: %.2f %%" %(100*accuracy_score(y_test,y_pred)))
    cnf_matrix = confusion_matrix(y_test, y_pred)    
    print('Confusion matrix:')
    print(cnf_matrix)    
    print('Accuracy:', 100*(np.diagonal(cnf_matrix).sum()/cnf_matrix.sum()),'%')
    normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
    print('\nConfusion matrix (with normalization:)')
    print(normalized_confusion_matrix)
    print('Precision Recall and F1-Score')
    print(metrics.classification_report(y_test, y_pred, digits=3))

