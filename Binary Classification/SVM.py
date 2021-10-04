import numpy as np                            # Thư viện tính toán với ma trận
from sklearn import linear_model              # Thư viện linear_model
from sklearn.metrics import accuracy_score    # Hàm đánh giá độ chính xác
from sklearn import svm                       # SVM
from matplotlib.pyplot import imread          # Load ảnh
import matplotlib.pyplot as plt
from sklearn import metrics
np.random.seed(1)
# Load ảnh
# BG(187L+73S=260) MZ(175L+45S=220)
pathBG = 'C:/Users/15108/Desktop/New data/Train Resized/mauBG/' 
pathMZ = 'C:/Users/15108/Desktop/New data/Train Resized/mauMZ/' 
train_BG = np.arange(1, 141)               # Chọn 140 ảnh đầu là tập train BG
test_BG = np.arange(141, 169)              # Chọn 28 ảnh còn lại tập test BG      
train_MZ= np.arange(1, 141)                # Chọn 140 ảnh đầu là tập train MZ
test_MZ= np.arange(141, 167)               # Chọn 26 ảnh còn lại tập test MZ
D = 110*110                                # Kích thước sau khi resize
d = 1000                                   # Kích thước mới dùng projection
ProjectionMatrix = np.random.randn(D, d)   # Khởi tạo ma trận chiếu ngẫu nhiên 
# Hàm đọc tên ảnh và lưu vào danh sách
def build_list_fn(pre,imgs,path): 
    list_fn = []                           # Khởi tạo list rỗng
    for im in imgs:                        # Thêm ảnh vào list
        fn = path + pre  + ' '+ str(im) + '.png' 
        list_fn.append(fn)           
    return list_fn 
# Hàm chuyển đổi ảnh màu thành ảnh xám
def rgb2gray(rgb):
    # Y' = 0.299R + 0.587G + 0.114B công thức chuyển đổi ảnh màu sang xám
    return rgb[:,:,0]*.299 + rgb[:,:,1]*.587 + rgb[:, :,2]*.114
# Hàm vetorize ảnh
def vectorize_img(filename):        
    rgb = imread(filename)      # Load ảnh    
    gray = rgb2gray(rgb)        # Chuyển đổi sang thang màu xám 
    im_vec = gray.reshape(1, D) # Chuyển thành vector hàng
    return im_vec 
# Xây dựng ma trận dữ liệu và chiếu xuống PM để giảm chiều dữ liệu
def build_data_matrix(imgs_BG,imgs_MZ):
    total_imgs = imgs_BG.shape[0] + imgs_MZ.shape[0]    # Tổng số lượng ảnh
    X_full = np.zeros((total_imgs, D))                  # Khởi tạo dữ liệu X          
    y = np.hstack((np.zeros((int(total_imgs/2), )), np.ones(int((total_imgs/2), ))))    
    list_fn_BG = build_list_fn('Bglri',imgs_BG,pathBG)  # Build ảnh Billgate
    list_fn_MZ = build_list_fn('Mzlri',imgs_MZ,pathMZ)  # Buil ảnh Mac Zuckerberg      
    list_fn = list_fn_BG + list_fn_MZ  
    # Chuyển đổi sang thang màu xám từng ảnh và giảm chiều dữ liệu
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])
    X = X_full@ProjectionMatrix
    return (X, y)
# Build tập train
(X_train_full, y_train) = build_data_matrix(train_BG,train_MZ)
x_mean = X_train_full.mean(axis = 0)
x_var  = X_train_full.var(axis = 0)
def feature_extraction(X):
    return (X - x_mean)/x_var     
# Chuẩn hóa dữ liệu tập train và tập test
X_train = feature_extraction(X_train_full)
X_train_full = None
(X_test_full, y_test) = build_data_matrix(test_BG, test_MZ)
X_test = feature_extraction(X_test_full)
X_test_full = None 
# 4 ảnh để test
fn1 = pathMZ + 'Mzlri 158.png'
fn2 = pathMZ + 'Mzlri 160.png'
fn3 = pathBG + 'Bglri 155.png'
fn4 = pathBG + 'Bglri 153.png'
# Hàm xử lý dữ liệu đầu vào (chuyển thành ảnh xám và giảm chiều dữ liệu)
def feature_extraction_fn(fn):    
    im = vectorize_img(fn)    
    im1 = np.dot(im, ProjectionMatrix)     
    return feature_extraction(im1)
# Hàm trực quan 
def display_result_SVM(fn):
    rgb = imread(fn)         # Đọc ảnh
    plt.axis('off')          # Tắt các trục tung hoành
    plt.imshow(rgb)          # Show ảnh
    plt.show()               # Show tất cả
# Train với SVM
def BG_MZ_SVM(X_train,y_train):
    y1 = y_train.reshape((X_train.shape[0],))     # Reshape y_train đúng chiều
    X1 = X_train                                  # Tập dữ liệu train
    clf = svm.SVC(kernel='linear')                # Dùng thư viện SVM linear 
    clf.fit(X1, y1)                               
    # Đánh giá bằng accuracy 30 ảnh trong tập test
    y_pred = clf.predict(X_test)
    print("Accuracy:",100*metrics.accuracy_score(y_test, y_pred),'%')
    # Load 4 ảnh từ tập test để thử
    x1 = feature_extraction_fn(fn1)
    p1 = clf.predict(x1)
    if p1 == [0]:
        print('BILL GATE')
    else:
        print('MARK ZUCKERBERG')
    x2 = feature_extraction_fn(fn2)
    p2 = clf.predict(x2)
    if p2 == [0]:
        print('BILL GATE')
    else:
        print('MARK ZUCKERBERG')
    x3 = feature_extraction_fn(fn3)
    p3 = clf.predict(x3)
    if p3 == [0]:
        print('BILL GATE')
    else:
        print('MARK ZUCKERBERG')
    x4 = feature_extraction_fn(fn4)
    p4 = clf.predict(x4)
    if p4 == [0]:
        print('BILL GATE')
    else:
        print('MARK ZUCKERBERG')
    # Trực quan hóa ảnh test
    display_result_SVM(fn1) 
    display_result_SVM(fn2)
    display_result_SVM(fn3)
    display_result_SVM(fn4)
