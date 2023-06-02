import numpy as np

# Computes multiplication of numpy arrays with matrix label code
# without actually constructing the matrices themselves
# See theory/04_fast_matrix_label_code.ipynb

# fmlc left = fast matrix label code LEFT multiplication
# PI[y]@z
# z is a k-1 COLUMN vector

def fmlc_left(z, Y, num_classes):
    # z has shape (n, num_classes-1)
    # Y has shape (n, num_classes), and is the one-hot label embedding of the labels
    Ytrim = Y[:,:-1]
    C = np.sum(Ytrim*z,axis=1).reshape(-1,1)
    return z - C - Ytrim*C

    
def fmlc_left_with_integer_labels(z,y, num_classes):
    # z has shape (n, num_classes-1)
    # y has shape (n,) and takes values in [0,1,..., num_classes-1]
    # basically converts y to one-hot vectors and reduce to the previous problem
    return fmlc_left(z, np.eye(num_classes)[y], num_classes)




# fmlc right = fast matrix label code RIGHT multiplication
# z@PI[y]
# z is a k-1 ROW vector

def fmlc_right(z, Y, num_classes):
    # z has shape (n, num_classes-1)
    # Y has shape (n, num_classes), and is the one-hot label embedding of the labels
    Ytrim = Y[:,:-1]
    C = np.sum(Ytrim*z,axis=1).reshape(-1,1)
    D = np.sum(z,axis=1).reshape(-1,1)
    return z - Ytrim*D - Ytrim*C

    
def fmlc_right_with_integer_labels(z,y, num_classes):
    # z has shape (n, num_classes-1)
    # y has shape (n,) and takes values in [0,1,..., num_classes-1]
    # basically converts y to one-hot vectors and reduce to the previous problem
    return fmlc_right(z, np.eye(num_classes)[y], num_classes)


# multiply by the R matrix on the right
# recall that R = [-ones Id] is a (k-1)-by-k matrix
# if z is a (k-1)-dimensional row vector
# then zR is a k-dimensional row vector

def Rmat_right(z):
    # z has shape (n,num_classes-1)
    return np.hstack((-z, np.sum(z,axis=1)w.reshape(-1,1)))