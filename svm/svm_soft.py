from svm_core import SVM

class SoftMarginSVM(SVM):
    def __init__(self, C=1.0, kernel='linear', max_iter=1000, tol=1e-3, gamma=0.1, degree=3, coef0=1, record_history=False):
        """
        Soft Margin SVM.
        Allows for misclassifications controlled by C.
        """
        super().__init__(kernel=kernel, C=C, max_iter=max_iter, tol=tol, 
                         gamma=gamma, degree=degree, coef0=coef0, record_history=record_history)
