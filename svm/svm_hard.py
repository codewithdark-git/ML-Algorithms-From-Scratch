from svm_core import SVM

class HardMarginSVM(SVM):
    def __init__(self, kernel='linear', max_iter=1000, tol=1e-3, gamma=0.1, degree=3, coef0=1, record_history=False):
        """
        Hard Margin SVM. 
        Effectively a Soft Margin SVM with a very large C (penalty for proper classification).
        """
        super().__init__(kernel=kernel, C=1e10, max_iter=max_iter, tol=tol, 
                         gamma=gamma, degree=degree, coef0=coef0, record_history=record_history)
