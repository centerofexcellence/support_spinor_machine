import numpy as np

class SpinorMachine:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    def _compute_kernel(self, X, Y=None):
        if self.kernel == 'linear':
            K = np.dot(X, Y.T)
        elif self.kernel == 'rbf':
            if self.gamma == 'auto':
                self.gamma = 1.0 / X.shape[1]
            X2 = np.sum(np.square(X), axis=1)
            if Y is None:
                Y = X
            Y2 = np.sum(np.square(Y), axis=1)
            K = np.dot(X, Y.T)
            K *= -2
            K += X2[:, np.newaxis]
            K += Y2[np.newaxis, :]
            K = np.exp(-self.gamma * K)
        elif self.kernel == 'poly':
            if self.gamma == 'auto':
                self.gamma = 1.0 / X.shape[1]
            K = (np.dot(X, Y.T) + self.coef0) ** self.degree
        else:
            raise ValueError('Invalid kernel specified')
        return K

    def _objective(self, alpha, Q, c, A, b):
        return (1 / 2) * np.dot(alpha.T, np.dot(Q, alpha)) + np.dot(c.T, alpha)

    def _compute_spinor(self, X):
        K = self._compute_kernel(X, self.X)
        A = self.alpha * self.y
        B = np.zeros((len(A), len(A)), dtype=np.complex)
        for i in range(len(A)):
            for j in range(len(A)):
                B[i, j] = np.exp(1j * A[i] * A[j] * K[i, j])
        return np.sum(B, axis=0)

    def _modulate_kernel(self, K):
        f = self._compute_spinor(self.X)
        if self.kernel == 'linear':
            return np.dot(np.dot(np.conj(f), K), f.T)
        elif self.kernel == 'rbf' or self.kernel == 'poly':
            return np.exp(-self.gamma * np.abs(np.dot(np.dot(np.conj(f), K), f.T)))

    def _solve_qp(self, Q, alpha, y, C):
        n = Q.shape[0]
        eps = 1e-6
        b = 0
        passes = 0
        while passes < n:
            num_changed_alphas = 0
            for i in range(n):
                Ei = np.dot(alpha*y, Q[:,i]) + b - y[i]
                if ((y[i]*Ei < -eps) and (alpha[i] < C)) or ((y[i]*Ei > eps) and (alpha[i] > 0)):
                    j = np.random.randint(low=0, high=n)
                    while j == i:
                        j = np.random.randint(low=0, high=n)
                    Ej = np.dot(alpha*y, Q[:,j]) + b - y[j]
                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]
                    L = max(0, alpha[j] - alpha[i]) if y[i]
                    H = min(C, C + alpha[j] - alpha[i]) if y[i] == y[j] else min(C, alpha[j] - alpha[i])
                    if L == H:
                        continue
                    eta = 2 * Q[i, j] - Q[i, i] - Q[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = min(alpha[j], H)
                    alpha[j] = max(alpha[j], L)
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])
                    b1 = b - Ei - y[i] * (alpha[i] - alpha_i_old) * Q[i, i] - y[j] * (alpha[j] - alpha_j_old) * Q[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - alpha_i_old) * Q[i, j] - y[j] * (alpha[j] - alpha_j_old) * Q[j, j]
                    if (0 < alpha[i]) and (alpha[i] < C):
                        b = b1
                    elif (0 < alpha[j]) and (alpha[j] < C):
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return alpha, b

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self._fit_binary()
        else:
            self._fit_multi()

    def _fit_binary(self):
        y_ = np.where(self.y == self.classes_[0], -1, 1)
        K = self._compute_kernel(self.X)
        K_mod = self._modulate_kernel(K)
        self.alpha, self.b = self._solve_qp(K_mod, np.zeros_like(y_), y_, self.C)
        self.support_vectors = self.X[self.alpha > 1e-4]
        self.alpha = self.alpha[self.alpha > 1e-4]
        self.support_vectors_y = y_[self.alpha > 1e-4]
        self.kernel_values = K[self.alpha > 1e-4, :][:, self.alpha > 1e-4]
        self.support_vectors_alpha_y = np.outer(self.alpha * self.support_vectors_y, np.ones_like(self.alpha))

    def _fit_multi(self):
        n_classes = len(self.classes_)
        self.classifiers = []
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                idx = np.logical_or(self.y == self.classes_[i], self.y == self.classes_[j])
                X_pair = self.X[idx]
                y_pair = self.y[idx]
                y_pair = np.where(y_pair == self.classes_[i], -1, 1)
                K = self._compute_kernel(X_pair)
                K_mod = self._modulate_kernel(K)
                alpha, b = self._solve_qp(K_mod, np.zeros_like(y_pair), y_pair, self.C)
                self.classifiers.append((X_pair, y_pair, alpha, b))

    def predict(self, X):
       
        if len(self.classes_) == 2:
            K = self._compute_kernel(X, self.support_vectors)
            K_mod = self._modulate_kernel(K)
            decision_values = np.dot(K_mod, self.alpha * self.support_vectors_y) - self.b
            y_pred = np.where(decision_values < 0, self.classes_[0], self.classes_[1])
        else:
            decision_values = np.zeros((len(X), len(self.classifiers)))
            for i, (X_pair, y_pair, alpha, b) in enumerate(self.classifiers):
                K = self._compute_kernel(X, X_pair)
                K_mod = self._modulate_kernel(K)
                decision_values[:, i] = np.dot(K_mod, alpha * y_pair) - b
            y_pred = np.zeros(len(X), dtype=self.classes_.dtype)
            for i in range(len(X)):
                votes = np.zeros(len(self.classes_))
                for j, class_pair in enumerate(self.classifiers):
                    if decision_values[i, j] > 0:
                        votes[np.where(self.classes_ == class_pair[1])[0]] += 1
                    else:
                        votes[np.where(self.classes_ == class_pair[0])[0]] += 1
                y_pred[i] = self.classes_[np.argmax(votes)]
        return y_pred
