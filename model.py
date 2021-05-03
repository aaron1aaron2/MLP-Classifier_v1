import numpy as np

# —中間層—
class BaseLayer:
    def __init__(self, name, wb_width, n_upper, n):
        self.name = name
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)

        self.h_w = np.zeros(( n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8
        
    def update(self, eta):      
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
            
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

    def get_weight(self):
        return {self.name: {'weight':self.w, 'bias':self.b}}
        
# -- 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)  # ReLU

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)  # ReLU的微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T) 

# -- 輸出層 --
class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # softmax 函數

    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T) 

# -- 丟棄層 --
class Dropout:
    def __init__(self, name, dropout_ratio):
        self.name = name
        self.dropout_ratio = dropout_ratio  # 丟棄率

    def forward(self, x, is_train):  # is_train: 訓練時為 True
        if is_train:
            rand = np.random.rand(*x.shape)  # 亂數的矩陣
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)  # 1:有効 0:無効
            self.y = x * self.dropout  # 隨時關閉神經元
        else:
            self.y = (1-self.dropout_ratio)*x
        
    def backward(self, grad_y):
        self.grad_x = grad_y * self.dropout
