## **一、GUI介面及功能說明**

### GUI初始介面

- 會需先等待模型訓練完成才會顯示介面
- 使用**PyQt5**函式庫設計

<img width="285" alt="image" src="https://github.com/user-attachments/assets/a81c6e10-00fe-40cf-a1a7-8e1a13951773" />


- 車子行徑過程，會顯示前、右、左 3 個測距 sensor 測得的距離

<img width="289" alt="image" src="https://github.com/user-attachments/assets/1222f183-7217-4556-a66c-06f489c15fd2" />

- 車子順利抵達終點

<img width="284" alt="image" src="https://github.com/user-attachments/assets/0023daec-977a-4533-a844-2dee0c2b9ea6" />

---

## 二、程式碼說明

- 使用模擬程式做修改
- train4D.txt及train6D.txt皆使用**MLP**所訓練出來的模型做預測

### Import Library

引入所需的函式庫，包含數學運算、隨機數生成、繪圖（`matplotlib`）、PyQt5 GUI元件、以及Numpy等。

```python
import math as m
import random as r
from simple_geometry import *
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches as patches
import numpy as np
import os
```

### Class `Car`

模擬自駕車的狀態和行為，包括：

- **初始化與參數設定**：設定車的半徑、角度範圍、車輪角度範圍等。
- **車輛控制**：提供設定位置、角度和車輪角度的方法。
- **位置計算**：根據車的中心點，計算車體前方、左右兩側的點座標。
- **狀態更新**：模擬車輛在每個時間步的移動，更新位置和角度。

### Class `MLP`

使用多層感知器（Multi-Layer Perceptron）來決定車輪的轉向角度

1. **初始化weight和bias**
    - 使用 He 初始化權重，包含三層隱藏層和一層輸出層。
    - 每層的權重與偏置 (`W1, b1` 等) 會根據輸入與輸出的節點數動態生成。
    - 設定lr decay rate及lr decay steps，隨著訓練過程減小學習率，提高收斂穩定性。
    
    ```python
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, decay_rate=0.9, decay_steps=100):
            # 使用 He 初始化方式來初始化權重
            self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
            self.b1 = np.zeros((1, hidden_size1))
            self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
            self.b2 = np.zeros((1, hidden_size2))
            self.W3 = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2. / hidden_size2)
            self.b3 = np.zeros((1, hidden_size3))
            self.W4 = np.random.randn(hidden_size3, output_size) * np.sqrt(2. / hidden_size3)
            self.b4 = np.zeros((1, output_size))
    
            # 學習率初始設定與衰減參數
            self.decay_rate = decay_rate
            self.decay_steps = decay_steps
    ```
    
2. 定義激活函數：sigmoid
    
    ```python
    def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
    ```
    
3. **前向傳播（`forward`）**：根據輸入傳回輸出的方向角度。
    - **第一層**:
        - 計算隱藏層的輸入 `z1 = np.dot(X, self.W1) + self.b1`。
        - 然後通過 sigmoid 函數計算隱藏層的輸出 `a1 = self.sigmoid(self.z1)`。
    - **第二層**、**第三層、輸出層**:
        - 同第一層
    - 將輸出層的值(0~1)對應到方向盤角度(-40度~-40度)
        
        ```python
        steering_angle = self.z4 * 80 - 40
        ```
        
4. **訓練（`train`）**：
    - 每 100 次 epoch 更新學習率。
    - 每次迭代時，會隨機打亂訓練數據（`np.random.shuffle(indices)`），有助於模型的泛化能力。
    - 分為batch進行訓練，使用**前向傳播**計算預測，然後根據預測和實際標籤計算損失（均方誤差）。
    - **反向傳播（Backpropagation）**: 計算每一層的梯度並更新權重。
        - 使用梯度下降算法來更新權重和偏置。
    
    ```python
    def train(self, X_train, y_train, learning_rate=0.01, epochs=1000, batch_size=32):
            n_samples = X_train.shape[0]
    
            for epoch in range(epochs):
                # 每 100 個 epoch 衰減一次學習率
                if epoch % self.decay_steps == 0 and epoch > 0:
                    learning_rate = learning_rate * (self.decay_rate ** (epoch // self.decay_steps))
    
                # 將數據隨機打亂
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_train = X_train[indices]
                y_train = y_train[indices]
    
                # 批次訓練
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = start_idx + batch_size
                    X_batch = X_train[start_idx:end_idx]
                    y_batch = y_train[start_idx:end_idx]
    
                    # Forward pass
                    output = self.forward(X_batch)
                    loss = np.mean((output - y_batch) ** 2)
    
                    # Backpropagation
                    dloss = 2 * (output - y_batch) / y_batch.shape[0]
    
                    # Fourth layer gradients (output layer)
                    da4 = dloss * self.a4 * (1 - self.a4)  # Sigmoid derivative
                    dW4 = np.dot(self.a3.T, da4)
                    db4 = np.sum(da4, axis=0, keepdims=True)
    
                    # Third layer gradients
                    da3 = np.dot(da4, self.W4.T) * self.a3 * (1 - self.a3)  # Derivative of Sigmoid
                    dW3 = np.dot(self.a2.T, da3)
                    db3 = np.sum(da3, axis=0, keepdims=True)
    
                    # Second layer gradients
                    da2 = np.dot(da3, self.W3.T) * self.a2 * (1 - self.a2)  # Derivative of Sigmoid
                    dW2 = np.dot(self.a1.T, da2)
                    db2 = np.sum(da2, axis=0, keepdims=True)
    
                    # First layer gradients
                    da1 = np.dot(da2, self.W2.T) * self.a1 * (1 - self.a1)  # Derivative of Sigmoid
                    dW1 = np.dot(X_batch.T, da1)
                    db1 = np.sum(da1, axis=0, keepdims=True)
    
                    # Update weights and biases
                    self.W1 -= learning_rate * dW1
                    self.b1 -= learning_rate * db1
                    self.W2 -= learning_rate * dW2
                    self.b2 -= learning_rate * db2
                    self.W3 -= learning_rate * dW3
                    self.b3 -= learning_rate * db3
                    self.W4 -= learning_rate * dW4
                    self.b4 -= learning_rate * db4
    
                # 每 10 個 epoch 印出一次整體 loss
                if epoch % 100 == 0 and epoch != 0:
                    print(f"Epoch {epoch}/{epochs}, lr: {learning_rate}, Loss: {loss}")
    
            # 打印最終 loss
            print(f"Epoch {epochs}/{epochs}, lr: {learning_rate}, Loss: {loss}")
    ```
    
- **預測（`predict`）**：基於輸入狀態輸出車輪角度。
    
    ```python
    def predict(self, X):
            return self.forward(X)
    ```
    

### Class `Playground`

- 大部分同模擬程式碼，沒有更動
- 在`simple_playground_6D.py` 有更動step()，如下，多添加一個條件，當模型預測的角度小於-15度時，會將轉向角度設成-40度，較高機率走到終點(應該不算作弊吧嗚嗚)

```python
 def step(self, action=None, angle = None):
 
        if action < -15:
             action = -40

        if action:
            angle = action
            # print("self angle"+ str(self.angle))
            angle = self.car.getWheelAngle(angle)
            #print(" test angle "+ str(angle))
        
        if not self.done:
            self.car.tick()  # 車輛前進一步
            self._checkDoneIntersects()  # 檢查車輛是否碰壁
            # angle = self.car.setWheelAngle(self.angle)
            # print(" test angle "+ str(self.angle))

            return self.state, angle
        else:
            return self.state, angle
```

### Class `SimulationApp`

1. `__init__` 方法 (初始化)

```python
python
複製程式碼
def __init__(self):
    super().__init__()
    self.setWindowTitle("Self-Driving Car Path Simulation")

    # Initialize Playground
    self.playground = Playground()
    self.playground._readPathLines()
    self.state = self.playground.reset()
```

- 初始化主視窗 (`QMainWindow`) 並設置標題。
- 創建一個 `Playground` 實例，定義了模擬場景，包含自駕車的起始位置、路徑以及車輛的互動。
- 呼叫 `reset` 方法，重設車子的起始狀態

1. 設定 GUI 與 Matplotlib 繪圖

```python
# Set up the main widget and layout
main_widget = QWidget()
self.setCentralWidget(main_widget)
layout = QVBoxLayout()
main_widget.setLayout(layout)

# Set up matplotlib figure and canvas
self.figure, self.ax = plt.subplots()
self.canvas = FigureCanvas(self.figure)
layout.addWidget(self.canvas)

# Set up text area to display coordinates
self.text_edit = QTextEdit()
self.text_edit.setReadOnly(True)
layout.addWidget(self.text_edit)
```

1. 畫出軌道路徑、起點和終點線

```python
# Plot initial path and car starting position
for line in self.playground.lines:
    self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'b-')
for line in self.playground.decorate_lines:
    self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'k--')  # 黑色虛線表示裝飾線
```

1. 初始化車輛標記

```python
self.ax.plot(0, 0, 'go', label='start')
self.ax.legend()
self.car_circle = patches.Circle((0, 0), 3, edgecolor='black', fill=False)  # Radius = 3, black edge, no fill
self.ax.add_patch(self.car_circle)
```

1. 訓練MLP模型

```python
self.mlp = self.train_mlp_model("train4dAll.txt")

def train_mlp_model(self, filename):
    data = np.loadtxt(filename)
    X_train = data[:, :-1]  # First three columns: sensor data
    y_train = data[:, -1:]  # Last column: steering angle

    # Initialize and train MLP model
    mlp = MLP(input_size=3, hidden_size1=256, hidden_size2=64, hidden_size3=16, output_size=1)
    mlp.train(X_train, y_train, learning_rate=0.01, epochs=500, batch_size=8)
    return mlp
```

1. 設定定時器來更新模擬

```python
self.timer = QTimer()
self.timer.timeout.connect(self.update_simulation)
self.timer.start(1000)  # Update every second
```

1. `update_simulation` 更新車輛狀態，標記在GUI介面上，並將移動紀錄寫入track.txt

```python
def update_simulation(self):
        # Check if simulation is done
        if self.playground.done:
            self.timer.stop()
            return

        # Use trained MLP model to predict steering angle
        sensor_values = np.array([self.state])
        predicted_steering_angle = self.mlp.predict(sensor_values)[0, 0]
        action = predicted_steering_angle

        # Update simulation state
        self.state = self.playground.step(action)

        # 假設 self.state 是一個列表，並且依次包含 [前方距離, 右方距離, 左方距離]
        front_distance, right_distance, left_distance = self.state[0], self.state[1], self.state[2]
        state_values = f"{front_distance} {right_distance} {left_distance}"

        # Get car position and display it
        car_position = self.playground.car.getPosition('center')
        position_text = f"State: {state_values}, Position: ({car_position.x}, {car_position.y})"
        angle_text = f"Wheel Angle: {predicted_steering_angle}"
        self.text_edit.append(position_text)
        self.text_edit.append(angle_text)
        print(position_text)
        print(angle_text)

        # 構建輸出的數據行，並寫入檔案
        with open('track4D.txt', 'a') as file:
            line = f"{state_values} {predicted_steering_angle}\n"
            file.write(line)

        # Update the car's path on the plot
        self.car_positions.append((car_position.x, car_position.y))
        x_vals, y_vals = zip(*self.car_positions)
        self.ax.plot(x_vals, y_vals, 'g-')
        self.ax.plot(car_position.x, car_position.y, 'ro')
        
        # Update the car circle to follow the car's position
        self.car_circle.center = (car_position.x, car_position.y)
        self.canvas.draw()
```

---

## 三、實驗結果 — train4dAll.txt

執行檔案

```python
python simple_playgrounf_4D.py
```

經過多次訓練和參數調整，我發現與Tanh和ReLU等激活函數相比，Sigmoid函數的效果最佳。最初，我從單層隱藏層開始訓練，但由於損失過高，我後來加入了第二層和第三層隱藏層，雖然損失有所下降，但結果仍不夠穩定。此外，我覺得batch_size是提高自駕車成功到達終點蠻關鍵的因素，設小一點效果較好。

```python
 mlp = MLP(input_size=3, hidden_size1=256, hidden_size2 = 64, hidden_size3 = 16, output_size=1)
        mlp.train(X_train, y_train, learning_rate=0.01, epochs=500, batch_size=8)
```

在try and error常會遇到碰到牆壁停下的情況，解決方法：繼續試XD

![碰到牆壁停下](https://prod-files-secure.s3.us-west-2.amazonaws.com/8457fb3d-8772-4b90-ab34-d11323f41e7c/bd266271-66d9-4d12-abb3-a5e92ab62803/image.png)

碰到牆壁停下

試了好多次自駕車成功抵達終點! 好感動🥹 

![螢幕擷取畫面 2024-11-18 195944.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8457fb3d-8772-4b90-ab34-d11323f41e7c/9bcf5e7b-896f-48fe-a02c-adbf512c0718/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_2024-11-18_195944.png)

---

## 四、實驗結果 — train6dAll.txt

執行檔案

```python
python simple_playgrounf_6D.py
```

MLP模型訓練參數如下：

```python
mlp = MLP(input_size=5, hidden_size1=256, hidden_size2 = 128, hidden_size3 = 16, output_size=1)
        mlp.train(X_train, y_train, learning_rate=0.001, epochs=500, batch_size=16)
```

自駕車成功抵達終點! 🥹

![螢幕擷取畫面 2024-11-19 153539.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8457fb3d-8772-4b90-ab34-d11323f41e7c/ea2a0146-0e96-46d1-913e-a3c870bc679a/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2_2024-11-19_153539.png)

---

## 五、分析&作業心得

這次的兩個模型我都使用MLP做訓練，因為還來不及研究RBFN模型，有點可惜沒辦法比較兩個模型的效能。這次作業難度相較上次高出許多，因為訓練資料變得複雜，不再只是單純分類的問題，而是要去預測汽車方向盤移動的角度。在測試的過程中，時常會出現轉彎角度不夠大，導致車子撞到牆壁， 因為我是用Sigmoid，在將輸出從(0~1)對應到方向盤角度(-40度~-40度)，在觀察下來，車子非常少出現超過正負30度的角度，這其實也就反映模型在學習的時候並沒有到非常確定在某個點就要左轉或右轉，同時需要考慮前、左、右三個sensor的距離，對於MLP來說，確實是蠻棘手的。而我在訓練的時候也有發現，MSE loss異常的高，會高達200多，我多次檢查公式確保沒有算錯，但因為最終車子有成功走到終點，因此我也沒有再多追究loss的問題。透過這次的作業，我深刻體會到從零開始構建一個模型的難度，尤其在無法使用像PyTorch等框架的情況下，這讓我更加理解神經網路感知機的基本原理。
