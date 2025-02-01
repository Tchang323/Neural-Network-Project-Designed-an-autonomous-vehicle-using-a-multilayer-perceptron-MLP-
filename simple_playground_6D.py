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

class Car():
    def __init__(self) -> None:
        self.radius = 6   #車體大小
        self.angle_min = -90 
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def diameter(self):
        return self.radius/2

    def reset(self): # 車輛的初始位置和角度
        self.angle = 90  # 車體初始角度+90度
        self.wheel_angle = 0 

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):  # 設定車輪轉向角度
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def getWheelAngle(self, angle):  # 設定車輪轉向角度
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)
        return self.wheel_angle
    
    def setPosition(self, newPosition: Point2D):  # 設定車的位置(xpos ypos)
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D: # 根據不同的需求返回車的中心點、右側點、左側點或前方座標位置
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius/2+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius/2+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):  # 計算車的位置
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):  # 設定車的行駛角度，確保角度再指定的範圍內
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):   # 更新車的位置和角度，模擬時間的推移
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.sin(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) / (self.radius*1.5))) / m.pi * 180

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


# 假設 MLP 類別已經定義在這裡，或是另外引用
class MLP:
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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        # 第一層隱藏層
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # 第二層隱藏層
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        # 第三層隱藏層
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        # 輸出層
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.a4 = self.sigmoid(self.z4)

        steering_angle = self.z4 * 80 - 40 # Output layer, no sigmoid for steering angle

        return steering_angle

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
        print(f"Epoch {epochs}/{epochs}, , lr: {learning_rate}, Loss: {loss}")

    def predict(self, X):
        return self.forward(X)

    
class Playground():
    def __init__(self):  # 初始化場地，包含起點線和中間的裝飾線
        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self.mlp = MLP(input_size = 3, hidden_size1=128, hidden_size2 = 32, hidden_size3 = 13, output_size = 1)
        self._setDefaultLine()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
            Line2D(18,40,30,40)   # finish line
        ]

        self.car = Car()
        self.reset()

    def _setDefaultLine(self):   # 場地的預設線條
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, 0, 6, 0),
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None
    
    def _readPathLines(self):  # 從檔案軌道座標點.txt讀取軌道的座標並設置
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    def predictAction(self, state):
        # Predict wheel angle with MLP based on sensor state
        state = np.array(state).reshape(1, -1)
        angle = self.mlp.forward(state)[0,0]
        return int(angle) # return the wheel angle for the car

    @property   # 返回車輛可以執行的動作數量
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):  # 返回觀測狀態的形狀
        return (len(self.state),)

    @ property
    def state(self):   # 車輛與障礙物或牆壁間的距離
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            ).distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    

    def _checkDoneIntersects(self):  # 檢查車輛駛否到達目的地，並檢測車輛是否有和牆壁碰撞
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # chack every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None, angle = None):
        '''
        更改此處code，依照自己的需求撰寫。
        '''
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

class SimulationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Self-Driving Car Path Simulation")

        if os.path.exists('track6D.txt'):
            os.remove('track6D.txt')

        # Initialize car 
        self.car = Car()

        # Initialize Playground
        self.playground = Playground()
        self.playground._readPathLines()
        self.state = self.playground.reset()

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

        # Plot initial path and car starting position
        for line in self.playground.lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'b-')
        # 繪製裝飾線 (start line和middle line)
        for line in self.playground.decorate_lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'k--')  # 黑色虛線表示裝飾線
        # self.ax.plot(self.playground.car_init_pos.x, self.playground.car_init_pos.y, 'go', label='start')
        self.ax.plot(0, 0, 'go', label='start')
        self.ax.legend()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Path Lines")
        self.ax.grid()
        self.ax.set_aspect('equal')  # 確保圓形不會變成橢圓
        self.canvas.draw()

        # List to store car trajectory
        self.car_positions = []

        # Initialize the circle patch for the car and add it to the plot
        self.car_circle = patches.Circle((0, 0), 3, edgecolor='black', fill=False)  # Radius = 3, black edge, no fill
        self.ax.add_patch(self.car_circle)
        
        # Load and train MLP model
        self.mlp = self.train_mlp_model("train6dAll.txt")

        # Set up timer for simulation steps
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(1000)  # Update every second

    def train_mlp_model(self, filename):
        # Load the training dataset
        data = np.loadtxt(filename)
        X_train = data[:, :-1]  # First three columns: sensor data
        y_train = data[:, -1:]  # Last column: steering angle

        # Initialize and train MLP model
        mlp = MLP(input_size=5, hidden_size1=256, hidden_size2 = 128, hidden_size3 = 16, output_size=1)
        mlp.train(X_train, y_train, learning_rate=0.001, epochs=500, batch_size=16)
        return mlp
    
    def update_simulation(self):
        # Check if simulation is done
        if self.playground.done:
            self.timer.stop()
            return

        # Get car position and display it
        car_position = self.playground.car.getPosition('center')

        # Use trained MLP model to predict steering angle
        sensor_values = np.array([car_position.x, car_position.y])
        sensor_values = np.append(sensor_values, [self.state])
        # print("sensor values: " + str(sensor_values))
        
        predicted_steering_angle = self.mlp.predict(sensor_values)[0, 0]
        action = predicted_steering_angle

        # Update simulation state
        self.state, wheel_angle = self.playground.step(action, angle = 0)

        position_text = f"State: {self.state}, Position: ({car_position.x}, {car_position.y})"
        angle_text = f"Wheel Angle: {wheel_angle}"
        self.text_edit.append(position_text)
        self.text_edit.append(angle_text)
        print(position_text)
        print(angle_text)

        # 假設 self.state 是一個列表，並且依次包含 [前方距離, 右方距離, 左方距離]
        front_distance, right_distance, left_distance = self.state[0], self.state[1], self.state[2]
        state_values = f"{front_distance} {right_distance} {left_distance}"

        # 構建輸出的數據行，並寫入檔案
        with open('track6D.txt', 'a') as file:
            line = f"{car_position.x} {car_position.y} {state_values} {wheel_angle}\n"
            file.write(line)

        # Update the car's path on the plot
        self.car_positions.append((car_position.x, car_position.y))
        x_vals, y_vals = zip(*self.car_positions)
        self.ax.plot(x_vals, y_vals, 'g-')
        self.ax.plot(car_position.x, car_position.y, 'ro')
        
        # Update the car circle to follow the car's position
        self.car_circle.center = (car_position.x, car_position.y)

        self.canvas.draw()

# Run the application
app = QApplication(sys.argv)
window = SimulationApp()
window.show()
sys.exit(app.exec_())

