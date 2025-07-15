'''
豆包回应测试：
https://github.com/AnaOnTram/Bouncing-Balls-Test
Write a Python program that shows 20 balls bouncing inside a spinning heptagon:
- All balls have the same radius.
- All balls have a number on it from 1 to 20.
- All balls drop from the heptagon center when starting.
- Colors are: #f8b862, #f6ad49, #f39800, #f08300, #ec6d51, #ee7948, #ed6d3d, #ec6800, #ec6800, #ee7800, #eb6238, #ea5506, #ea5506, #eb6101, #e49e61, #e45e32, #e17b34, #dd7a56, #db8449, #d66a35
- The balls should be affected by gravity and friction, and they must bounce off the rotating walls realistically. There should also be collisions between balls.
- The material of all the balls determines that their impact bounce height will not exceed the radius of the heptagon, but higher than ball radius.
- All balls rotate with friction, the numbers on the ball can be used to indicate the spin of the ball.
- The heptagon is spinning around its center, and the speed of spinning is 360 degrees per 5 seconds.
- The heptagon size should be large enough to contain all the balls.
- Do not use the pygame library; implement collision detection algorithms and collision response etc. by yourself. The following Python libraries are allowed: tkinter, math, numpy, dataclasses, typing, sys.
- All codes should be put in a single Python file.

'''
import tkinter as tk
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Vector:
    x: float
    y: float

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar)

    def dot(self, other: 'Vector') -> float:
        return self.x * other.x + self.y * other.y

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> 'Vector':
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)

    def rotate(self, angle: float) -> 'Vector':
        cos = math.cos(angle)
        sin = math.sin(angle)
        return Vector(
            self.x * cos - self.y * sin,
            self.x * sin + self.y * cos
        )

@dataclass
class Ball:
    pos: Vector
    vel: Vector
    radius: float
    mass: float
    number: int
    color: str
    rotation: float = 0.0
    angular_vel: float = 0.0

class HeptagonBouncingBalls:
    def __init__(self, root):
        self.root = root
        self.root.title("弹跳小球")
        self.width, self.height = 800, 600
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()
        
        self.balls = self._create_balls()
        self.heptagon_center = Vector(self.width/2, self.height/2)
        self.heptagon_radius = min(self.width, self.height) * 0.4
        self.heptagon_angle = 0
        self.heptagon_angular_vel = 2 * math.pi / 5  # 360度每5秒
        
        self.gravity = Vector(0, 0.2)
        self.friction_coef = 0.99
        self.elasticity = 0.8
        self.ball_elasticity = 0.7
        
        self.animation_id = None
        self.start_animation()

    def _create_balls(self) -> List[Ball]:
        ball_radius = min(self.width, self.height) * 0.025
        ball_mass = 1.0
        colors = ["#f8b862", "#f6ad49", "#f39800", "#f08300", "#ec6d51", "#ee7948", "#ed6d3d", 
                 "#ec6800", "#ec6800", "#ee7800", "#eb6238", "#ea5506", "#ea5506", "#eb6101", 
                 "#e49e61", "#e45e32", "#e17b34", "#dd7a56", "#db8449", "#d66a35"]
        
        balls = []
        center = Vector(self.width/2, self.height/2)
        
        for i in range(20):
            angle = i * 2 * math.pi / 20
            # 随机初始速度
            speed = np.random.uniform(1.0, 3.0)
            vel_x = speed * math.cos(angle)
            vel_y = speed * math.sin(angle)
            
            ball = Ball(
                pos=center,
                vel=Vector(vel_x, vel_y),
                radius=ball_radius,
                mass=ball_mass,
                number=i+1,
                color=colors[i % len(colors)]
            )
            balls.append(ball)
        
        return balls

    def _get_heptagon_vertices(self) -> List[Vector]:
        vertices = []
        for i in range(7):
            angle = self.heptagon_angle + i * 2 * math.pi / 7
            x = self.heptagon_center.x + self.heptagon_radius * math.cos(angle)
            y = self.heptagon_center.y + self.heptagon_radius * math.sin(angle)
            vertices.append(Vector(x, y))
        return vertices

    def _check_heptagon_collision(self, ball: Ball) -> Optional[Tuple[Vector, Vector]]:
        vertices = self._get_heptagon_vertices()
        for i in range(7):
            p1 = vertices[i]
            p2 = vertices[(i+1)%7]
            
            line_vec = p2 - p1
            ball_vec = ball.pos - p1
            
            # 计算投影长度
            proj_length = ball_vec.dot(line_vec) / line_vec.dot(line_vec)
            proj_length = max(0, min(1, proj_length))  # 限制在[0,1]范围内
            
            # 计算最近点
            closest_point = p1 + line_vec * proj_length
            
            # 计算距离
            dist_vec = ball.pos - closest_point
            distance = dist_vec.magnitude()
            
            if distance < ball.radius:
                # 计算法向量（指向内部）
                normal = dist_vec.normalize()
                
                # 确保法向量指向七边形内部
                heptagon_center_to_point = closest_point - self.heptagon_center
                if normal.dot(heptagon_center_to_point) > 0:
                    normal = normal * -1
                
                return closest_point, normal
        
        return None

    def _handle_heptagon_collision(self, ball: Ball):
        collision = self._check_heptagon_collision(ball)
        if collision:
            closest_point, normal = collision
            
            # 计算切向量
            tangent = Vector(-normal.y, normal.x)
            
            # 分解速度为法向和切向分量
            normal_vel = normal.dot(ball.vel)
            tangent_vel = tangent.dot(ball.vel)
            
            # 应用弹性碰撞 - 法向速度反向并衰减
            new_normal_vel = -normal_vel * self.elasticity
            
            # 应用摩擦力 - 切向速度衰减
            new_tangent_vel = tangent_vel * self.friction_coef
            
            # 更新速度
            ball.vel = normal * new_normal_vel + tangent * new_tangent_vel
            
            # 防止球陷入边界
            overlap = ball.radius - (ball.pos - closest_point).magnitude()
            ball.pos = ball.pos + normal * (overlap + 0.1)
            
            # 计算旋转
            ball.angular_vel += normal_vel * 0.05

    def _check_ball_collision(self, ball1: Ball, ball2: Ball) -> bool:
        dist_vec = ball2.pos - ball1.pos
        distance = dist_vec.magnitude()
        return distance < ball1.radius + ball2.radius

    def _handle_ball_collision(self, ball1: Ball, ball2: Ball):
        if not self._check_ball_collision(ball1, ball2):
            return
        
        dist_vec = ball2.pos - ball1.pos
        distance = dist_vec.magnitude()
        
        # 计算法向量
        normal = dist_vec.normalize()
        
        # 计算切向量
        tangent = Vector(-normal.y, normal.x)
        
        # 计算法向速度分量
        v1n = normal.dot(ball1.vel)
        v1t = tangent.dot(ball1.vel)
        v2n = normal.dot(ball2.vel)
        v2t = tangent.dot(ball2.vel)
        
        # 计算碰撞后的法向速度 (一维弹性碰撞公式)
        m1, m2 = ball1.mass, ball2.mass
        v1n_after = (v1n * (m1 - m2) + 2 * m2 * v2n) / (m1 + m2)
        v2n_after = (v2n * (m2 - m1) + 2 * m1 * v1n) / (m1 + m2)
        
        # 应用弹性系数
        v1n_after *= self.ball_elasticity
        v2n_after *= self.ball_elasticity
        
        # 更新速度
        ball1.vel = normal * v1n_after + tangent * v1t
        ball2.vel = normal * v2n_after + tangent * v2t
        
        # 防止球重叠
        overlap = 0.5 * (ball1.radius + ball2.radius - distance + 0.1)
        ball1.pos = ball1.pos - normal * overlap
        ball2.pos = ball2.pos + normal * overlap
        
        # 更新旋转
        ball1.angular_vel += (v2n - v1n) * 0.03
        ball2.angular_vel += (v1n - v2n) * 0.03

    def update(self):
        self.canvas.delete("all")
        
        # 更新七边形角度
        self.heptagon_angle += self.heptagon_angular_vel * 0.016  # 基于帧时间
        
        # 绘制七边形
        vertices = self._get_heptagon_vertices()
        heptagon_coords = []
        for vertex in vertices:
            heptagon_coords.extend([vertex.x, vertex.y])
        self.canvas.create_polygon(heptagon_coords, fill="", outline="white", width=2)
        
        # 更新每个球
        for ball in self.balls:
            # 应用重力
            ball.vel = ball.vel + self.gravity
            
            # 应用空气阻力
            ball.vel = ball.vel * self.friction_coef
            
            # 更新位置
            ball.pos = ball.pos + ball.vel
            
            # 应用旋转摩擦
            ball.angular_vel *= 0.99
            ball.rotation += ball.angular_vel
            
            # 处理与七边形边界的碰撞
            self._handle_heptagon_collision(ball)
        
        # 检测并处理球之间的碰撞
        for i in range(len(self.balls)):
            for j in range(i+1, len(self.balls)):
                self._handle_ball_collision(self.balls[i], self.balls[j])
        
        # 绘制所有球
        for ball in self.balls:
            x, y = ball.pos.x, ball.pos.y
            r = ball.radius
            
            # 绘制球
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=ball.color, outline="white")
            
            # 绘制球上的数字，考虑旋转
            angle_rad = math.radians(ball.rotation)
            text_x = x + r/2 * math.cos(angle_rad)
            text_y = y + r/2 * math.sin(angle_rad)
            self.canvas.create_text(text_x, text_y, text=str(ball.number), fill="white", font=("Arial", 10, "bold"))
        
        # 继续动画循环
        self.animation_id = self.root.after(16, self.update)

    def start_animation(self):
        if self.animation_id is None:
            self.update()

    def stop_animation(self):
        if self.animation_id is not None:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None

if __name__ == "__main__":
    root = tk.Tk()
    app = HeptagonBouncingBalls(root)
    root.mainloop()    