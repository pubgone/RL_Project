# import gym
# from gym import spaces
# from gym.utils import seeding
# import sys
# from time import sleep
# import signal
# # from gym.utils import rendering
# import pygame
# '''
#     一个用于创建指定大小、类型和进入奖励的网格的类。

#     属性:
#         x: 网格的宽度
#         y: 网格的高度
#         grid_type: 网格的类型
#         enter_reward: 进入该网格的奖励

#     方法:
#         __str__: 返回网格的字符串表示
# '''

# class Grid(object):
#     def __init__(
#             self,
#             x: int = None,  # 网格宽度x
#             y: int = None,  # 网格高度y
#             grid_type: int = 0,  # 网格类型
#             enter_reward: float = 0.0):  # 进入网格的奖励
#         self.x = x
#         self.y = y
#         self.grid_type = grid_type
#         self.enter_reward = enter_reward
#         self.name = "X{0}-Y{1}".format(self.x, self.y)

#     def __str__(self):
#         return "Grid: {name:{3}, x:{0}, y:{1}, grid_type:{2}}".format(self.x, self.y, self.grid_type, self.name)


# '''
#     一个用于管理网格矩阵的类。

#     属性:
#         n_height: 网格的高度
#         n_width: 网格的宽度
#         default_reward: 默认奖励
#         default_type: 默认类型
#         grids: 存储网格对象的列表
#         len: 网格的总数
# '''

# class GridMatrix(object):
#     def __init__(
#             self,
#             n_width: int,  # 网格宽度
#             n_height: int,  # 网格高度
#             default_type: int = 0,  # 默认网格类型
#             default_reward: float = 0.0,  # 默认奖励
#     ):
#         self.n_height = n_height
#         self.n_width = n_width
#         self.default_reward = default_reward
#         self.default_type = default_type
#         self.grids = None  # list(Grid) 
#         self.len = n_width * n_height  
#         self.reset()

#     def reset(self):
#         self.grids = []
#         for x in range(self.n_height):
#             for y in range(self.n_width):
#                 self.grids.append(Grid(x, y, self.default_type, self.default_reward))

#     def get_grid(self, x, y=None):
#         """
#         获取指定位置的网格对象。
#         参数: x和y可以是整数或元组。
#         返回: 网格对象
#         """
#         xx, yy = None, None
#         if isinstance(x, int):
#             xx, yy = x, y
#         elif isinstance(x, tuple):
#             xx, yy = x[0], x[1]
#         assert (0 <= xx < self.n_width and 0 <= yy < self.n_height)  # 确保坐标在范围内
#         index = yy * self.n_width + xx  # 计算索引
#         return self.grids[index]

#     def set_reward(self, x, y, reward):
#         grid = self.get_grid(x, y)
#         if grid is not None:
#             grid.enter_reward = reward
#         else:
#             raise ("grid doesn't exist")

#     def set_type(self, x, y, grid_type):
#         grid = self.get_grid(x, y)
#         if grid is not None:
#             grid.grid_type = grid_type
#         else:
#             raise ("grid doesn't exist")

#     def get_reward(self, x, y):
#         grid = self.get_grid(x, y)
#         if grid is None:
#             return None
#         return grid.enter_reward

#     def get_type(self, x, y):
#         grid = self.get_grid(x, y)
#         if grid is None:
#             return None
#         return grid.grid_type


# '''
#     一个基于网格世界的强化学习环境类，继承自gym.Env。

#     属性:
#         n_width: 网格宽度
#         n_height: 网格高度
#         u_size: 每个网格的大小
#         default_reward: 默认奖励
#         default_type: 默认类型
#         screen_width: 窗口宽度
#         screen_height: 窗口高度
#         grids: 网格矩阵对象
#         reward: 当前奖励
#         action: 当前动作
#         action_space: 动作空间
#         observation_space: 状态空间
#         state: 当前状态
#         ends: 终止状态的位置
#         start: 起始位置
#         types: 特殊类型的网格位置和类型
#         rewards: 特殊奖励的网格位置和奖励
#         viewer: 可视化对象
# '''

# class GraidWorldEnv(gym.Env):
#     #自定义环境渲染环境模式申明在新版本0.26.2中必须
#     metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}
#     def __init__(self, render_mode=None):
#         #物理参数
#         self.n_width = 5
#         self.n_height = 5
#         self.u_size = 40
#         self.default_reward = 0
#         self.default_type = 0
#         self.screen_width = self.u_size * self.n_width
#         self.screen_height = self.u_size * self.n_height
#         self.grids = GridMatrix(n_width=self.n_width, n_height=self.n_height, default_reward=self.default_reward, default_type=self.default_type)
#         self.reward = 0
#         self.action = None
#         self.action_space = spaces.Discrete(4)
#         self.observation_space = spaces.Discrete(self.n_height * self.n_width)
#         self.agent_pos = [0, 0]  # 智能体的初始位置
#         self.state = None
#         self.ends = [(0, 0), (4, 3)]
#         self.start = (0, 4)
#         self.types = [(2, 2, 1)]
#         self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]
#         self.refresh_setting()
#         self.viewer = None
#         self.render_mode = render_mode
#         self.screen = None
#         self.seed()
#         self.reset()


#     # def __init__(
#     #         self,
#     #         n_width: int = 5,  # 网格宽度
#     #         n_height: int = 5,  # 网格高度
#     #         u_size=40,  # 每个网格的大小
#     #         default_reward: float = 0,
#     #         default_type=0):# 渲染模式
#     #     self.n_width = n_width
#     #     self.n_height = n_height
#     #     self.default_reward = default_reward
#     #     self.default_type = default_type
#     #     self.u_size = u_size
#     #     self.screen = None  # Pygame窗口对象
#     #     self.screen_width = u_size * n_width  # 窗口宽度
#     #     self.screen_height = u_size * n_height  # 窗口高度
#     #     self.agent_pos = [0, 0]  # 智能体的初始位置

#     #     self.grids = GridMatrix(n_width=self.n_width,
#     #                             n_height=self.n_height,
#     #                             default_reward=self.default_reward,
#     #                             default_type=self.default_type)
#     #     self.reward = 0  # 当前奖励
#     #     self.action = None  # 当前动作

#     #     # 动作空间：0左，1右，2上，3下
#     #     self.action_space = spaces.Discrete(4)
#     #     # 状态空间：网格的总数
#     #     self.observation_space = spaces.Discrete(self.n_height * self.n_width)

#     #     self.state = None  # 当前状态
#     #     self.ends = [(0, 0), (4, 3)]  # 终止状态位置
#     #     self.start = (0, 4)  # 起始位置
#     #     self.types = [(2, 2, 1)]  # 特殊类型网格
#     #     self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]  # 特殊奖励网格
#     #     self.refresh_setting()
#     #     self.viewer = None  # 可视化对象
#     #     self.seed()  # 设置随机种子
#     #     self.reset()

#     def seed(self, seed=None):
#         # 设置随机种子
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def step(self, action):
#         # 执行动作并返回新的状态、奖励、是否终止和额外信息
#         assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
#         self.action = action  # 保存当前动作
#         old_x, old_y = self._state_to_xy(self.state)
#         new_x, new_y = old_x, old_y

#         # 根据动作更新坐标
#         if action == 0:
#             new_x -= 1  # 左
#         elif action == 1:
#             new_x += 1  # 右
#         elif action == 2:
#             new_y += 1  # 上
#         elif action == 3:
#             new_y -= 1  # 下

#         # 边界检查
#         if new_x < 0:
#             new_x = 0
#         if new_x >= self.n_width:
#             new_x = self.n_width - 1
#         if new_y < 0:
#             new_y = 0
#         if new_y >= self.n_height:
#             new_y = self.n_height - 1

#         # 墙壁检查
#         if self.grids.get_type(new_x, new_y) == 1:
#             new_x, new_y = old_x, old_y

#         self.reward = self.grids.get_reward(new_x, new_y)
#         done = self._is_end_state(new_x, new_y)
#         self.state = self._xy_to_state(new_x, new_y)
#         info = {"x": new_x, "y": new_y, "grids": self.grids}
#         truncated = False  # 新版本 Gym 需要返回截断信息
#         return self.state, self.reward, done, truncated, info

#     # 将状态转换为坐标
#     def _state_to_xy(self, s):
#         x = s % self.n_width
#         y = int((s - x) / self.n_width)
#         return x, y

#     # 将坐标转换为状态
#     def _xy_to_state(self, x, y=None):
#         if isinstance(x, int):
#             assert (isinstance(y, int)), "incomplete Position info"
#             return x + self.n_width * y
#         elif isinstance(x, tuple):
#             return x[0] + self.n_width * x[1]
#         return -1  # 错误情况

#     # 刷新设置，设置特殊奖励和类型
#     def refresh_setting(self):
#         for x, y, r in self.rewards:
#             self.grids.set_reward(x, y, r)
#         for x, y, t in self.types:
#             self.grids.set_type(x, y, t)

#     # 重置环境到初始状态
#     def reset(self):
#         self.state = self._xy_to_state(self.start)
#         return self.state

#     # 检查是否为终止状态
#     def _is_end_state(self, x, y=None):
#         if y is not None:
#             xx, yy = x, y
#         elif isinstance(x, int):
#             xx, yy = self._state_to_xy(x)
#         else:
#             assert (isinstance(x, tuple)), ""
#             xx, yy = x[0], x[1]
#         for end in self.ends:
#             if xx == end[0] and yy == end[1]:
#                 return True
#         return False

#     # 渲染环境
#     def render(self):
#         if self.render_mode == "human":
#             self._render_gui()
#         if self.render_mode == "rgb_array":
#             return self.get_rgb_array
#     def _render_gui(self):
#         if self.screen is None:
#             pygame.init()
#             # running = True
#             self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#             pygame.display.set_caption("Grid Environment") 
#             self.screen.fill((255, 255, 255))  # 填充背景为白色
#             for x in range(self.n_width):
#                 for y in range(self.n_height):
#                     u_size = self.u_size
#                     m = 2  # 边距
#                     rect = pygame.Rect(x * u_size + m, y * u_size + m, u_size - 2 * m, u_size - 2 * m)#绘制网格矩形
#                     r = self.grids.get_reward(x, y) / 10
#                     if r < 0:
#                         # color = (0.9 - r, 0.9 + r, 0.9 + r)
#                         color = (0, 0, 100)
#                     elif r > 0:
#                         color = (0.3, 0.5 + r, 0.3)
#                     else:
#                         color = (0.9, 0.9, 0.9)
#                     if self.grids.get_type(x, y) == 1:
#                         color = (0.3, 0.3, 0.3)
#                     pygame.draw.rect(self.screen, color, rect)
#                     pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # 绘制边框
#                     if self._is_end_state(x, y):
#                         pygame.draw.rect(self.screen, (0.9, 0.9, 0), rect, 3)
#                     if self.start[0] == x and self.start[1] == y:
#                         pygame.draw.rect(self.screen, (0.5, 0.5, 0.8), rect, 3)
#             agent_rect = pygame.Rect(
#                 self.agent_pos[1] * self.u_size + 10,
#                 self.agent_pos[0] * self.u_size + 10,
#                 self.u_size - 20,
#                 self.u_size - 20
#             )
#             pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)  # 智能体为蓝色
#             pygame.display.flip()
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     # sys.exit()
#     def get_rgb_array(self):
#         self._render_gui()
#         image_data = pygame.surfarray.array3d(pygame.display.get_surface())
#         return image_data
    
# from gym.envs.registration import register

# register(
#     id="GraidWorldEnv-v1",
#     entry_point="graidworld_env:GraidWorldEnv",
#     max_episode_steps=200,
# )

# if __name__ == "__main__":
#     # env.refresh_setting()  # 刷新设置
#     env = gym.make("GraidWorldEnv-v1", render_mode="human")
#     env.seed(1)  # 设置随机种子
#     print("env.action_space:", env.action_space)
#     print("env.observation_space:", env.observation_space)
#     # signal.signal(signal.SIGINT, CtrlCHandler)  # 设置中断处理
#     episode_num = 100
#     for _ in range(episode_num):
#         env.reset()
#         done = False
#         while not done:
#             action = env.action_space.sample()  # 随机选择动作
#             env.render()  # 渲染环境
#             sleep(0.5)
#             next_state, reward, done, truncated, info = env.step(action) 
#             pygame.update()
#     env.close()
#     sys.exit(0)
import gym
from gym import spaces
from gym.utils import seeding
import sys
import pygame
import numpy as np

# 网格类定义
class Grid(object):
    def __init__(self, x=0, y=0, grid_type=0, enter_reward=0.0):
        self.x = x
        self.y = y
        self.grid_type = grid_type
        self.enter_reward = enter_reward
        self.name = f"X{x}-Y{y}"

    def __str__(self):
        return f"Grid {self.name}: Type={self.grid_type}, Reward={self.enter_reward}"

# 网格矩阵管理类
class GridMatrix(object):
    def __init__(self, n_width, n_height, default_type=0, default_reward=0.0):
        self.n_width = n_width
        self.n_height = n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self.grids = []
        self.reset()

    def reset(self):
        self.grids = [
            Grid(x, y, self.default_type, self.default_reward)
            for y in range(self.n_height)
            for x in range(self.n_width)
        ]

    def get_index(self, x, y):
        return y * self.n_width + x

    def get_grid(self, x, y):
        if 0 <= x < self.n_width and 0 <= y < self.n_height:
            return self.grids[self.get_index(x, y)]
        return None

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid:
            grid.enter_reward = reward

    def set_type(self, x, y, grid_type):
        grid = self.get_grid(x, y)
        if grid:
            grid.grid_type = grid_type

# 自定义环境类
class GraidWorldEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 10
    }

    def __init__(self, render_mode=None):
        # 环境参数
        self.n_width = 5
        self.n_height = 5
        self.u_size = 40
        self.window_size = (
            self.n_width * self.u_size,
            self.n_height * self.u_size
        )
        
        # 初始化动作和观察空间
        self.action_space = spaces.Discrete(4)  # 0:左 1:右 2:上 3:下
        self.observation_space = spaces.Discrete(self.n_width * self.n_height)
        
        # 环境状态设置
        self.start = (0, 4)
        self.ends = [(0, 0), (4, 3)]
        self.types = [(2, 2, 1)]  # 障碍物位置
        self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]
        self.agent_pos = list(self.start)
        
        # 渲染相关
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # 初始化网格系统
        self.grids = GridMatrix(
            n_width=self.n_width,
            n_height=self.n_height,
            default_type=0,
            default_reward=0.0
        )
        self._refresh_settings()
        self.reset()

    def _refresh_settings(self):
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)

    def _xy_to_state(self, x, y):
        return y * self.n_width + x

    def _state_to_xy(self, state):
        x = state % self.n_width
        y = state // self.n_width
        return x, y

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start)
        return self._xy_to_state(*self.agent_pos), {}

    def step(self, action):
        x, y = self.agent_pos
        new_x, new_y = x, y

        # 执行动作
        if action == 0:   # 左
            new_x = max(x - 1, 0)
        elif action == 1: # 右
            new_x = min(x + 1, self.n_width - 1)
        elif action == 2: # 上
            new_y = max(y - 1, 0)
        elif action == 3: # 下
            new_y = min(y + 1, self.n_height - 1)

        # 碰撞检测
        grid = self.grids.get_grid(new_x, new_y)
        if grid and grid.grid_type == 1:  # 障碍物
            new_x, new_y = x, y

        # 更新状态
        self.agent_pos = [new_x, new_y]
        state = self._xy_to_state(new_x, new_y)
        
        # 计算奖励和终止条件
        reward = grid.enter_reward if grid else 0.0
        terminated = any([(new_x, new_y) == end for end in self.ends])
        truncated = False
        
        return state, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("GridWorld Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        # 绘制网格
        for x in range(self.n_width):
            for y in range(self.n_height):
                rect = pygame.Rect(
                    x * self.u_size,
                    y * self.u_size,
                    self.u_size, self.u_size
                )
                grid = self.grids.get_grid(x, y)
                
                # 设置颜色
                if (x, y) in self.ends:
                    color = (255, 215, 0)  # 金色终点
                elif grid.grid_type == 1:
                    color = (169, 169, 169)  # 灰色障碍物
                else:
                    r = min(max(grid.enter_reward / 10, 0), 1)
                    if grid.enter_reward > 0:
                        color = (34, 139 + int(86*r), 34)  # 渐变绿色
                    elif grid.enter_reward < 0:
                        color = (139 + int(116*r), 0, 0)   # 渐变红色
                    else:
                        color = (245, 245, 245)  # 白色

                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        # 绘制智能体
        agent_rect = pygame.Rect(
            self.agent_pos[0] * self.u_size + 5,
            self.agent_pos[1] * self.u_size + 5,
            self.u_size - 10,
            self.u_size - 10
        )
        pygame.draw.rect(canvas, (30, 144, 255), agent_rect)  # 蓝色智能体

        # 更新显示
        self.screen.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# 环境注册
from gym.envs.registration import register
register(
    id="GraidWorldEnv-v2",
    entry_point=__name__ + ":GraidWorldEnv",
    max_episode_steps=200,
)

# 测试运行
if __name__ == "__main__":
    env = GraidWorldEnv(render_mode="human")
    observation, _ = env.reset()
    
    for _ in range(5):  # 运行5个回合
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                done = True
        env.reset()
    
    env.close()