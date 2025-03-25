import gym
from gym import spaces
from gym.utils import seeding
import sys
from time import sleep
import signal
# from gym.utils import rendering
import pygame
'''
    一个用于创建指定大小、类型和进入奖励的网格的类。

    属性:
        x: 网格的宽度
        y: 网格的高度
        grid_type: 网格的类型
        enter_reward: 进入该网格的奖励

    方法:
        __str__: 返回网格的字符串表示
'''

class Grid(object):
    def __init__(
            self,
            x: int = None,  # 网格宽度x
            y: int = None,  # 网格高度y
            grid_type: int = 0,  # 网格类型
            enter_reward: float = 0.0):  # 进入网格的奖励
        self.x = x
        self.y = y
        self.grid_type = grid_type
        self.enter_reward = enter_reward
        self.name = "X{0}-Y{1}".format(self.x, self.y)

    def __str__(self):
        return "Grid: {name:{3}, x:{0}, y:{1}, grid_type:{2}}".format(self.x, self.y, self.grid_type, self.name)


'''
    一个用于管理网格矩阵的类。

    属性:
        n_height: 网格的高度
        n_width: 网格的宽度
        default_reward: 默认奖励
        default_type: 默认类型
        grids: 存储网格对象的列表
        len: 网格的总数
'''

class GridMatrix(object):
    def __init__(
            self,
            n_width: int,  # 网格宽度
            n_height: int,  # 网格高度
            default_type: int = 0,  # 默认网格类型
            default_reward: float = 0.0,  # 默认奖励
    ):
        self.n_height = n_height
        self.n_width = n_width
        self.default_reward = default_reward
        self.default_type = default_type
        self.grids = None  # list(Grid) 
        self.len = n_width * n_height  
        self.reset()

    def reset(self):
        self.grids = []
        for x in range(self.n_height):
            for y in range(self.n_width):
                self.grids.append(Grid(x, y, self.default_type, self.default_reward))

    def get_grid(self, x, y=None):
        """
        获取指定位置的网格对象。
        参数: x和y可以是整数或元组。
        返回: 网格对象
        """
        xx, yy = None, None
        if isinstance(x, int):
            xx, yy = x, y
        elif isinstance(x, tuple):
            xx, yy = x[0], x[1]
        assert (0 <= xx < self.n_width and 0 <= yy < self.n_height)  # 确保坐标在范围内
        index = yy * self.n_width + xx  # 计算索引
        return self.grids[index]

    def set_reward(self, x, y, reward):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.enter_reward = reward
        else:
            raise ("grid doesn't exist")

    def set_type(self, x, y, grid_type):
        grid = self.get_grid(x, y)
        if grid is not None:
            grid.grid_type = grid_type
        else:
            raise ("grid doesn't exist")

    def get_reward(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.enter_reward

    def get_type(self, x, y):
        grid = self.get_grid(x, y)
        if grid is None:
            return None
        return grid.grid_type


'''
    一个基于网格世界的强化学习环境类，继承自gym.Env。

    属性:
        n_width: 网格宽度
        n_height: 网格高度
        u_size: 每个网格的大小
        default_reward: 默认奖励
        default_type: 默认类型
        screen_width: 窗口宽度
        screen_height: 窗口高度
        grids: 网格矩阵对象
        reward: 当前奖励
        action: 当前动作
        action_space: 动作空间
        observation_space: 状态空间
        state: 当前状态
        ends: 终止状态的位置
        start: 起始位置
        types: 特殊类型的网格位置和类型
        rewards: 特殊奖励的网格位置和奖励
        viewer: 可视化对象
'''

class GridWorldEnv(gym.Env):
    #自定义环境渲染环境模式申明在新版本0.26.2中必须
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 60}
    # def __init__(self, render_mode=None):
    #     #物理参数
    #     self.n_width = 5
    #     self.n_height = 5
    #     self.u_size = 40
    #     self.default_reward = 0
    #     self.default_type = 0
    #     self.screen_width = self.u_size * self.n_width
    #     self.screen_height = self.u_size * self.n_height
    #     self.grids = GridMatrix(n_width=self.n_width, n_height=self.n_height, default_reward=self.default_reward, default_type=self.default_type)
    #     self.reward = 0
    #     self.action = None
    #     self.action_space = spaces.Discrete(4)
    #     self.observation_space = spaces.Discrete(self.n_height * self.n_width)
    #     self.state = None
    #     self.ends = [(0, 0), (4, 3)]
    #     self.start = (0, 4)
    #     self.types = [(2, 2, 1)]
    #     self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]
    #     self.refresh_setting()
    #     self.viewer = None
    #     self.render_mode = render_mode
    #     self.seed()
    #     self.reset()


    def __init__(
            self,
            n_width: int = 5,  # 网格宽度
            n_height: int = 5,  # 网格高度
            u_size=40,  # 每个网格的大小
            default_reward: float = 0,
            default_type=0):# 渲染模式
        self.n_width = n_width
        self.n_height = n_height
        self.default_reward = default_reward
        self.default_type = default_type
        self.u_size = u_size
        self.screen = None  # Pygame窗口对象
        self.screen_width = u_size * n_width  # 窗口宽度
        self.screen_height = u_size * n_height  # 窗口高度
        self.agent_pos = [0, 0]  # 智能体的初始位置

        self.grids = GridMatrix(n_width=self.n_width,
                                n_height=self.n_height,
                                default_reward=self.default_reward,
                                default_type=self.default_type)
        self.reward = 0  # 当前奖励
        self.action = None  # 当前动作

        # 动作空间：0左，1右，2上，3下
        self.action_space = spaces.Discrete(4)
        # 状态空间：网格的总数
        self.observation_space = spaces.Discrete(self.n_height * self.n_width)

        self.state = None  # 当前状态
        self.ends = [(0, 0), (4, 3)]  # 终止状态位置
        self.start = (0, 4)  # 起始位置
        self.types = [(2, 2, 1)]  # 特殊类型网格
        self.rewards = [(0, 0, 1), (4, 3, 5), (2, 2, -10)]  # 特殊奖励网格
        self.refresh_setting()
        self.viewer = None  # 可视化对象
        self.seed()  # 设置随机种子
        self.reset()

    def seed(self, seed=None):
        # 设置随机种子
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # 执行动作并返回新的状态、奖励、是否终止和额外信息
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.action = action  # 保存当前动作
        old_x, old_y = self._state_to_xy(self.state)
        new_x, new_y = old_x, old_y

        # 根据动作更新坐标
        if action == 0:
            new_x -= 1  # 左
        elif action == 1:
            new_x += 1  # 右
        elif action == 2:
            new_y += 1  # 上
        elif action == 3:
            new_y -= 1  # 下

        # 边界检查
        if new_x < 0:
            new_x = 0
        if new_x >= self.n_width:
            new_x = self.n_width - 1
        if new_y < 0:
            new_y = 0
        if new_y >= self.n_height:
            new_y = self.n_height - 1

        # 墙壁检查
        if self.grids.get_type(new_x, new_y) == 1:
            new_x, new_y = old_x, old_y

        self.reward = self.grids.get_reward(new_x, new_y)
        done = self._is_end_state(new_x, new_y)
        self.state = self._xy_to_state(new_x, new_y)
        info = {"x": new_x, "y": new_y, "grids": self.grids}
        truncated = False  # 新版本 Gym 需要返回截断信息
        return self.state, self.reward, done, truncated, info

    # 将状态转换为坐标
    def _state_to_xy(self, s):
        x = s % self.n_width
        y = int((s - x) / self.n_width)
        return x, y

    # 将坐标转换为状态
    def _xy_to_state(self, x, y=None):
        if isinstance(x, int):
            assert (isinstance(y, int)), "incomplete Position info"
            return x + self.n_width * y
        elif isinstance(x, tuple):
            return x[0] + self.n_width * x[1]
        return -1  # 错误情况

    # 刷新设置，设置特殊奖励和类型
    def refresh_setting(self):
        for x, y, r in self.rewards:
            self.grids.set_reward(x, y, r)
        for x, y, t in self.types:
            self.grids.set_type(x, y, t)

    # 重置环境到初始状态
    def reset(self):
        self.state = self._xy_to_state(self.start)
        return self.state

    # 检查是否为终止状态
    def _is_end_state(self, x, y=None):
        if y is not None:
            xx, yy = x, y
        elif isinstance(x, int):
            xx, yy = self._state_to_xy(x)
        else:
            assert (isinstance(x, tuple)), ""
            xx, yy = x[0], x[1]
        for end in self.ends:
            if xx == end[0] and yy == end[1]:
                return True
        return False

    # 渲染环境
    # def render(self):
    #     if self.render_mode == "human":
    #         self._render_gui()
    #     if self.render_mode == "rgb_array":
    #         return self.get_rgb_array
    # def _render_gui(self):
    #     if self.screen is None:
    #         pygame.init()
    #         self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
    #         pygame.display.set_caption("Grid Environment")   
    #     self.screen.fill((255, 255, 255))  # 填充背景为白色

    def render(self, mode='human'):
        """渲染环境"""
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported for rendering")

        # 初始化Pygame窗口
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Grid Environment")

        # 绘制网格
        self.screen.fill((255, 255, 255))  # 填充背景为白色
        for x in range(self.n_width):
            for y in range(self.n_height):
                u_size = self.u_size
                m = 2  # 边距
                rect = pygame.Rect(x * u_size + m, y * u_size + m, u_size - 2 * m, u_size - 2 * m)
                
                r = self.grids.get_reward(x, y) / 10
                if r < 0:
                    color = (0.9 - r, 0.9 + r, 0.9 + r)
                elif r > 0:
                    color = (0.3, 0.5 + r, 0.3)
                else:
                    color = (0.9, 0.9, 0.9)
                
                if self.grids.get_type(x, y) == 1:  # 墙壁
                    color = (0.3, 0.3, 0.3)
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # 绘制边框

                # 绘制特殊格子的边框
                if self._is_end_state(x, y):
                    pygame.draw.rect(self.screen, (0.9, 0.9, 0), rect, 3)
                if self.start[0] == x and self.start[1] == y:
                    pygame.draw.rect(self.screen, (0.5, 0.5, 0.8), rect, 3)

        # 绘制智能体
        agent_rect = pygame.Rect(
            self.agent_pos[1] * self.u_size + 10,
            self.agent_pos[0] * self.u_size + 10,
            self.u_size - 20,
            self.u_size - 20
        )
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)  # 智能体为蓝色

        # 更新显示
        pygame.display.flip()

        # 处理关闭事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()



def CtrlCHandler(signum, frame):
    env.close()
    print("User interrupt!")
    sys.exit(0)
from gym.envs.registration import register

# register(
#     id="GraidWorldEnv-v0",
#     entry_point="graid_world_env:GraidWorldEnv",
#     # max_episode_steps=200,
# )

if __name__ == "__main__":
    env = GridWorldEnv()  # 创建环境
    env.refresh_setting()  # 刷新设置
    # env = gym.make("GraidWorldEnv-v0")
    env.seed(1)  # 设置随机种子
    print("env.action_space:", env.action_space)
    print("env.observation_space:", env.observation_space)
    signal.signal(signal.SIGINT, CtrlCHandler)  # 设置中断处理
    episode_num = 100
    for e in range(episode_num):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # 随机选择动作
            env.render()  # 渲染环境
            sleep(0.5)
            next_state, reward, done, truncated, info = env.step(action) 
    env.close()
    sys.exit(0)