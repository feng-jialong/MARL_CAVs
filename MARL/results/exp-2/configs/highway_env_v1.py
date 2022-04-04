import numpy as np
from gym.envs.registration import register
from typing import Tuple
import random

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle

class HighwayEnv(AbstractEnv):
    """
    A multi-agent highway environment.
    """
    n_a = 5
    n_s = 60  # max_obs(10)*特征维度(6),hardcode max_obs
        
    def __init__(self):
        self.scene_type = 0  # 初始场景
        self.use_centralized_critic = False
        super().__init__()

    @classmethod  # 类方法，无需实例化类即可调用，主要用于构造函数__init__中
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
                },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "screen_width": 1500,  # 动画的宽度，单位为像素
            "screen_height": 120,   # 动画的高度，同上
            "centering_position": [0.5, 0.5],  # 参见road和commom中的graphics.py
            "scaling": 3,   # 世界坐标[m]与像素坐标[px]之间的缩放比例，每米多少像素，现在这个参数是必须的了
            "simulation_frequency": 15,  # 每秒仿真步数[Hz]
            "duration": 20,  # 仿真秒
            "policy_frequency": 5,  # 每秒控制步数[Hz]
            "lanes_number": 3,
            "CAV_number": 3,
            "HDV_number": 0
        })
        return config

    # 所有AV的平均reward，即global reward
    def _reward(self, action: int) -> float:
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        close_vehicles = self.road.close_vehicles_to(vehicle,
                                                     100,  # hardcode
                                                     count=-1,  # 考察观测范围内的所有车
                                                     see_behind=True)
        close_vehicles = [v for v in close_vehicles if v not in self.controlled_vehicles]
        # 0: flocking, 1:overtaking
        mode = 1 if len(close_vehicles) else 0
        vehicle.mode = mode
        
        # region 危险决策惩罚
        vehicle.safety = (-1 if vehicle.is_dangerous else 0)
        safety_reward = self.config['SAFE_DECISION_REWARD'] * vehicle.safety

        # endregion
        
        # region 车速奖励
        if mode == 1:
            scaled_speed = utils.lmap(vehicle.speed,
                            [vehicle.SPEED_MIN,vehicle.SPEED_MAX], 
                            [0, 1])
            speed_reward = np.clip(scaled_speed, 0, 1)  # 线性
            # speed_reward = np.clip(2**scaled_speed-1,0,1)  # 下凸
        else:
            scaled_speed = utils.lmap(vehicle.speed,
                            [vehicle.SPEED_MIN,self.config['DESIRE_SPEED']], 
                            [-1, 0])
            # speed_reward = np.clip(2**(scaled_speed+1)-2,0,1)
            speed_reward = np.clip(scaled_speed,-1, 0)  # 线性
        # endregion
        
        # region 聚群奖励
        vehicle.flocking_distance = np.linalg.norm(vehicle.position-self.flocking_center)
        if mode == 1:
            flocking_reward = 0
        else:
            flocking_reward = self.config['FLOCKING_REWARD'] * 1/vehicle.flocking_distance
        # endregion
        
        # region 冲突惩罚
        '''
        v_front,_ = self.road.neighbour_vehicles(vehicle,vehicle.lane_index)
        if vehicle.crashed == True:
            if (v_front is not None) and vehicle._is_colliding(v_front):
                collision_reward = -1
            else:
                collision_reward = -0.5  # 被追尾车辆惩罚应该小一些
        else:
            collision_reward = 0
        '''
        collision_reward = -1 if vehicle.crashed else 0
        # endregion

        # region 舒适度奖励
        vehicle.comfort = (0 if action==1 else -1)
        comfort_reward = self.config["COMFORT_REWARD"] * vehicle.comfort
        # endregion
        
        reward = self.config["COLLISION_REWARD"] * collision_reward \
                 + self.config["HIGH_SPEED_REWARD"] * speed_reward \
                 + flocking_reward \
                 + comfort_reward \
                 + safety_reward
        return reward
    
    # 计算regional reward，即local reward
    def _regional_reward(self):
        # 相同与相邻车道neighbor
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = [vehicle]
            neighbor_lanes = self.road.network.side_lanes(vehicle.lane_index) + [vehicle.lane_index]

            for lane in neighbor_lanes:
                v_f, v_r = self.road.neighbour_vehicles(vehicle,lane)
                if type(v_f) is MDPVehicle and v_f is not None:
                    neighbor_vehicle.append(v_f)
                if type(v_r) is MDPVehicle and v_r is not None:
                    neighbor_vehicle.append(v_r)
            
            # 邻居的奖励加权    
            # regional_reward = sum((0.3 if v is vehicle else 1)*v.local_reward for v in neighbor_vehicle) 
            
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward/sum(1 for _ in filter(None.__ne__,neighbor_vehicle))
        
        # 距离较近neighbor
        pass
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []

        obs, reward, done, info = super().step(action)  # info是一个step的info
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        info['flocking_distance'] = 0
        info['comfort'] = 0
        info['safety'] = 0
        cnt = 0
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed, v.heading])  #TODO :记录车头朝向
            if v.mode == 0:
                self.flocking_center += v.position
                cnt += 1
            # 奖励
            v.local_reward = self._agent_reward(action,v)
            # 记录每一step的各项评价指标
            # 聚群
            info['flocking_distance'] += v.flocking_distance
            # 舒适度
            info['comfort'] += (v.comfort)          
            # 安全性
            info['safety'] += (v.safety)
        info["agents_info"] = agent_info
        
        # 更新flocking_center，用于计算集群奖励flockin_reward
        self.flocking_center /= max(cnt,1)
        
        info['flocking_distance'] /= self.config['CAV_number']
        info['comfort'] /= self.config['CAV_number']
        info['safety'] /= self.config['CAV_number']
        
        info['comfort'] += 1
        info['safety'] += 1
        
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # self._regional_reward()
        # info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        return obs, reward, done, info  # obs传入算法用于训练

    def _is_terminal(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config['CAV_number'],self.config['HDV_number'])
        self.T = int(self.config["duration"] * self.config["policy_frequency"])
        
        # 初始化flocking_center, 车群中心
        self.flocking_center = np.array([vehicle.position for vehicle in self.controlled_vehicles if vehicle.mode==0]).mean(axis=0)

    def _make_road(self,) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        n_lanes = self.config["lanes_number"]

        net = RoadNetwork.straight_road_network(lanes=n_lanes,
                                                start=0,   
                                                length=200+300+300,  # 两百米用于CAV，两百米用于HDV
                                                speed_limit=30,
                                                nodes_str=("a","b"))
        road = Road(network=net, 
                    np_random=self.np_random, 
                    record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, num_CAV=5, num_HDV=0) -> None:
        road = self.road
        n_lanes = self.config["lanes_number"]
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        CAV_initial_speed = 25    # CAV的初始速度，90 m/s, hardcode
        HDV_initial_speed = 15    # HDV的初始速度，54 m/s, hardcode
        HDV_start = 150
        
        '''
        # 生成九辆车的对齐的CAV车群
        CAV_lon_dist = 40  # 对齐CAV车群中车辆的纵向距离
        """Spawn points for CAV"""
        spawn_point_c = [CAV_lon_dist*i for i in [1,2,3]]

        """spawn the CAV on the straight road"""
        for lane in range(n_lanes):
            for i in range(2):
                ego_vehicle = self.action_type.vehicle_class(road,
                                                             road.network.get_lane(("a","b",lane)).position(spawn_point_c[i],0),
                                                             speed=CAV_initial_speed)
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)
        '''
        # 随机生成CAV车队，既有紧凑也有松散
        """Spawn points for CAV"""
        spawn_pt = [5,20,35,50,65]
        spawn_point_c = [random.sample(spawn_pt,len(spawn_pt)) for _ in range(n_lanes)]
        
        for _ in range(num_CAV):
            lane = np.random.choice(n_lanes)
            ego_vehicle = self.action_type.vehicle_class(road,
                                                    road.network.get_lane(("a","b",lane)).position(spawn_point_c[lane].pop(0),0),
                                                    speed=CAV_initial_speed)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        
        # 生成对应于训练场景的HDV
        """Spawn points for HDV"""
        if self.scene_type == 1:  # recognize different scene type
            spawn_point_h = [[0],[],[]]
        elif self.scene_type == 2:
            spawn_point_h = [[],[],[0]]
        elif self.scene_type == 3:
            spawn_point_h = [[],[0],[]]
        elif self.scene_type == 4:
            spawn_point_h = [[50],[0],[]]
        elif self.scene_type == 5:
            spawn_point_h = [[],[0],[50]]
        elif self.scene_type == 6:
            spawn_point_h = [[0],[],[0]]
        elif self.scene_type == 7:
            spawn_point_h = [[],[0],[0]]
        elif self.scene_type == 8:
            spawn_point_h = [[0],[0],[]]
        elif self.scene_type == 9:
            spawn_point_h = [[0],[50],[0]]
        elif self.scene_type == 10:
            spawn_point_h = [[50],[0],[0]]
        elif self.scene_type == 11:
            spawn_point_h = [[0],[0],[50]]
        elif self.scene_type == 0:
            spawn_point_h = [[],[],[]]
            
        """spawn the HDV on the straight road"""
        for i in range(n_lanes):
            for loc in spawn_point_h[i]:
                road.vehicles.append(
                    other_vehicles_type(road,
                                        road.network.get_lane(("a","b",i)).position(loc+HDV_start,0),
                                        speed=HDV_initial_speed,
                                        target_speed=HDV_initial_speed))  # 在此设置HDV期望速度

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class HighwayEnvMARL(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                "type": "Kinematics"
            }}
        })
        return config


register(
    id='highway-v1',
    entry_point='highway_env.envs:HighwayEnv',
)

register(
    id='highway-multi-agent-v0',
    entry_point='highway_env.envs:HighwayEnvMARL',
)
