import copy
import os
from typing import List, Tuple, Optional, Callable
import gym
import random
from gym import Wrapper
import numpy as np
from queue import PriorityQueue

from highway_env import utils
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.idm_controller import idm_controller, generate_actions
from highway_env.envs.common.mdp_controller import mdp_controller
from highway_env.road.objects import Obstacle, Landmark

Observation = np.ndarray
DEFAULT_WIDTH: float = 4


class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    observation_type: ObservationType
    action_type: ActionType
    automatic_rendering_callback: Optional[Callable]
    metadata = {'render.modes': ['human', 'rgb_array']}

    PERCEPTION_DISTANCE = 6.0 * MDPVehicle.SPEED_MAX
    """The maximum distance of any vehicle present in the observation [m]"""

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # Seeding
        self.np_random = None
        self.seed = self.config["seed"]

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.ACTIONS_ALL = {'LANE_LEFT': 0,
                            'IDLE': 1,
                            'LANE_RIGHT': 2,
                            'FASTER': 3,
                            'SLOWER': 4}

        self.reset()

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "TimeToCollision"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",   # 设置HDV的跟车模型
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,   # 是否在动画中显示车辆轨迹（前后数个仿真步时车辆的位置）
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,    # 是否开启手动控制(?)
            "real_time_rendering": False,
            "seed": 0,
            "action_masking": True
        }

    def seed(self, seeding: int = None) -> List[int]:
        seed = np.random.seed(self.seed)
        return [seed]

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self,self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.

        If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self, is_training=True, testing_seeds=0, num_CAV=0) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        if is_training:
            np.random.seed(self.seed)
            random.seed(self.seed)
        else:
            np.random.seed(testing_seeds)
            random.seed(testing_seeds)
            
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.seed += 1  # 改变随机数种子
        self.done = False
        self.vehicle_speed = []
        self.vehicle_pos = []
        self._reset()
        self.define_spaces()
        for i, v in enumerate(self.road.vehicles):
            v.id = i
        if self.use_centralized_critic:
            obs = [self.observation_type.observe(),self.observation_type.observe(is_global=True)]
        else:
            obs = self.observation_type.observe()
        
        # region get action masks
        if self.config["action_masking"]:
            available_actions = [[0] * self.n_a] * len(self.controlled_vehicles)
            for i in range(len(self.controlled_vehicles)):
                available_action = self._get_available_actions(self.controlled_vehicles[i], self)
                for a in available_action:
                    available_actions[i][a] = 1
        else:
            available_actions = [[1] * self.n_a] * len(self.controlled_vehicles)
        # endregion
            
        return obs, np.array(available_actions)

    def _reset(self, num_CAV=1) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def _get_available_actions(self, vehicle, env_copy):
        """
        Get the list of currently available actions.
        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.
        :return: the list of available actions
        """
        actions = [env_copy.ACTIONS_ALL['IDLE']]
        for l_index in env_copy.road.network.side_lanes(vehicle.lane_index):
            if l_index[2] < vehicle.lane_index[2] \
                    and env_copy.road.network.get_lane(l_index).is_reachable_from(vehicle.position):
                actions.append(env_copy.ACTIONS_ALL['LANE_LEFT'])
            if l_index[2] > vehicle.lane_index[2] \
                    and env_copy.road.network.get_lane(l_index).is_reachable_from(vehicle.position):
                actions.append(env_copy.ACTIONS_ALL['LANE_RIGHT'])
        if vehicle.speed_index < vehicle.SPEED_COUNT - 1:
            actions.append(env_copy.ACTIONS_ALL['FASTER'])
        if vehicle.speed_index > 0:
            actions.append(env_copy.ACTIONS_ALL['SLOWER'])
        return actions

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        average_speed = 0
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1
        '''
        # action masking
        if self.config["action_masking"]:
            self.new_action = self._get_available_actions(self.controlled_vehicles[i], self)
        else:
            self.new_action = action
        '''
        self.new_action = action

        # action is a tuple, e.g., (2, 3, 0, 1)
        self._simulate(self.new_action)

        if self.use_centralized_critic:
            obs = [self.observation_type.observe(),self.observation_type.observe(is_global=True)]
        else:
            obs = self.observation_type.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()

        # get action masks
        if self.config["action_masking"]:
            available_actions = [[0] * self.n_a] * len(self.controlled_vehicles)
            for i in range(len(self.controlled_vehicles)):
                available_action = self._get_available_actions(self.controlled_vehicles[i], self)
                for a in available_action:
                    available_actions[i][a] = 1
        else:
            available_actions = [[1] * self.n_a] * len(self.controlled_vehicles)

        for v in self.controlled_vehicles:
            average_speed += v.speed
        average_speed = average_speed / len(self.controlled_vehicles)

        self.vehicle_speed.append([v.speed for v in self.controlled_vehicles])
        self.vehicle_pos.append(([v.position[0] for v in self.controlled_vehicles]))
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
            "new_action": self.new_action,
            "action_mask": np.array(available_actions),
            "average_speed": average_speed,
            "vehicle_speed": np.array(self.vehicle_speed),
            "vehicle_position": np.array(self.vehicle_pos)
        }

        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return obs, reward, terminal, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)  # defined in action.py

            self.road.act()  # Execute an action
            self.road.step(1 / self.config["simulation_frequency"])  # propagate the vehicle state given its actions.
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
        self.should_update_rendering = False

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    # 针对固定的CAV求action mask, 实用性比较小
    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        if not isinstance(self.action_type, DiscreteMetaAction):
            raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [self.action_type.actions_indexes['IDLE']]
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
        if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['FASTER'])
        if self.vehicle.speed_index > 0 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['SLOWER'])
        return actions

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.

        If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
        such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback is not None:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def _compute_headway_distance(self, vehicle):
        headway_distance = 60    # 可适当增大，不过60够了
        for v in self.road.vehicles:
            if (v.lane_index == vehicle.lane_index) and (v.position[0] > vehicle.position[0]):
                hd = v.position[0] - vehicle.position[0]
                if hd < headway_distance:
                    headway_distance = hd
        return headway_distance

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.
        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)
        return state_copy

    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        """
        Change the type of all vehicles on the road

        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behaviour(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1 / self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result

    def check_collision(self, vehicle, other, other_trajectories):
        """
        Check for collision with another vehicle.

        :param other: the other vehicle' trajectories or object
        other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]
        """
        if vehicle.crashed or other is vehicle:
            return

        if isinstance(other, Vehicle):
            if self._is_colliding(vehicle, other, other_trajectories):
                vehicle.speed = other_trajectories[2] = min([vehicle.speed, other_trajectories[2]], key=abs)
                vehicle.crashed = other.crashed = True

        elif isinstance(other, Obstacle):
            if self._is_colliding(vehicle, other, other_trajectories):
                vehicle.speed = min([vehicle.speed, 0], key=abs)
                vehicle.crashed = other.hit = True
        elif isinstance(other, Landmark):
            if self._is_colliding(vehicle, other, other_trajectories):
                other.hit = True

    def _is_colliding(self, vehicle, other, other_trajectories):
        # Fast spherical pre-check
        # other_trajectories: [vehicle.position, vehicle.heading, vehicle.speed]

        # Euclidean distance
        if np.linalg.norm(other_trajectories[0] - vehicle.position) > vehicle.LENGTH:
            return False

        # Accurate rectangular check
        return utils.rotated_rectangles_intersect(
            (vehicle.position, 0.9 * vehicle.LENGTH, 0.9 * vehicle.WIDTH, vehicle.heading),
            (other_trajectories[0], 0.9 * other.LENGTH, 0.9 * other.WIDTH, other_trajectories[1]))


class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = np.array(list(info["agents_rewards"]))
        done = np.array(list(info["agents_dones"]))
        return obs, reward, done, info
