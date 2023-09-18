import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import gym
import numpy as np

# Add sconepy folders to path
if sys.platform.startswith("win"):
    sys.path.append("C:/Program Files/SCONE/bin")
elif sys.platform.startswith("linux"):
    sys.path.append("/opt/scone/lib")
elif sys.platform.startswith('darwin'):
    sys.path.append("/Applications/SCONE.app/Contents/MacOS/lib")

import sconepy


def find_model_file(model_file):
    this_dir, this_file = os.path.split(__file__)
    return os.path.join(this_dir, "data", model_file)


class SconeGym(gym.Env, ABC):
    """
    Main general purpose class that gives you a gym-ready sconepy interface
    It has to be inherited by the environments you actually want to use and
    some methods have to be defined (see end of class). This class would probably
    be a good starting point for new environments.
    New environments also have to be registered in sconegym/__init__.py !
    """

    def __init__(self,
                 model_file,
                 left_leg_idxs,
                 right_leg_idxs,
                 name='none',
                 clip_actions = False,
                 target_vel = 1.2,
                 leg_switch = True,
                 use_delayed_sensors=False,
                 use_delayed_actuators=False,
                 run = False,
                 obs_type = '2D',
                 *args, **kwargs):
        # Internal settings
        self.name = name
        self.episode = 0
        self.total_reward = 0.0
        self.init_dof_pos_std = 0.05
        self.init_dof_vel_std = 0.1
        self.init_load = 0.5
        self.init_activations_mean = 0.3
        self.init_activations_std = 0.1
        self.min_com_height = 0.5
        self.step_size = 0.01
        self.total_steps = 0
        self.steps = 0
        self.has_reset = False
        self.store_next = False
        # Settings from kwargs
        self.target_vel = target_vel
        self.use_delayed_sensors = use_delayed_sensors
        self.use_delayed_actuators = use_delayed_actuators
        self.clip_actions = clip_actions
        self.leg_switch = leg_switch
        self.run = run
        self.obs_type = obs_type
        self.left_leg_idxs = left_leg_idxs
        self.right_leg_idxs = right_leg_idxs

        super().__init__(*args, **kwargs)
        sconepy.set_log_level(3)
        self.model = sconepy.load_model(model_file)
        self.init_dof_pos = self.model.dof_position_array().copy()
        self.init_dof_vel = self.model.dof_velocity_array().copy()
        self.set_output_dir("DATE_TIME." + self.model.name())
        self._find_head_body()
        self._setup_action_observation_spaces()


    def step(self, action):
        """
        takes an action and advances environment by 1 step.
        """
        if self.clip_actions:
            action = np.clip(action, 0, 0.5)
        else:
            action = np.clip(action, 0, 1.0)
        if not self.has_reset:
            raise Exception("You have to call reset() once before step()")

        if self.use_delayed_actuators:
            self.model.set_delayed_actuator_inputs(action)
        else:
            self.model.set_actuator_inputs(action)

        self.model.advance_simulation_to(self.time + self.step_size)
        reward = self._get_rew()
        obs = self._get_obs()
        done = self._get_done()
        reward = self._apply_termination_cost(reward, done)
        self.time += self.step_size
        self.total_reward += reward

        if done:
            if self.store_next:
                self.model.write_results(
                    self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
                )
                self.store_next = False
            self.episode += 1
        return obs, reward, done, {}

    def write_now(self):
        if self.store_next:
            self.model.write_results(
                self.output_dir, f"{self.episode:05d}_{self.total_reward:.3f}"
            )
            self.store_next = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        """
        Reset and randomize the initial state.
        """
        self.episode_number = np.random.randint(0, 1000000)
        self.model.reset()
        self.has_reset = True
        self.time = 0
        self.total_reward = 0.0
        self.steps = 0

        # Check if data should be stored (slow)
        self.model.set_store_data(self.store_next)
        # Randomize initial pose
        dof_pos = self.init_dof_pos + np.random.normal(
            0, self.init_dof_pos_std, len(self.init_dof_pos)
        )
        self.model.set_dof_positions(dof_pos)
        dof_vel = self.init_dof_vel + np.random.normal(
            0, self.init_dof_vel_std, len(self.init_dof_vel)
        )
        self.model.set_dof_velocities(dof_vel)
        if self.leg_switch:
            if np.random.uniform() < 0.5:
                self._switch_legs()

        # Randomize initial muscle activations
        muscle_activations = np.clip(
            np.random.normal(
                self.init_activations_mean,
                self.init_activations_std,
                size=len(self.model.muscles()),
            ),
            0.01,
            1.0,
        )
        self.prev_acts = muscle_activations
        self.prev_excs = self.model.muscle_excitation_array()
        self.model.init_muscle_activations(muscle_activations)

        # Initialize state and equilibrate muscles
        self.model.init_state_from_dofs()

        if self.init_load > 0:
            self.model.adjust_state_for_load(self.init_load)

        obs = self._get_obs()
        if return_info:
            return obs, (obs, {})
        else:
            return obs

    def store_next_episode(self):
        """
        Primes the environment to store the next episode.
        This also calls reset() to ensure that the data is
        written correctly.
        """
        self.store_next = True
        self.reset()

    def set_output_dir(self, dir_name):
        self.output_dir = sconepy.replace_string_tags(dir_name)

    def manually_load_model(self):
        self.model = sconepy.load_model(self.model_file)
        self.model.set_store_data(True)

    def render(self, *args, **kwargs):
        """
        Not yet supported
        """
        return

    def model_velocity(self):
        return self.model.com_vel().x

    def _setup_action_observation_spaces(self):
        num_act = len(self.model.actuators())
        self.action_space = gym.spaces.Box(
            low=np.zeros(shape=(num_act,)),
            high=np.ones(shape=(num_act,)),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-10000, high=10000, shape=self._get_obs().shape, dtype=np.float32
        )

    def _find_head_body(self):
        head_names = ["torso", "head", "lumbar"]
        self.head_body = None
        for b in self.model.bodies():
            if b.name() in head_names:
                self.head_body = b
        if self.head_body is None:
            raise Exception("Could not find head body")

    def _switch_legs(self):
        """
        Switches leg joint angles. Good for initial
        state randomization.
        """
        pos = self.model.dof_position_array()
        vel = self.model.dof_velocity_array()
        for left, right in zip(self.left_leg_idxs, self.right_leg_idxs):
            pos[left], pos[right] = pos[right], pos[left]
            vel[left], vel[right] = vel[right], vel[left]
        self.model.set_dof_positions(pos)
        self.model.set_dof_velocities(vel)

    def apply_args(self):
        pass

    def _apply_termination_cost(self, reward, done):
        return reward

    # these all need to be defined by environments
    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _get_rew(self):
        pass

    @abstractmethod
    def _get_done(self):
        pass


class GaitGym(SconeGym):
    def __init__(self, model_file, *args, **kwargs):
        self._max_episode_steps = 1000
        super().__init__(model_file, *args, **kwargs)
        self.rwd_dict = None
        self.mass = np.sum([x.mass() for x in self.model.bodies()])
        self.vel_coeff = 0.0
        self.grf_coeff = 0.0
        self.smooth_coeff = 0.0
        self.nmuscle_coeff = 0.0
        self.joint_limit_coeff = 0.0
        self.self_contact_coeff = 0.0

    def _get_obs(self):
        if self.obs_type == '2D':
            return self._get_obs_2d()
        elif self.obs_type == '3D':
            return self._get_obs_3d()
        else:
            raise NotImplementedError

    def _get_obs_3d(self):
        acts = self.model.muscle_activation_array()
        self.prev_acts = self.model.muscle_activation_array().copy()
        self.prev_excs = self.model.muscle_excitation_array()
        dof_values = self.model.dof_position_array()
        dof_vels = self.model.dof_velocity_array()
        # No x or y position in the state
        dof_values[3] = 0.0
        dof_values[5] = 0.0
        return np.concatenate(
            [
                self.model.muscle_fiber_length_array(),
                self.model.muscle_fiber_velocity_array(),
                self.model.muscle_force_array(),
                self.model.muscle_excitation_array(),
                self.head_body.orientation().array(),
                self.head_body.ang_vel().array(),
                self._get_feet_relative_position(),
                dof_values,
                dof_vels,
                acts,
            ],
            dtype=np.float32,
        ).copy()

    def _get_feet_relative_position(self):
        pelvis = (
            [x for x in self.model.bodies() if "pelvis" in x.name()][0]
            .com_pos()
            .array()
        )
        foot_l = (
            [x for x in self.model.bodies() if "calcn_l" in x.name()][0]
            .com_pos()
            .array()
        )
        foot_r = (
            [x for x in self.model.bodies() if "calcn_r" in x.name()][0]
            .com_pos()
            .array()
        )
        return np.concatenate([foot_l - pelvis, foot_r - pelvis], dtype=np.float32)

    def _get_obs_2d(self):
        acts = self.model.muscle_activation_array()
        self.prev_acts = self.model.muscle_activation_array().copy()
        self.prev_excs = self.model.muscle_excitation_array()
        dof_values = self.model.dof_position_array()
        dof_vels = self.model.dof_velocity_array()
        # No x position in the state
        dof_values[1] = 0.0

        if not self.use_delayed_sensors:
            return np.concatenate(
                [
                    self.model.muscle_fiber_length_array(),
                    self.model.muscle_fiber_velocity_array(),
                    self.model.muscle_force_array(),
                    self.model.muscle_excitation_array(),
                    self.head_body.orientation().array(),
                    self.head_body.ang_vel().array(),
                    self._get_feet_relative_position(),
                    dof_values,
                    dof_vels,
                    acts,
                ],
                dtype=np.float32,
            ).copy()

        else:
            return np.concatenate(
                [
                    self.model.delayed_muscle_fiber_length_array(),
                    self.model.delayed_muscle_fiber_velocity_array(),
                    self.model.delayed_muscle_force_array(),
                    self.model.delayed_vestibular_array(),
                    self.model.muscle_excitation_array(),
                    self.model.muscle_activation_array(),
                ],
                dtype=np.float32,
            ).copy()

    def _get_rew(self):
        """
        Reward function.
        """
        self.total_steps += 1
        self.steps += 1
        return self.custom_reward()

    def custom_reward(self):
        self._update_rwd_dict()
        return np.sum(list(self.rwd_dict.values()))

    def _update_rwd_dict(self):
        self.rwd_dict = {
            "gaussian_vel": self.vel_coeff * self._gaussian_plateau_vel(),
            "grf": -self.grf_coeff * self._grf(),
            "smooth": -self.smooth_coeff * self._exc_smooth_cost(),
            "number_muscles": -self.nmuscle_coeff * self._number_muscle_cost(),
            "constr": -self.joint_limit_coeff * self._joint_limit_torques(),
            "self_contact": -self.self_contact_coeff * self._get_self_contact(),
        }
        return self.rwd_dict

    def get_rwd_dict(self):
        if not self.rwd_dict:
            self.rwd_dict = self._update_rwd_dict()
        rwd_dict = {k: v for k, v in self.rwd_dict.items()}
        return rwd_dict

    def _number_muscle_cost(self):
        """
        Get number of muscle with activity over 0.15.
        """
        return self._get_active_muscles(0.15)

    def _get_active_muscles(self, threshold):
        """
        Get the number of muscles whose activity is above the threshold.
        """
        return (
            np.sum(
                np.where(self.model.muscle_activation_array() > threshold)[0].shape[0]
            )
            / self.action_space.shape[0]
        )

    def _gaussian_vel(self):
        vel = self.model_velocity()
        return np.exp(-np.square(vel - self.target_vel))

    def _gaussian_plateau_vel(self):
        if self.run:
            return self.model_velocity()
        vel = self.model_velocity()
        if vel < self.target_vel:
            return self._gaussian_vel()
        else:
            return 1.0

    def _exc_smooth_cost(self):
        excs = self.model.muscle_excitation_array()
        delta_excs = excs - self.prev_excs
        return np.mean(np.square(delta_excs))

    def _get_self_contact(self):
        ignore_bodies = ["calcn_r", "calcn_l"]
        contact_force = np.sum(
            [
                np.abs(x.contact_force().array())
                for x in self.model.bodies()
                if x.name() not in ignore_bodies
            ]
        )
        return np.clip(contact_force, -100, 100) / 100

    def _joint_limit_torques(self):
        return np.mean(
            [np.mean(np.abs(x.limit_torque().array())) for x in self.model.joints()]
        )

    def _grf(self):
        grf = self.model.contact_load()
        return max(0, grf - 1.2)

    def _get_done(self) -> bool:
        """
        The episode ends if the center of mass is below min_com_height.
        """
        if self.model.com_pos().y < self.min_com_height:
            return True
        if self.head_body.com_pos().y < 0.9:
            return True
        return False

    @property
    def horizon(self):
        # TODO put this in model kwargs such that it works with deprl
        return 1000


# Tutorial environments to see features
# The Measure one needs to be fixed

# TODO @thomas add right model file
class GaitGymMeasureH0918(GaitGym):
    """
    Shows how to use custom measures from the .scone files in
    python.
    """

    def __init__(self, *args, **kwargs):
        self.name = "H0918"
        self.delay = False
        super().__init__(find_model_file("H0918_hfd_measure.scone"), *args, **kwargs)

    def custom_reward(self):
        self.rwd_dict = self.create_rwd_dict()
        return self.model.current_measure()


