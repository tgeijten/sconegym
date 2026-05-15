import os
from gym.envs.registration import register
curr_dir = os.path.dirname(os.path.abspath(__file__))

# gaitgym register helper function
def register_gaitgym(name, model_file, arg_list):
    register(id=name, entry_point="sconegym.gaitgym:GaitGym",
       kwargs={ 'model_file': curr_dir + model_file, **arg_list })

# Default reward weights
rew_keys_run3d_v2 = {
    "vel_coeff": 10,
    "grf_coeff": -0.3,
    "joint_limit_coeff": -0.4,
    "smooth_coeff": -0.1,
    "nmuscle_coeff": -1.5,
    "self_contact_limit": 100.0,
    "self_contact_coeff": -0.05,
    "muscle_activation_coeff": 0.0,
    "lateral_deviation_coeff": -5.0,
    "hand_distance_coeff": -10.0,
    "hand_foot_coordination_coeff": 2.0,
    "head_stability_coeff": -0.2,
    "head_stability_threashold": 2.0,
    "head_posture_coeff": -0.1,
    "head_posture_threshold": 10.0,
    "root_posture_coeff": -0.04,
    "root_posture_threshold": 15.0,
}

# Default gym settings
run3d_v2 = {
    "obs_type": "3D",
    "clip_actions": False,
    "run": True,
    "target_vel": 1.2,
    "grf_threshold": 3.5,
    "obs_exclude_root_y": True,
    "obs_exclude_root_z": False,
    "use_hand_position_sensors": True,
    "hand_distance_threshold": 0.5,
    "hand_foot_coordination_velocity": 3.0,
    "lateral_deviation_threshold": 0.5,
    "target_head_tilt": -15.0,
    "target_root_tilt": -5.0,
    "velocity_ramp_start": 3.0,
    "velocity_ramp_end": 4.0,
    "init_activations_mean": 0.01,
    "init_activations_std": 0,
    "fall_recovery_time": 0.5,
    "rew_keys": rew_keys_run3d_v2
}

# Register gyms
register_gaitgym("sconerun_h32130-v2", '/data-v2/H32130FIG4_260315.scone', run3d_v2)
