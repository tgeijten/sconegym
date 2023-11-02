import os
from gym.envs.registration import register
curr_dir = os.path.dirname(os.path.abspath(__file__))


# Walk Environments
register(id="sconewalk_h0918-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H0918-v1.scone',
             'obs_type': '2D',
             'left_leg_idxs': [3, 4, 5],
             'right_leg_idxs': [6, 7, 8],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         }
        )

register(id="sconewalk_h0918_osim-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H0918_osim-v1.scone',
             'obs_type': '2D',
             'left_leg_idxs': [3, 4, 5],
             'right_leg_idxs': [6, 7, 8],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         }
        )

register(id="sconewalk_h1622-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H1622-v1.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10],
             'right_leg_idxs': [11, 12, 13, 14, 15],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         }
        )

register(id="sconewalk_h0918-v0",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H0918.scone',
             'obs_type': '2D',
             'left_leg_idxs': [3, 4, 5],
             'right_leg_idxs': [6, 7, 8],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         }
        )

register(id="sconewalk_h1622-v0",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H1622.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10],
             'right_leg_idxs': [11, 12, 13, 14, 15],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         }
        )

register(id="sconewalk_h2190-v0",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H2190.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10, 11],
             'right_leg_idxs': [12, 13, 14, 15, 16, 17],
             'clip_actions': True,
             'run': False,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": 0.0,
             }
         },
        )


# Run Environments
register(id="sconerun_h0918-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H0918S-v1.scone',
             'obs_type': '2D',
             'left_leg_idxs': [3, 4, 5],
             'right_leg_idxs': [6, 7, 8],
             'clip_actions': False,
             'run': True,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": -10,
             }
         }
        )

register(id="sconerun_h1622-v1",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H1622S-v1.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10],
             'right_leg_idxs': [11, 12, 13, 14, 15],
             'clip_actions': False,
             'run': True,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": -10,
             }
         })

register(id="sconerun_h0918-v0",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H0918_S2.scone',
             'obs_type': '2D',
             'left_leg_idxs': [3, 4, 5],
             'right_leg_idxs': [6, 7, 8],
             'clip_actions': False,
             'run': True,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": -10,
             }
         }
        )

register(id="sconerun_h1622-v0",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H1622_S2.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10],
             'right_leg_idxs': [11, 12, 13, 14, 15],
             'clip_actions': False,
             'run': True,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": -10,
             }
         })

register(id="sconerun_h2190-v0",
         entry_point="sconegym.gaitgym:GaitGym",
         kwargs={
             'model_file': curr_dir + '/data/H2190_S2.scone',
             'obs_type': '3D',
             'left_leg_idxs': [6, 7, 8, 9, 10, 11],
             'right_leg_idxs': [12, 13, 14, 15, 16, 17],
             'clip_actions': False,
             'run': True,
             'target_vel': 1.2,
             'leg_switch': True,
             'rew_keys':{
                "vel_coeff": 10,
                "grf_coeff": -0.07281,
                "joint_limit_coeff": -0.1307,
                "smooth_coeff": -0.097,
                "nmuscle_coeff": -1.57929,
                "self_contact_coeff": -10,
             }
         }
        )
