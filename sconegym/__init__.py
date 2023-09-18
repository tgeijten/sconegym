import os
from gym.envs.registration import register

curr_dir = os.path.dirname(os.path.abspath(__file__))


# Walk Environments
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
             'name': 'h0918',
             'leg_switch': True})


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
             'name': 'h1622',
             'leg_switch': True})


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
             'name': 'h2190',
             'leg_switch': True})


# Run Environments
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
             'name': 'h0918',
             'leg_switch': True})


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
             'name': 'h1622',
             'leg_switch': True})


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
             'name': 'h2190',
             'leg_switch': True})

