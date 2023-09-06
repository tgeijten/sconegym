from gym.envs.registration import register

# Normal models
register(id="sconegaith0918-v0", entry_point="sconegym.gaitgym:GaitGymH0918")
register(id="sconegaith1622-v0", entry_point="sconegym.gaitgym:GaitGymH1622")
register(id="sconegaith2190-v0", entry_point="sconegym.gaitgym:GaitGymH2190")

# Stronger models
register(id="sconegaith0918S2-v0", entry_point="sconegym.gaitgym:GaitGymH0918S2")
register(id="sconegaith1622S2-v0", entry_point="sconegym.gaitgym:GaitGymH1622S2")
register(id="sconegaith2190S2-v0", entry_point="sconegym.gaitgym:GaitGymH2190S2")

# special settings tutorial
# TODO automate registration for all envs
register(id="sconegaith0918_delay-v0", entry_point="sconegym.gaitgym:GaitGymDelayH0918")
register(id="sconegaith0918_measure-v0", entry_point="sconegym.gaitgym:GaitGymMeasureH0918")
