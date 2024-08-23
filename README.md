# sconegym
Gym environments for predictive biomechanical simulations using reinforcement learning.
## Installation
1. Make sure Python 3.9 is installed on your system
2. Install the [latest version of SCONE](https://scone.software)
3. *Optional*: for faster simulations and access to all gym environments, be sure to activate the [Hyfydy simulation engine](https://scone.software/doku.php?id=hyfydy) inside SCONE. Without Hyfydy, you are limited to using OpenSim, which takes much longer to optimize. More information and a free trial can be found on the [Hyfydy website](https://hyfydy.com). 
4. Clone the sconegym repository
5. Open a console, navigate to your local sconegym folder and type:
6. `pip install -r requirements.txt` to install the requirements
7. `pip install -e .` (the `-e` flag will ensure the package automatically gets updated when you update the sconegym repository)
8. To see if everything works, try out the `example_environment.py` or `test_environments.py` from the sconegym folder
9. To test sconegym in combination with [depRL](https://github.com/martius-lab/depRL), try running `example_deprl.py`, after following the instructions on the [depRL website](https://github.com/martius-lab/depRL)
## Render and Analyze
Results of an optimization can be rendered and analyzed in SCONE Studio, using the following steps:
1. Open SCONE Studio
2. In the Optimization Results pane on the left, navigate to any checkpoint file (extension `.pt`) and double-click the file
3. A number of rollouts will be performed, and results will be stored as `.sto` files inside the SCONE data folder, in a subfolder below the `.pt file`. The name of the subfolder starts with `run_checkpoint` and ends with the checkpoint number.
4. Double-click on any of the `.sto` files to open and display them
5. In addition to the 3D renders, results can be analyzed via the Analysis Window, and gaits analysis can be performed via `Tools -> Gait Analysis`
6. For more information, see the [SCONE Website](https://scone.software)
## Contributors
* Pierre Schumacher [@P-Schumacher](https://github.com/P-Schumacher)
* Thomas Geijtenbeek [@tgeijten](https://github.com/tgeijten)
