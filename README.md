# TAMP Block Stacking - Project 5

Task and Motion Planning system for robotic block manipulation using Genesis simulator, OMPL, and Pyperplan.

Authors

Luis Alzamora
John Hoang Do
Josh Ethan Nirmal

Course: RBE 550 Motion Planning, Fall 2025
Institution: Worcester Polytechnic Institute

# Pre-requisite:
- Ubuntu 22.04
- Python 3.10 or newer (check using python3 --version)
- Virtual environment is created using python venv. Instructions might be adapted for using conda.
- Additional requirements:
- torch>=2.0.0
- torchvision>=0.15.0
- genesis-world>=0.2.0
- pyperplan>=2.0.0
- numpy>=1.24.0
- scipy>=1.10.0
- matplotlib>=3.7.0

# Installation steps:
1) Setup python venv: python3 -m venv project5
2) Activate: source project5/bin/activate
Update pip: pip install --upgrade pip
3) Install pytorch: Check CUDA: nvcc --version. If CUDA version is less than 12.6, use CPU. Otherwise, use appropriate version for pytorch
a) CPU: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
4) Install OMPL:
    a) Check python version: python --version
    b) Download OMPL python bindings, then move the downloaded file to the directory in use
    c) Install: pip install ompl-1.7.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
5) Install Genesis (will take a few minutes): pip install genesis-world
6) Install Pyperplan: pip install pyperplan
7) Install dependencies: pip install numpy scipy matplotlib
8) Navigate to project folder (if not yet)
9) Check to make sure Genesis is running properly: python3 demo.py gpu (will take a few minutes)
10) python3 tamp_main.py --goal1

To run other goals, replace --goal1 with the option for the goal you want, such as:
1) --goal1        # 2 3-block towers
2) --goal2        # 5-block tower
3) --goal3        # Extra-Credit (6-blocks)
4) --goal3-ext    # Extra Credit (10-blocks)
5) --goal4a       # 6 2-block towers in 3x4 grid
6) --goal4b       # 3 red and 3 green blocks in L shapes

# Project Structure
project5/

├── abstraction.py          # Symbolic state abstraction (PDDL predicates)
├── goals.py                # Goal definitions (1-4)
├── task_planner.py         # Pyperplan interface (A* search)
├── planning.py             # Motion primitives & OMPL integration
├── tamp_main.py            # Main TAMP loop & execution
├── scenes.py               # Genesis scene setup
├── robot_adapter.py        # Robot control wrapper
├── demo.py                 # Genesis installation test
└── blocksworld_domain.pddl # PDDL domain file

# Reference: 
Genesis Simulator: https://genesis-world.readthedocs.io
OMPL: https://ompl.kavrakilab.org
Pyperplan: https://github.com/aibasel/pyperplan
https://www.youtube.com/watch?v=iEE3HqHF34o (Video is for windows install but can be helpful for linux for reference)
https://www.youtube.com/watch?v=RBZ16oUv5A0

CPU vs. GPU: The following steps can be done to check if your computer has GPU and can be used instead of relying on CPU
1) lspci | grep -i nvidia
2) ubuntu-drivers devices
3) sudo apt update
4) sudo apt install nvidia-driver-XXX (choose the one said recommended in step 2) 
