Running the code
source venv/bin/activate   # activate your virtual environment
python tamp_main.py --goal1  # run Goal 1

To run other goals, replace --goal1 with the option for the goal you want, such as:
    --goal2 //5-block tower
    --goal3 //Extra-Credit (6-blocks)
    --goal3-ext //Extra Credit (10-blocks)
    --goal4a //yellow blocks
    --goal4b //red and green blocks in L shapes

Installation SetpsSteps to install everything so far: 
1) Install conda  conda create -y -n rbe550 python=3.11
2) Install pytorch (I have cuda 13.00) pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
3) Install OMPL python Bindings (Donwload from here[] depending on your system 
Then pip install ompl-1.7.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.wh
4) Install Genesis pip install genesis

conda activate rbe550
python run demo.py

Pre-requisite:
- Ubuntu 22.04
- Virtual environment is created using python venv. Instructions might be adapted for using conda.

Installation steps:
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
7) Navigate to project folder (if not yet)
8) python3 demo.py gpu (will take a few minutes)
9) python3 tamp_main.py


Reference: https://www.youtube.com/watch?v=iEE3HqHF34o (Video is for windows install but can be helpful for linux for reference)
https://www.youtube.com/watch?v=RBZ16oUv5A0

CPU vs. GPU: The following steps can be done to check if your computer has GPU and can be used instead of relying on CPU
1) lspci | grep -i nvidia
2) ubuntu-drivers devices
3) sudo apt update
4) sudo apt install nvidia-driver-XXX (choose the one said recommended in step 2) 