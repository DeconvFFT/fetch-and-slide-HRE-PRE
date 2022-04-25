# fetch-and-slide-HRE-PRE
In this project, I attempt to solve fetch and slide open gym environment with Hindsight Experience Replay and the I experiment with Prioritised experience replay to see if there are any performance improvements

# Environments tested
- FetchSlide-v1
- FetchPickAndPlace-v1

# Environment details

- **FetchSlide-v1**

    This environment 

- **FetchPickAndPlace-v1**

    This environment

# Algorithms

## Hindsight Experience Replay (HER)

## Priortised Experience Replay (PER)

# Network Acrhitecture

## Actor Network

## Critic Network

## Target updates

# Experiment Results
![actor55](/Users/saumyamehta/Desktop/RL/fetch-and-slide-HRE-PRE/video/per_colab/actor_100/vid.mp4)
# Abalation study

# Multi processing setup
- Used Mpi4py module to use message passing capabilities within python. I use it to exploit multiple processors on my system.
- mpiutils has two classes to synchronize data across multiple cpus

    - **sync_networks:**

        This function synchronises network's parameters across multiple cpus. It ensures that we can easily collect data from each network by broadcasting parameters from each network

    - **sync_grads:**

        This function synchronises gradients across networks by reducing flat gradients across networks

# Running on google colab
- Google colab needs some dependencies that are missing from the base python environment they provide. The following steps helped me enable openai-gym support on google colab.
    - !apt-get install -y \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        software-properties-common
        !apt-get install -y patchelf
    
        This command installs the necessary libraries to run mujoco environment like libglew and mesa.

    - !pip install gym : This command loads openai gym package into the base environment
    - !pip install free-mujoco-py: This command installs mujoco-py which enables you to run mujoco with openai gym. This is different from local mujoco installation and does not come with opencl support
    - !pip install mpi4py: This command enables mpi support for python. This is important if you want to work on a complex environment like mujoco. It helps you run different environments on different cpus and then you can gather results/ broadcast parameters across cpus.
    - !mpirun --allow-run-as-root -np 8 python3 main.py --parameter1=value --parameter2=value...: The "--allow-run-as-root" is not recommended for google colab, but I found that I couldn't run my program with mpi without this command.


# Setup difficulties with m1 mac
- There were no conda packages available for mujoco-py, so I had to install mujoco from pip. This required setting a few things before installing mujoco. The install script is in install-mujoco_dummy.sh file. Replace version of mujoco with the approporiate version in your install.
- After creating the install script, you need to setup CC, CXX, LDFLAGS and CXXFLAGS in order to ensure mujoco runs off clang instead of gcc. The paths would be in llvm folder inside opt/homebrew/opt.
- You might have to downgrade mujoco version in order for gym to use mujoco. This could be due to dependency issues with the c++ source files(dylib).

# How to run
- Use the folllowing command to train the model
    ```
        mpirun -np 8 python3 main.py --per=True 
    ```
- Use the following command to test the model
    ```
        python3 main.py --mode=test --per=True
    ```