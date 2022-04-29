# fetch-and-slide-HRE-PRE
In this project, I attempt to solve fetch and slide open gym environment with Hindsight Experience Replay and the I experiment with Prioritised experience replay to see if there are any performance improvements

# Environments tested
- FetchSlide-v1
- FetchPickAndPlace-v1

# Environment details

- **FetchSlide-v1**

    - It is an openai robotics environment which uses mujoco physics engine simulator to handle the environment physics. The state space is continuous and the action space is 4 dimensional. The first three dimensions specify the location of agent and 4th dimension specify the distance between claws of the agent. Actions will decide where the robot hand will be in space. There are two sets of goals, **Achieved Goal** and **Desired Goal**. Achieved goal is the goal achieved after performing an action in a given state. Desired Goal is the goal which we want the agent to achieve after performing an action from a state.
    - The agent(robot arm) tries to slide a puck towards a goal location on a table in front of it. The surface of the table has some friction as well. The main aim in this environment is for the agent to learn to slide the puck towards it's desired location. 

- **FetchPickAndPlace-v1**

    - It is an openai robotics environment which uses mujoco physics engine simulator to handle the environment physics. The state space is continuous and the action space is 4 dimensional. The first three dimensions specify the location of agent and 4th dimension specify the distance between claws of the agent. Actions will decide where the robot hand will be in space. There are two sets of goals, **Achieved Goal** and **Desired Goal**. Achieved goal is the goal achieved after performing an action in a given state. Desired Goal is the goal which we want the agent to achieve after performing an action from a state.
    - The agent(robot arm) tries to pick up a block and place it at the goal location on front of it. The location can either be in space above the table or on the table. The main aim in this environment is for the agent to learn to pick up the block and place it at the desired location

# Algorithms

- For this project, I end up using **Deep deterministic policy gradient** as the off policy algorithm of choice. And then I add HER and HER+PER on top of that to learn.

## Deep deterministic Policy Gradient (DDPG)
![DDPG algorithm](/algorithms/ddpg_algo.png)

- DDPG algorithm creates noisy actions by adding OU noise to   
## Hindsight Experience Replay (HER)

## Priortised Experience Replay (PER)

# Network Acrhitecture

## Actor Network

## Critic Network

## Target updates

# Experiment Results

![FetchSlide-v1 100 epochs](https://user-images.githubusercontent.com/27497059/165143321-05d9f8fa-cb39-4324-911b-92804219b567.mp4)

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
