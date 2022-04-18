# fetch-and-slide-HRE-PRE
In this project, I attempt to solve fetch and slide open gym environment with Hindsight Experience Replay and the I experiment with Prioritised experience replay to see if there are any performance improvements

# Multi processing setup
- Used Mpi4py module to use message passing capabilities within python. I use it to exploit multiple processors on my system.
- mpiutils has two classes to synchronize data across multiple cpus

    - **sync_networks:**

        This function synchronises network's parameters across multiple cpus. It ensures that we can easily collect data from each network by broadcasting parameters from each network

    - **sync_grads:**

        This function synchronises gradients across networks by reducing flat gradients across networks
        
# Setup difficulties with m1 mac
- There were no conda packages available for mujoco-py, so I had to install mujoco from pip. This required setting a few things before installing mujoco. The install script is in install-mujoco_dummy.sh file. Replace version of mujoco with the approporiate version in your install.
- After creating the install script, you need to setup CC, CXX, LDFLAGS and CXXFLAGS in order to ensure mujoco runs off clang instead of gcc. The paths would be in llvm folder inside opt/homebrew/opt.
- You might have to downgrade mujoco version in order for gym to use mujoco. This could be due to dependency issues with the c++ source files(dylib).