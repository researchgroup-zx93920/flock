# Starting point
cwd=$(pwd)

# Ensure library dir
mkdir -p ~/ulib

# PACKAGE 1: Install spdlogs 
cd ~/ulib
git clone https://github.com/gabime/spdlog.git
cd spdlog && mkdir build && cd build
cmake .. && make -j

# PACKAGE 2: argp.h >> Installed by default on linux
