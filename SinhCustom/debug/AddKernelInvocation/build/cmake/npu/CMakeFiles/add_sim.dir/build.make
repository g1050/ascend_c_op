# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/python3.9.2/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/python3.9.2/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/build

# Utility rule file for add_sim.

# Include any custom commands dependencies for this target.
include cmake/npu/CMakeFiles/add_sim.dir/compiler_depend.make

# Include the progress variables for this target.
include cmake/npu/CMakeFiles/add_sim.dir/progress.make

add_sim: cmake/npu/CMakeFiles/add_sim.dir/build.make
.PHONY : add_sim

# Rule to build all files generated by this target.
cmake/npu/CMakeFiles/add_sim.dir/build: add_sim
.PHONY : cmake/npu/CMakeFiles/add_sim.dir/build

cmake/npu/CMakeFiles/add_sim.dir/clean:
	cd /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/build/cmake/npu && $(CMAKE_COMMAND) -P CMakeFiles/add_sim.dir/cmake_clean.cmake
.PHONY : cmake/npu/CMakeFiles/add_sim.dir/clean

cmake/npu/CMakeFiles/add_sim.dir/depend:
	cd /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/cmake/npu /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/build /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/build/cmake/npu /data/xkgao/code/ascend_c_op/SinhCustom/debug/AddKernelInvocation/build/cmake/npu/CMakeFiles/add_sim.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : cmake/npu/CMakeFiles/add_sim.dir/depend

