# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build

# Include any dependencies generated for this target.
include cmake/npu/CMakeFiles/add_npu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include cmake/npu/CMakeFiles/add_npu.dir/compiler_depend.make

# Include the progress variables for this target.
include cmake/npu/CMakeFiles/add_npu.dir/progress.make

# Include the compile flags for this target's objects.
include cmake/npu/CMakeFiles/add_npu.dir/flags.make

cmake/npu/CMakeFiles/add_npu.dir/__/__/add_custom.cpp.o: cmake/npu/CMakeFiles/add_npu.dir/flags.make
cmake/npu/CMakeFiles/add_npu.dir/__/__/add_custom.cpp.o: /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_custom.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CCE object cmake/npu/CMakeFiles/add_npu.dir/__/__/add_custom.cpp.o"
	cd /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/cmake/npu && /usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec -xcce $(CCE_DEFINES) $(CCE_INCLUDES) -I"/usr/local/Ascend/ascend-toolkit/latest/acllib/include" -I"/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw" -I"/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw/impl" -I"/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw/interface" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/tikcfw" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/tikcfw/impl" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/tikcfw/interface" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/include"  $(CCE_FLAGS) --cce-aicore-arch=dav-m200 -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-fp-ceiling=2 -mllvm -cce-aicore-record-overflow=false --cce-auto-sync -mllvm -api-deps-filter -fPIC -pthread -o CMakeFiles/add_npu.dir/__/__/add_custom.cpp.o -c /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_custom.cpp

cmake/npu/CMakeFiles/add_npu.dir/__/__/main.cpp.o: cmake/npu/CMakeFiles/add_npu.dir/flags.make
cmake/npu/CMakeFiles/add_npu.dir/__/__/main.cpp.o: /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CCE object cmake/npu/CMakeFiles/add_npu.dir/__/__/main.cpp.o"
	cd /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/cmake/npu && /usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec -xcce $(CCE_DEFINES) $(CCE_INCLUDES) -I"/usr/local/Ascend/ascend-toolkit/latest/acllib/include" -I"/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw" -I"/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw/impl" -I"/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/tikcfw/interface" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/tikcfw" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/tikcfw/impl" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikcpp/tikcfw/interface" -I"/usr/local/Ascend/ascend-toolkit/latest/tools/tikicpulib/lib/include"  $(CCE_FLAGS) --cce-aicore-arch=dav-m200 -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-fp-ceiling=2 -mllvm -cce-aicore-record-overflow=false --cce-auto-sync -mllvm -api-deps-filter -fPIC -pthread -o CMakeFiles/add_npu.dir/__/__/main.cpp.o -c /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/main.cpp

# Object files for target add_npu
add_npu_OBJECTS = \
"CMakeFiles/add_npu.dir/__/__/add_custom.cpp.o" \
"CMakeFiles/add_npu.dir/__/__/main.cpp.o"

# External object files for target add_npu
add_npu_EXTERNAL_OBJECTS =

/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_npu: cmake/npu/CMakeFiles/add_npu.dir/__/__/add_custom.cpp.o
/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_npu: cmake/npu/CMakeFiles/add_npu.dir/__/__/main.cpp.o
/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_npu: cmake/npu/CMakeFiles/add_npu.dir/build.make
/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_npu: cmake/npu/CMakeFiles/add_npu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CCE executable /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_npu"
	cd /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/cmake/npu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/add_npu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cmake/npu/CMakeFiles/add_npu.dir/build: /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/add_npu
.PHONY : cmake/npu/CMakeFiles/add_npu.dir/build

cmake/npu/CMakeFiles/add_npu.dir/clean:
	cd /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/cmake/npu && $(CMAKE_COMMAND) -P CMakeFiles/add_npu.dir/cmake_clean.cmake
.PHONY : cmake/npu/CMakeFiles/add_npu.dir/clean

cmake/npu/CMakeFiles/add_npu.dir/depend:
	cd /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/cmake/npu /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/cmake/npu /data/xkgao/code/ascend_c_op/Addcmul/debug/AddKernelInvocation/build/cmake/npu/CMakeFiles/add_npu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cmake/npu/CMakeFiles/add_npu.dir/depend
