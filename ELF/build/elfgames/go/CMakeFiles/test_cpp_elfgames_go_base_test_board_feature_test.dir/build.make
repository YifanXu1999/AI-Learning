# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Go/ELF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Go/ELF/build

# Include any dependencies generated for this target.
include elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/depend.make

# Include the progress variables for this target.
include elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/progress.make

# Include the compile flags for this target's objects.
include elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/flags.make

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/flags.make
elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o: ../src_cpp/elfgames/go/base/test/board_feature_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o"
	cd /Go/ELF/build/elfgames/go && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o -c /Go/ELF/src_cpp/elfgames/go/base/test/board_feature_test.cc

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.i"
	cd /Go/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Go/ELF/src_cpp/elfgames/go/base/test/board_feature_test.cc > CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.i

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.s"
	cd /Go/ELF/build/elfgames/go && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Go/ELF/src_cpp/elfgames/go/base/test/board_feature_test.cc -o CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.s

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.requires:

.PHONY : elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.requires

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.provides: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.requires
	$(MAKE) -f elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/build.make elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.provides.build
.PHONY : elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.provides

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.provides.build: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o


# Object files for target test_cpp_elfgames_go_base_test_board_feature_test
test_cpp_elfgames_go_base_test_board_feature_test_OBJECTS = \
"CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o"

# External object files for target test_cpp_elfgames_go_base_test_board_feature_test
test_cpp_elfgames_go_base_test_board_feature_test_EXTERNAL_OBJECTS =

elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/build.make
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: elfgames/go/libelfgames_go9.a
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: lib/libgtest.a
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: /usr/lib/x86_64-linux-gnu/libzmq.so
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: elf/libelf.a
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: /root/miniconda3/envs/elf/lib/libpython3.7m.so
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: third_party/tbb/tbb_cmake_build_subdir_release/libtbb.so.2
elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_cpp_elfgames_go_base_test_board_feature_test"
	cd /Go/ELF/build/elfgames/go && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/build: elfgames/go/test_cpp_elfgames_go_base_test_board_feature_test

.PHONY : elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/build

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/requires: elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/base/test/board_feature_test.cc.o.requires

.PHONY : elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/requires

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/clean:
	cd /Go/ELF/build/elfgames/go && $(CMAKE_COMMAND) -P CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/cmake_clean.cmake
.PHONY : elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/clean

elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/depend:
	cd /Go/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Go/ELF /Go/ELF/src_cpp/elfgames/go /Go/ELF/build /Go/ELF/build/elfgames/go /Go/ELF/build/elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : elfgames/go/CMakeFiles/test_cpp_elfgames_go_base_test_board_feature_test.dir/depend

