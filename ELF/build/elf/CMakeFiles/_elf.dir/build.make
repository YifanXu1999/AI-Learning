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
include elf/CMakeFiles/_elf.dir/depend.make

# Include the progress variables for this target.
include elf/CMakeFiles/_elf.dir/progress.make

# Include the compile flags for this target's objects.
include elf/CMakeFiles/_elf.dir/flags.make

elf/CMakeFiles/_elf.dir/pybind_module.cc.o: elf/CMakeFiles/_elf.dir/flags.make
elf/CMakeFiles/_elf.dir/pybind_module.cc.o: ../src_cpp/elf/pybind_module.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object elf/CMakeFiles/_elf.dir/pybind_module.cc.o"
	cd /Go/ELF/build/elf && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/_elf.dir/pybind_module.cc.o -c /Go/ELF/src_cpp/elf/pybind_module.cc

elf/CMakeFiles/_elf.dir/pybind_module.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/_elf.dir/pybind_module.cc.i"
	cd /Go/ELF/build/elf && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Go/ELF/src_cpp/elf/pybind_module.cc > CMakeFiles/_elf.dir/pybind_module.cc.i

elf/CMakeFiles/_elf.dir/pybind_module.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/_elf.dir/pybind_module.cc.s"
	cd /Go/ELF/build/elf && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Go/ELF/src_cpp/elf/pybind_module.cc -o CMakeFiles/_elf.dir/pybind_module.cc.s

elf/CMakeFiles/_elf.dir/pybind_module.cc.o.requires:

.PHONY : elf/CMakeFiles/_elf.dir/pybind_module.cc.o.requires

elf/CMakeFiles/_elf.dir/pybind_module.cc.o.provides: elf/CMakeFiles/_elf.dir/pybind_module.cc.o.requires
	$(MAKE) -f elf/CMakeFiles/_elf.dir/build.make elf/CMakeFiles/_elf.dir/pybind_module.cc.o.provides.build
.PHONY : elf/CMakeFiles/_elf.dir/pybind_module.cc.o.provides

elf/CMakeFiles/_elf.dir/pybind_module.cc.o.provides.build: elf/CMakeFiles/_elf.dir/pybind_module.cc.o


# Object files for target _elf
_elf_OBJECTS = \
"CMakeFiles/_elf.dir/pybind_module.cc.o"

# External object files for target _elf
_elf_EXTERNAL_OBJECTS =

elf/_elf.cpython-37m-x86_64-linux-gnu.so: elf/CMakeFiles/_elf.dir/pybind_module.cc.o
elf/_elf.cpython-37m-x86_64-linux-gnu.so: elf/CMakeFiles/_elf.dir/build.make
elf/_elf.cpython-37m-x86_64-linux-gnu.so: elf/libelf.a
elf/_elf.cpython-37m-x86_64-linux-gnu.so: /root/miniconda3/envs/elf/lib/libpython3.7m.so
elf/_elf.cpython-37m-x86_64-linux-gnu.so: third_party/tbb/tbb_cmake_build_subdir_release/libtbb.so.2
elf/_elf.cpython-37m-x86_64-linux-gnu.so: elf/CMakeFiles/_elf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module _elf.cpython-37m-x86_64-linux-gnu.so"
	cd /Go/ELF/build/elf && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/_elf.dir/link.txt --verbose=$(VERBOSE)
	cd /Go/ELF/build/elf && /usr/bin/strip /Go/ELF/build/elf/_elf.cpython-37m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
elf/CMakeFiles/_elf.dir/build: elf/_elf.cpython-37m-x86_64-linux-gnu.so

.PHONY : elf/CMakeFiles/_elf.dir/build

elf/CMakeFiles/_elf.dir/requires: elf/CMakeFiles/_elf.dir/pybind_module.cc.o.requires

.PHONY : elf/CMakeFiles/_elf.dir/requires

elf/CMakeFiles/_elf.dir/clean:
	cd /Go/ELF/build/elf && $(CMAKE_COMMAND) -P CMakeFiles/_elf.dir/cmake_clean.cmake
.PHONY : elf/CMakeFiles/_elf.dir/clean

elf/CMakeFiles/_elf.dir/depend:
	cd /Go/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Go/ELF /Go/ELF/src_cpp/elf /Go/ELF/build /Go/ELF/build/elf /Go/ELF/build/elf/CMakeFiles/_elf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : elf/CMakeFiles/_elf.dir/depend

