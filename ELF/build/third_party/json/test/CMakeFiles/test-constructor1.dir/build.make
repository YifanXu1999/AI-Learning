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
include third_party/json/test/CMakeFiles/test-constructor1.dir/depend.make

# Include the progress variables for this target.
include third_party/json/test/CMakeFiles/test-constructor1.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/json/test/CMakeFiles/test-constructor1.dir/flags.make

third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o: third_party/json/test/CMakeFiles/test-constructor1.dir/flags.make
third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o: ../third_party/json/test/src/unit-constructor1.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o"
	cd /Go/ELF/build/third_party/json/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o -c /Go/ELF/third_party/json/test/src/unit-constructor1.cpp

third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.i"
	cd /Go/ELF/build/third_party/json/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Go/ELF/third_party/json/test/src/unit-constructor1.cpp > CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.i

third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.s"
	cd /Go/ELF/build/third_party/json/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Go/ELF/third_party/json/test/src/unit-constructor1.cpp -o CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.s

third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.requires:

.PHONY : third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.requires

third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.provides: third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.requires
	$(MAKE) -f third_party/json/test/CMakeFiles/test-constructor1.dir/build.make third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.provides.build
.PHONY : third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.provides

third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.provides.build: third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o


# Object files for target test-constructor1
test__constructor1_OBJECTS = \
"CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o"

# External object files for target test-constructor1
test__constructor1_EXTERNAL_OBJECTS = \
"/Go/ELF/build/third_party/json/test/CMakeFiles/catch_main.dir/src/unit.cpp.o"

third_party/json/test/test-constructor1: third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o
third_party/json/test/test-constructor1: third_party/json/test/CMakeFiles/catch_main.dir/src/unit.cpp.o
third_party/json/test/test-constructor1: third_party/json/test/CMakeFiles/test-constructor1.dir/build.make
third_party/json/test/test-constructor1: third_party/json/test/CMakeFiles/test-constructor1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-constructor1"
	cd /Go/ELF/build/third_party/json/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-constructor1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third_party/json/test/CMakeFiles/test-constructor1.dir/build: third_party/json/test/test-constructor1

.PHONY : third_party/json/test/CMakeFiles/test-constructor1.dir/build

third_party/json/test/CMakeFiles/test-constructor1.dir/requires: third_party/json/test/CMakeFiles/test-constructor1.dir/src/unit-constructor1.cpp.o.requires

.PHONY : third_party/json/test/CMakeFiles/test-constructor1.dir/requires

third_party/json/test/CMakeFiles/test-constructor1.dir/clean:
	cd /Go/ELF/build/third_party/json/test && $(CMAKE_COMMAND) -P CMakeFiles/test-constructor1.dir/cmake_clean.cmake
.PHONY : third_party/json/test/CMakeFiles/test-constructor1.dir/clean

third_party/json/test/CMakeFiles/test-constructor1.dir/depend:
	cd /Go/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Go/ELF /Go/ELF/third_party/json/test /Go/ELF/build /Go/ELF/build/third_party/json/test /Go/ELF/build/third_party/json/test/CMakeFiles/test-constructor1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/json/test/CMakeFiles/test-constructor1.dir/depend

