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
include third_party/json/test/CMakeFiles/test-ubjson.dir/depend.make

# Include the progress variables for this target.
include third_party/json/test/CMakeFiles/test-ubjson.dir/progress.make

# Include the compile flags for this target's objects.
include third_party/json/test/CMakeFiles/test-ubjson.dir/flags.make

third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o: third_party/json/test/CMakeFiles/test-ubjson.dir/flags.make
third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o: ../third_party/json/test/src/unit-ubjson.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o"
	cd /Go/ELF/build/third_party/json/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o -c /Go/ELF/third_party/json/test/src/unit-ubjson.cpp

third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.i"
	cd /Go/ELF/build/third_party/json/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Go/ELF/third_party/json/test/src/unit-ubjson.cpp > CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.i

third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.s"
	cd /Go/ELF/build/third_party/json/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Go/ELF/third_party/json/test/src/unit-ubjson.cpp -o CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.s

third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.requires:

.PHONY : third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.requires

third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.provides: third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.requires
	$(MAKE) -f third_party/json/test/CMakeFiles/test-ubjson.dir/build.make third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.provides.build
.PHONY : third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.provides

third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.provides.build: third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o


# Object files for target test-ubjson
test__ubjson_OBJECTS = \
"CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o"

# External object files for target test-ubjson
test__ubjson_EXTERNAL_OBJECTS = \
"/Go/ELF/build/third_party/json/test/CMakeFiles/catch_main.dir/src/unit.cpp.o"

third_party/json/test/test-ubjson: third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o
third_party/json/test/test-ubjson: third_party/json/test/CMakeFiles/catch_main.dir/src/unit.cpp.o
third_party/json/test/test-ubjson: third_party/json/test/CMakeFiles/test-ubjson.dir/build.make
third_party/json/test/test-ubjson: third_party/json/test/CMakeFiles/test-ubjson.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Go/ELF/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test-ubjson"
	cd /Go/ELF/build/third_party/json/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-ubjson.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third_party/json/test/CMakeFiles/test-ubjson.dir/build: third_party/json/test/test-ubjson

.PHONY : third_party/json/test/CMakeFiles/test-ubjson.dir/build

third_party/json/test/CMakeFiles/test-ubjson.dir/requires: third_party/json/test/CMakeFiles/test-ubjson.dir/src/unit-ubjson.cpp.o.requires

.PHONY : third_party/json/test/CMakeFiles/test-ubjson.dir/requires

third_party/json/test/CMakeFiles/test-ubjson.dir/clean:
	cd /Go/ELF/build/third_party/json/test && $(CMAKE_COMMAND) -P CMakeFiles/test-ubjson.dir/cmake_clean.cmake
.PHONY : third_party/json/test/CMakeFiles/test-ubjson.dir/clean

third_party/json/test/CMakeFiles/test-ubjson.dir/depend:
	cd /Go/ELF/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Go/ELF /Go/ELF/third_party/json/test /Go/ELF/build /Go/ELF/build/third_party/json/test /Go/ELF/build/third_party/json/test/CMakeFiles/test-ubjson.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third_party/json/test/CMakeFiles/test-ubjson.dir/depend

