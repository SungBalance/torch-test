# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

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
CMAKE_COMMAND = /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Bootstrap.cmk/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cheezestick/workspace/torch-memory/cmake-3.21.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cheezestick/workspace/torch-memory/cmake-3.21.2

# Include any dependencies generated for this target.
include Tests/CMakeLib/CMakeFiles/testEncoding.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Tests/CMakeLib/CMakeFiles/testEncoding.dir/compiler_depend.make

# Include the progress variables for this target.
include Tests/CMakeLib/CMakeFiles/testEncoding.dir/progress.make

# Include the compile flags for this target's objects.
include Tests/CMakeLib/CMakeFiles/testEncoding.dir/flags.make

Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.o: Tests/CMakeLib/CMakeFiles/testEncoding.dir/flags.make
Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.o: Tests/CMakeLib/testEncoding.cxx
Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.o: Tests/CMakeLib/CMakeFiles/testEncoding.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cheezestick/workspace/torch-memory/cmake-3.21.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.o"
	cd /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.o -MF CMakeFiles/testEncoding.dir/testEncoding.cxx.o.d -o CMakeFiles/testEncoding.dir/testEncoding.cxx.o -c /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib/testEncoding.cxx

Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testEncoding.dir/testEncoding.cxx.i"
	cd /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib/testEncoding.cxx > CMakeFiles/testEncoding.dir/testEncoding.cxx.i

Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testEncoding.dir/testEncoding.cxx.s"
	cd /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib/testEncoding.cxx -o CMakeFiles/testEncoding.dir/testEncoding.cxx.s

# Object files for target testEncoding
testEncoding_OBJECTS = \
"CMakeFiles/testEncoding.dir/testEncoding.cxx.o"

# External object files for target testEncoding
testEncoding_EXTERNAL_OBJECTS =

Tests/CMakeLib/testEncoding: Tests/CMakeLib/CMakeFiles/testEncoding.dir/testEncoding.cxx.o
Tests/CMakeLib/testEncoding: Tests/CMakeLib/CMakeFiles/testEncoding.dir/build.make
Tests/CMakeLib/testEncoding: Source/kwsys/libcmsys.a
Tests/CMakeLib/testEncoding: Tests/CMakeLib/CMakeFiles/testEncoding.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cheezestick/workspace/torch-memory/cmake-3.21.2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testEncoding"
	cd /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testEncoding.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Tests/CMakeLib/CMakeFiles/testEncoding.dir/build: Tests/CMakeLib/testEncoding
.PHONY : Tests/CMakeLib/CMakeFiles/testEncoding.dir/build

Tests/CMakeLib/CMakeFiles/testEncoding.dir/clean:
	cd /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib && $(CMAKE_COMMAND) -P CMakeFiles/testEncoding.dir/cmake_clean.cmake
.PHONY : Tests/CMakeLib/CMakeFiles/testEncoding.dir/clean

Tests/CMakeLib/CMakeFiles/testEncoding.dir/depend:
	cd /home/cheezestick/workspace/torch-memory/cmake-3.21.2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cheezestick/workspace/torch-memory/cmake-3.21.2 /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib /home/cheezestick/workspace/torch-memory/cmake-3.21.2 /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib /home/cheezestick/workspace/torch-memory/cmake-3.21.2/Tests/CMakeLib/CMakeFiles/testEncoding.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Tests/CMakeLib/CMakeFiles/testEncoding.dir/depend

