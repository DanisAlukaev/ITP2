# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.16

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.1.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.1.1\bin\cmake\win\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/Plotting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Plotting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Plotting.dir/flags.make

CMakeFiles/Plotting.dir/main.cpp.obj: CMakeFiles/Plotting.dir/flags.make
CMakeFiles/Plotting.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Plotting.dir/main.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Plotting.dir\main.cpp.obj -c "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\main.cpp"

CMakeFiles/Plotting.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Plotting.dir/main.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\main.cpp" > CMakeFiles\Plotting.dir\main.cpp.i

CMakeFiles/Plotting.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Plotting.dir/main.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\main.cpp" -o CMakeFiles\Plotting.dir\main.cpp.s

# Object files for target Plotting
Plotting_OBJECTS = \
"CMakeFiles/Plotting.dir/main.cpp.obj"

# External object files for target Plotting
Plotting_EXTERNAL_OBJECTS =

Plotting.exe: CMakeFiles/Plotting.dir/main.cpp.obj
Plotting.exe: CMakeFiles/Plotting.dir/build.make
Plotting.exe: CMakeFiles/Plotting.dir/linklibs.rsp
Plotting.exe: CMakeFiles/Plotting.dir/objects1.rsp
Plotting.exe: CMakeFiles/Plotting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Plotting.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Plotting.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Plotting.dir/build: Plotting.exe

.PHONY : CMakeFiles/Plotting.dir/build

CMakeFiles/Plotting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Plotting.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Plotting.dir/clean

CMakeFiles/Plotting.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\cmake-build-debug" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\cmake-build-debug" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 2\Plotting\cmake-build-debug\CMakeFiles\Plotting.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Plotting.dir/depend

