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
CMAKE_SOURCE_DIR = "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/Fast_Fourier_Transform.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Fast_Fourier_Transform.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Fast_Fourier_Transform.dir/flags.make

CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.obj: CMakeFiles/Fast_Fourier_Transform.dir/flags.make
CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.obj"
	C:\MinGW\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Fast_Fourier_Transform.dir\main.cpp.obj -c "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\main.cpp"

CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.i"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\main.cpp" > CMakeFiles\Fast_Fourier_Transform.dir\main.cpp.i

CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.s"
	C:\MinGW\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\main.cpp" -o CMakeFiles\Fast_Fourier_Transform.dir\main.cpp.s

# Object files for target Fast_Fourier_Transform
Fast_Fourier_Transform_OBJECTS = \
"CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.obj"

# External object files for target Fast_Fourier_Transform
Fast_Fourier_Transform_EXTERNAL_OBJECTS =

Fast_Fourier_Transform.exe: CMakeFiles/Fast_Fourier_Transform.dir/main.cpp.obj
Fast_Fourier_Transform.exe: CMakeFiles/Fast_Fourier_Transform.dir/build.make
Fast_Fourier_Transform.exe: CMakeFiles/Fast_Fourier_Transform.dir/linklibs.rsp
Fast_Fourier_Transform.exe: CMakeFiles/Fast_Fourier_Transform.dir/objects1.rsp
Fast_Fourier_Transform.exe: CMakeFiles/Fast_Fourier_Transform.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Fast_Fourier_Transform.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Fast_Fourier_Transform.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Fast_Fourier_Transform.dir/build: Fast_Fourier_Transform.exe

.PHONY : CMakeFiles/Fast_Fourier_Transform.dir/build

CMakeFiles/Fast_Fourier_Transform.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Fast_Fourier_Transform.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Fast_Fourier_Transform.dir/clean

CMakeFiles/Fast_Fourier_Transform.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\cmake-build-debug" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\cmake-build-debug" "C:\Users\pc\Documents\Study\2019-2020\ITP II\Assignment 5\Fast Fourier Transform\cmake-build-debug\CMakeFiles\Fast_Fourier_Transform.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Fast_Fourier_Transform.dir/depend

