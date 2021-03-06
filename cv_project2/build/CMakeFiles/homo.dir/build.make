# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/ksu/Documents/TestCode/opencv/cv_project2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ksu/Documents/TestCode/opencv/cv_project2/build

# Include any dependencies generated for this target.
include CMakeFiles/homo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/homo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/homo.dir/flags.make

CMakeFiles/homo.dir/src/project2_homo.o: CMakeFiles/homo.dir/flags.make
CMakeFiles/homo.dir/src/project2_homo.o: ../src/project2_homo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ksu/Documents/TestCode/opencv/cv_project2/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/homo.dir/src/project2_homo.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/homo.dir/src/project2_homo.o -c /home/ksu/Documents/TestCode/opencv/cv_project2/src/project2_homo.cpp

CMakeFiles/homo.dir/src/project2_homo.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/homo.dir/src/project2_homo.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ksu/Documents/TestCode/opencv/cv_project2/src/project2_homo.cpp > CMakeFiles/homo.dir/src/project2_homo.i

CMakeFiles/homo.dir/src/project2_homo.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/homo.dir/src/project2_homo.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ksu/Documents/TestCode/opencv/cv_project2/src/project2_homo.cpp -o CMakeFiles/homo.dir/src/project2_homo.s

CMakeFiles/homo.dir/src/project2_homo.o.requires:
.PHONY : CMakeFiles/homo.dir/src/project2_homo.o.requires

CMakeFiles/homo.dir/src/project2_homo.o.provides: CMakeFiles/homo.dir/src/project2_homo.o.requires
	$(MAKE) -f CMakeFiles/homo.dir/build.make CMakeFiles/homo.dir/src/project2_homo.o.provides.build
.PHONY : CMakeFiles/homo.dir/src/project2_homo.o.provides

CMakeFiles/homo.dir/src/project2_homo.o.provides.build: CMakeFiles/homo.dir/src/project2_homo.o

# Object files for target homo
homo_OBJECTS = \
"CMakeFiles/homo.dir/src/project2_homo.o"

# External object files for target homo
homo_EXTERNAL_OBJECTS =

../bin/homo: CMakeFiles/homo.dir/src/project2_homo.o
../bin/homo: CMakeFiles/homo.dir/build.make
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_nonfree.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
../bin/homo: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
../bin/homo: CMakeFiles/homo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/homo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/homo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/homo.dir/build: ../bin/homo
.PHONY : CMakeFiles/homo.dir/build

CMakeFiles/homo.dir/requires: CMakeFiles/homo.dir/src/project2_homo.o.requires
.PHONY : CMakeFiles/homo.dir/requires

CMakeFiles/homo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/homo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/homo.dir/clean

CMakeFiles/homo.dir/depend:
	cd /home/ksu/Documents/TestCode/opencv/cv_project2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ksu/Documents/TestCode/opencv/cv_project2 /home/ksu/Documents/TestCode/opencv/cv_project2 /home/ksu/Documents/TestCode/opencv/cv_project2/build /home/ksu/Documents/TestCode/opencv/cv_project2/build /home/ksu/Documents/TestCode/opencv/cv_project2/build/CMakeFiles/homo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/homo.dir/depend

