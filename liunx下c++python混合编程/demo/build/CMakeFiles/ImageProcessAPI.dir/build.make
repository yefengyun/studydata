# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/yfy/hello

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yfy/hello/build

# Include any dependencies generated for this target.
include CMakeFiles/ImageProcessAPI.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ImageProcessAPI.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ImageProcessAPI.dir/flags.make

CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o: CMakeFiles/ImageProcessAPI.dir/flags.make
CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o: ../ImageProcessAPI.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yfy/hello/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o -c /home/yfy/hello/ImageProcessAPI.cpp

CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yfy/hello/ImageProcessAPI.cpp > CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.i

CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yfy/hello/ImageProcessAPI.cpp -o CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.s

CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.requires:

.PHONY : CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.requires

CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.provides: CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.requires
	$(MAKE) -f CMakeFiles/ImageProcessAPI.dir/build.make CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.provides.build
.PHONY : CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.provides

CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.provides.build: CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o


# Object files for target ImageProcessAPI
ImageProcessAPI_OBJECTS = \
"CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o"

# External object files for target ImageProcessAPI
ImageProcessAPI_EXTERNAL_OBJECTS =

libImageProcessAPI.so: CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o
libImageProcessAPI.so: CMakeFiles/ImageProcessAPI.dir/build.make
libImageProcessAPI.so: /usr/local/lib/libboost_python35.so
libImageProcessAPI.so: /usr/lib/x86_64-linux-gnu/libpython3.5m.so
libImageProcessAPI.so: /usr/local/lib/libopencv_stitching.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_superres.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_videostab.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_hfs.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_stereo.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_surface_matching.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_dpm.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_aruco.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_line_descriptor.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_ccalib.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_face.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_rgbd.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_img_hash.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_freetype.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_bioinspired.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_xphoto.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_structured_light.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_reg.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_fuzzy.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_saliency.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_xfeatures2d.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_bgsegm.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_optflow.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_xobjdetect.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_tracking.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_shape.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_photo.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_ximgproc.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_calib3d.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_objdetect.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_video.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_datasets.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_text.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_ml.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_features2d.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_flann.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_plot.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_highgui.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_dnn.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_videoio.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_imgproc.so.3.4.2
libImageProcessAPI.so: /usr/local/lib/libopencv_core.so.3.4.2
libImageProcessAPI.so: CMakeFiles/ImageProcessAPI.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yfy/hello/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libImageProcessAPI.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ImageProcessAPI.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ImageProcessAPI.dir/build: libImageProcessAPI.so

.PHONY : CMakeFiles/ImageProcessAPI.dir/build

CMakeFiles/ImageProcessAPI.dir/requires: CMakeFiles/ImageProcessAPI.dir/ImageProcessAPI.cpp.o.requires

.PHONY : CMakeFiles/ImageProcessAPI.dir/requires

CMakeFiles/ImageProcessAPI.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ImageProcessAPI.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ImageProcessAPI.dir/clean

CMakeFiles/ImageProcessAPI.dir/depend:
	cd /home/yfy/hello/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yfy/hello /home/yfy/hello /home/yfy/hello/build /home/yfy/hello/build /home/yfy/hello/build/CMakeFiles/ImageProcessAPI.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ImageProcessAPI.dir/depend

