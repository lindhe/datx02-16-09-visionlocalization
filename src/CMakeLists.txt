add_library(apriltags
  CameraUtil.cpp
  DebugImage.cpp
  Geometry.cpp 
  GrayModel.cpp
  MathUtil.cpp 
  Refine.cpp
  TagDetector.cpp 
  TagFamily.cpp
  TagFamilies.cpp
  UnionFindSimple.cpp
)

set(AT_LIBS apriltags ${OPENCV_LDFLAGS})

add_executable(GulliView GulliView.cpp)
target_link_libraries(GulliView ${AT_LIBS})
target_link_libraries(GulliView ${Boost_LIBRARIES})

# -- Client used for testing for UDP transfer. Not needed
# for final program ---
#add_executable(GulliViewClient GulliViewClient.cpp)
#target_link_libraries(GulliViewClient ${Boost_LIBRARIES})

add_executable(GulliViewServer GulliViewServer.cpp)
target_link_libraries(GulliViewServer ${Boost_LIBRARIES})

if (GLUT_LIBRARY)

  add_executable(gltest gltest.cpp)
  target_link_libraries(gltest ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${AT_LIBS})

endif()

if (CAIRO_FOUND)

  add_executable(maketags maketags.cpp)
  target_link_libraries(maketags ${CAIRO_LIBRARIES} ${AT_LIBS} ${CAIRO_LIBS})

endif() 
