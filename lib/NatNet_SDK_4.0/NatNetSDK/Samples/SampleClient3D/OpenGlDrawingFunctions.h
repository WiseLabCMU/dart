/* 
Copyright © 2014 NaturalPoint Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


#ifndef _OPEN_GL_DRAWING_FUNCTIONS_H_
#define _OPEN_GL_DRAWING_FUNCTIONS_H_

#include <Windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <cmath>

//////////////////////////////////////////////////////////////////////////
/// Functions for simple drawing into the current OpenGL rendering context.
//////////////////////////////////////////////////////////////////////////
class OpenGLDrawingFunctions
{
public:
  // Constants used by the drawing functions.
  static const float X1;
  static const float Z1;
  static const float vdata[12][3];
  static const int tindices[20][3];

  //////////////////////////////////////////////////////////////////////////
  /// <summary>Normalizes a 3-vector to unit length.</summary>
  //////////////////////////////////////////////////////////////////////////
  static void Normalize(GLfloat *a) 
  {
    GLfloat d = (1.0F / std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]));
    a[0] *= d; a[1] *= d; a[2] *= d;
  }

  static void DrawTriangle(const GLfloat *a, const GLfloat *b, const GLfloat *c, int div, float r);

  static void DrawSphere(int ndiv, float radius=1.0);

  static void DrawBox(GLfloat x, GLfloat y, GLfloat z, GLfloat qx, GLfloat qy, GLfloat qz, GLfloat qw);

  static void DrawGrid();

  static void DrawCube(float scale);

};

#endif // _OPEN_GL_DRAWING_FUNCTIONS_H_