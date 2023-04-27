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

#include "OpenGLDrawingFunctions.h"
#include "NATUtils.h"

const float OpenGLDrawingFunctions::X1 = .5257311F;
const float OpenGLDrawingFunctions::Z1 = .8506508F;

const float OpenGLDrawingFunctions::vdata[12][3] = {    
  {-X1, 0.0, Z1}, {X1, 0.0, Z1}, {-X1, 0.0, -Z1}, {X1, 0.0, -Z1},    
  {0.0, Z1, X1}, {0.0, Z1, -X1}, {0.0, -Z1, X1}, {0.0, -Z1, -X1},    
  {Z1, X1, 0.0}, {-Z1, X1, 0.0}, {Z1, -X1, 0.0}, {-Z1, -X1, 0.0} };

const int OpenGLDrawingFunctions::tindices[20][3] = { 
  {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},    
  {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},    
  {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6}, 
  {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };


void OpenGLDrawingFunctions::DrawTriangle(const GLfloat *a, const GLfloat *b, const GLfloat *c, int div, float r)
{
  if (div<=0) 
  {
    glNormal3fv(a); glVertex3f(a[0]*r, a[1]*r, a[2]*r);
    glNormal3fv(b); glVertex3f(b[0]*r, b[1]*r, b[2]*r);
    glNormal3fv(c); glVertex3f(c[0]*r, c[1]*r, c[2]*r);
  } 
  else 
  {
    GLfloat ab[3], ac[3], bc[3];
    for (int i=0;i<3;i++) 
    {
      ab[i]=(a[i]+b[i])/2;
      ac[i]=(a[i]+c[i])/2;
      bc[i]=(b[i]+c[i])/2;
    }
    Normalize(ab); Normalize(ac); Normalize(bc);
    DrawTriangle(a, ab, ac, div-1, r);
    DrawTriangle(b, bc, ab, div-1, r);
    DrawTriangle(c, ac, bc, div-1, r);
    DrawTriangle(ab, bc, ac, div-1, r);  
  }  
}

void OpenGLDrawingFunctions::DrawSphere(int ndiv, float radius) {
  glBegin(GL_TRIANGLES);
  for (int i=0;i<20;i++)
    DrawTriangle(vdata[tindices[i][0]], vdata[tindices[i][1]], vdata[tindices[i][2]], ndiv, radius);
  glEnd();
}

void OpenGLDrawingFunctions::DrawBox(GLfloat x, GLfloat y, GLfloat z, GLfloat qx, GLfloat qy, GLfloat qz, GLfloat qw)
{
  GLfloat s = 50;
  GLfloat m[9], q[] = {qx,qy,qz,qw};
  NATUtils::QaternionToRotationMatrix(q, m);
  GLfloat p1[] = {s,s,s}, p2[] = {-s,s,s}, p3[] = {-s,-s,s}, p4[] = {s,-s,s};
  GLfloat p5[] = {s,s,-s}, p6[] = {-s,s,-s}, p7[] = {-s,-s,-s}, p8[] = {s,-s,-s};
  NATUtils::Vec3MatrixMult(p1, m);
  NATUtils::Vec3MatrixMult(p2, m);
  NATUtils::Vec3MatrixMult(p3, m);
  NATUtils::Vec3MatrixMult(p4, m);
  NATUtils::Vec3MatrixMult(p5, m);
  NATUtils::Vec3MatrixMult(p6, m);
  NATUtils::Vec3MatrixMult(p7, m);
  NATUtils::Vec3MatrixMult(p8, m);
  p1[0] += x; p1[1] += y; p1[2] += z;
  p2[0] += x; p2[1] += y; p2[2] += z;
  p3[0] += x; p3[1] += y; p3[2] += z;
  p4[0] += x; p4[1] += y; p4[2] += z;
  p5[0] += x; p5[1] += y; p5[2] += z;
  p6[0] += x; p6[1] += y; p6[2] += z;
  p7[0] += x; p7[1] += y; p7[2] += z;
  p8[0] += x; p8[1] += y; p8[2] += z;

  GLfloat n[3];

  glBegin(GL_POLYGON);
  n[0] = 0; n[1] = 0; n[2] = 1;
  NATUtils::Vec3MatrixMult(n, m);
  glNormal3f(n[0], n[1], n[2]);
  glVertex3f(p1[0],p1[1],p1[2]);
  glVertex3f(p2[0],p2[1],p2[2]);
  glVertex3f(p3[0],p3[1],p3[2]);
  glVertex3f(p4[0],p4[1],p4[2]);
  glEnd();

  glBegin(GL_POLYGON);
  n[0] = 0; n[1] = 0; n[2] = -1;
  NATUtils::Vec3MatrixMult(n, m);
  glNormal3f(n[0], n[1], n[2]);
  glVertex3f(p5[0],p5[1],p5[2]);
  glVertex3f(p8[0],p8[1],p8[2]);
  glVertex3f(p7[0],p7[1],p7[2]);
  glVertex3f(p6[0],p6[1],p6[2]);
  glEnd();

  glBegin(GL_POLYGON);
  n[0] = -1; n[1] = 0; n[2] = 0;
  NATUtils::Vec3MatrixMult(n, m);
  glNormal3f(n[0], n[1], n[2]);
  glVertex3f(p2[0],p2[1],p2[2]);
  glVertex3f(p6[0],p6[1],p6[2]);
  glVertex3f(p7[0],p7[1],p7[2]);
  glVertex3f(p3[0],p3[1],p3[2]);
  glEnd();

  glBegin(GL_POLYGON);
  n[0] = 1; n[1] = 0; n[2] = 0;
  NATUtils::Vec3MatrixMult(n, m);
  glNormal3f(n[0], n[1], n[2]);
  glVertex3f(p1[0],p1[1],p1[2]);
  glVertex3f(p4[0],p4[1],p4[2]);
  glVertex3f(p8[0],p8[1],p8[2]);
  glVertex3f(p5[0],p5[1],p5[2]);
  glEnd();

  glBegin(GL_POLYGON);
  n[0] = 0; n[1] = 1; n[2] = 0;
  NATUtils::Vec3MatrixMult(n, m);
  glNormal3f(n[0], n[1], n[2]);
  glVertex3f(p6[0],p6[1],p6[2]);
  glVertex3f(p2[0],p2[1],p2[2]);
  glVertex3f(p1[0],p1[1],p1[2]);
  glVertex3f(p5[0],p5[1],p5[2]);
  glEnd();

  glBegin(GL_POLYGON);
  n[0] = 0; n[1] = -1; n[2] = 0;
  NATUtils::Vec3MatrixMult(n, m);
  glNormal3f(n[0], n[1], n[2]);
  glVertex3f(p7[0],p7[1],p7[2]);
  glVertex3f(p8[0],p8[1],p8[2]);
  glVertex3f(p4[0],p4[1],p4[2]);
  glVertex3f(p3[0],p3[1],p3[2]);
  glEnd();
}

void OpenGLDrawingFunctions::DrawCube(float scale)
{
    const float sizex = 0.4f * scale;
    const float sizey = 0.4f * scale;
    const float sizez = 0.4f * scale;

    glBegin(GL_QUADS);

    // FRONT
    glColor3f(0.0f, 0.5294f, 1.0f);
    glNormal3f(0.0f, 0.5294f, 1.0f);
    glVertex3f(-sizex, -sizey, sizez);
    glVertex3f(sizex, -sizey, sizez);
    glVertex3f(sizex, sizey, sizez);
    glVertex3f(-sizex, sizey, sizez);

    // BACK
    glNormal3f(0.0f, 0.5294f, 1.0f);
    glVertex3f(-sizex, -sizey, -sizez);
    glVertex3f(-sizex, sizey, -sizez);
    glVertex3f(sizex, sizey, -sizez);
    glVertex3f(sizex, -sizey, -sizez);


    // LEFT
    glColor3f(1.0f, 0.0f, 0.0f);
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(-sizex, -sizey, sizez);
    glVertex3f(-sizex, sizey, sizez);
    glVertex3f(-sizex, sizey, -sizez);
    glVertex3f(-sizex, -sizey, -sizez);

    // RIGHT
    glNormal3f(1.0f, 0.0f, 0.0f);
    glVertex3f(sizex, -sizey, -sizez);
    glVertex3f(sizex, sizey, -sizez);
    glVertex3f(sizex, sizey, sizez);
    glVertex3f(sizex, -sizey, sizez);


    // TOP
    glColor3f(0.0f, 1.0f, 0.0f);
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-sizex, sizey, sizez);
    glVertex3f(sizex, sizey, sizez);
    glVertex3f(sizex, sizey, -sizez);
    glVertex3f(-sizex, sizey, -sizez);

    // BOTTOM
    glNormal3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-sizex, -sizey, sizez);
    glVertex3f(-sizex, -sizey, -sizez);
    glVertex3f(sizex, -sizey, -sizez);
    glVertex3f(sizex, -sizey, sizez);

    glEnd();

}

void OpenGLDrawingFunctions::DrawGrid()
{
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glPushMatrix();

  float halfSize = 2000.0f;      // world is in mms - set to 2 cubic meters
  float step = 100.0f;           // line every .1 meter
  float major = 200.0f;          // major every .2 meters
  float yloc = 0.0f; 

  glEnable (GL_LINE_STIPPLE);
  glLineWidth (0.25);
  glDepthMask(true);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_ALWAYS);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE);
  glEnable(GL_COLOR_MATERIAL);

  float r,g,b,a;
  r = g = b = a = 0.7f;

  for(float x=-halfSize; x<=halfSize; x+=step)
  {
    if( (x==0) || (x==-halfSize) || (x==halfSize) )         // edge or center line
    {
      glColor4f(.76f*r,.76f*g,.76f*b,.76f*a);         
    }
    else
    {
      float ff = fmod(x,major);                           
      if(ff==0.0f)                                        // major line
      {
        glColor4f(.55f*r,0.55f*g,0.55f*b,0.55f*a);  
      }
      else                                                // minor line
      {
        glColor4f(0.3f*r,0.3f*g,0.3f*b,0.3f*a);     
      }
    }

    glBegin(GL_LINES);				
    glVertex3f(x, 0, halfSize);	    // vertical
    glVertex3f(x, 0, -halfSize);
    glVertex3f(halfSize, 0, x);     // horizontal
    glVertex3f(-halfSize, 0, x);
    glEnd();						

  }    

  glPopAttrib();
  glPopMatrix();

}

