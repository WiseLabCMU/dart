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

#include <stdio.h>
#include <stdarg.h>

#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>

#include "GLPrint.h"


void GLPrint::SetDeviceContext(HDC hDC)
{
    HFONT	glFont;										
    HFONT	oldfont;
    LOGFONT logicalFont;

    // Display list for the 96 ascii characters (95 printable plus del)
    int numAsciiChars = 96;
    m_base = glGenLists(numAsciiChars);

    logicalFont.lfHeight          = -16;     
    logicalFont.lfWidth           = 0; 
    logicalFont.lfEscapement      = 0; 
    logicalFont.lfOrientation     = 0; 
    logicalFont.lfWeight          = FW_BLACK; //FW_NORMAL; 
    logicalFont.lfItalic          = FALSE; 
    logicalFont.lfUnderline       = FALSE; 
    logicalFont.lfStrikeOut       = FALSE; 
    logicalFont.lfCharSet         = ANSI_CHARSET; 
    logicalFont.lfOutPrecision    = OUT_TT_PRECIS; 
    logicalFont.lfClipPrecision   = CLIP_DEFAULT_PRECIS; 
    logicalFont.lfQuality         = ANTIALIASED_QUALITY; 
    logicalFont.lfPitchAndFamily  = FF_DONTCARE | DEFAULT_PITCH; 
    lstrcpy(logicalFont.lfFaceName, TEXT("Arial"));

    glFont = CreateFontIndirect( &logicalFont );
    oldfont = (HFONT)SelectObject(hDC, glFont);           

    /*
    wgl bitmap fonts
    - pre-rendered into display list
    - no rotation
    - scale independent
    */
    BOOL bSuccess = wglUseFontBitmaps(hDC, 32, numAsciiChars, m_base);

    // we're done with the font object so release it
    SelectObject(hDC, oldfont);
    DeleteObject(glFont);
}

void GLPrint::Print(double x, double y, const char *format, ...)
{
  char		text[256];								
  va_list		ap;										
  if (format == nullptr)									
    return;											

  // parse formatted string/args into text string
  va_start(ap, format);									
  vsprintf_s(text, format, ap);					
  va_end(ap);											

  // wgl text
  glPushMatrix();
  glTranslated(x,y,0.0f);
  glRasterPos2d(0.0,0.0);
  // draw the text
  glListBase(m_base - 32);								
  glCallLists((GLsizei)strlen(text), GL_UNSIGNED_BYTE, text);	
  glPopMatrix();
}

