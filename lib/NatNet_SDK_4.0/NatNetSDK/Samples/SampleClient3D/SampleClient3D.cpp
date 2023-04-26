/* 
Copyright © 2012 NaturalPoint Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */


// NatNetSample.cpp : Defines the entry point for the application.
//
#ifdef WIN32
#  define _CRT_SECURE_NO_WARNINGS
#  define WIN32_LEAN_AND_MEAN // Exclude rarely-used stuff from Windows headers
#endif

#include <cstring> // For memset.
#include <windows.h>
#include <winsock.h>
#include "resource.h"

#include <GL/gl.h>
#include <GL/glu.h>

//NatNet SDK
#include "NatNetTypes.h"
#include "NatNetCAPI.h"
#include "NatNetClient.h"
#include "natutils.h"

#include "GLPrint.h"
#include "RigidBodyCollection.h"
#include "MarkerPositionCollection.h"
#include "OpenGLDrawingFunctions.h"

#include <map>
#include <string>

#include <math.h>

#define ID_RENDERTIMER 101

#define MATH_PI 3.14159265F

// globals
// Class for printing bitmap fonts in OpenGL
GLPrint glPrinter;


HINSTANCE hInst;

// OpenGL rendering context.
HGLRC openGLRenderContext = nullptr;

// Our NatNet client object.
NatNetClient natnetClient;

// Objects for saving off marker and rigid body data streamed
// from NatNet.
MarkerPositionCollection markerPositions;
RigidBodyCollection rigidBodies;

std::map<int, std::string> mapIDToName;

// Ready to render?
bool render = true;

// Show rigidbody info
bool showText = true;

// Used for converting NatNet data to the proper units.
float unitConversion = 1.0f;

// World Up Axis (default to Y)
int upAxis = 1; // 

// NatNet server IP address.
int IPAddress[4] = { 127, 0, 0, 1 };

// Timecode string 
char szTimecode[128] = "";

// Initial Eye position and rotation
float g_fEyeX = 0, g_fEyeY = 1, g_fEyeZ = 5;
float g_fRotY = 0;
float g_fRotX = 0;


// functions
// Win32
BOOL InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK NatNetDlgProc(HWND, UINT, WPARAM, LPARAM);
// OpenGL
void RenderOGLScene();
void Update(HWND hWnd);
// NatNet
void NATNET_CALLCONV DataHandler(sFrameOfMocapData* data, void* pUserData);    // receives data from the server
void NATNET_CALLCONV MessageHandler(Verbosity msgType, const char* msg);      // receives NatNet error messages
bool InitNatNet(LPSTR szIPAddress, LPSTR szServerIPAddress, ConnectionType connType);
bool ParseRigidBodyDescription(sDataDescriptions* pDataDefs);

//****************************************************************************
//
// Windows Functions 
//

// Register our window.
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wcex.lpfnWndProc = (WNDPROC)WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = NULL;
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCE(IDC_NATNETSAMPLE);
    wcex.lpszClassName = "NATNETSAMPLE";
    wcex.hIconSm = NULL;

    return RegisterClassEx(&wcex);
}

// WinMain
int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
    LPSTR lpCmdLine, int nCmdShow)
{
    MyRegisterClass(hInstance);

    if (!InitInstance(hInstance, nCmdShow))
        return false;

    MSG msg;
    while (true)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE))
        {
            if (!GetMessage(&msg, NULL, 0, 0))
                break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            if (render)
                Update(msg.hwnd);
        }
    }

    return (int)msg.wParam;
}

// Initialize new instances of our application
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    hInst = hInstance;

    HWND hWnd = CreateWindow("NATNETSAMPLE", "SampleClient 3D", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);

    if (!hWnd)
        return false;

    // Define pixel format
    PIXELFORMATDESCRIPTOR pfd;
    int nPixelFormat;
    memset(&pfd, NULL, sizeof(pfd));
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 16;
    pfd.iLayerType = PFD_MAIN_PLANE;

    // Set pixel format. Needed for drawing OpenGL bitmap fonts.
    HDC hDC = GetDC(hWnd);
    nPixelFormat = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC, nPixelFormat, &pfd);

    // Create and set the current OpenGL rendering context.
    openGLRenderContext = wglCreateContext(hDC);
    wglMakeCurrent(hDC, openGLRenderContext);

    // Set some OpenGL options.
    glClearColor(0.400f, 0.400f, 0.400f, 1.0f);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);

    // Set the device context for our OpenGL printer object.
    glPrinter.SetDeviceContext(hDC);

    wglMakeCurrent(0, 0);
    ReleaseDC(hWnd, hDC);

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // Make a good guess as to the IP address of our NatNet server.
    in_addr MyAddress[10];
    int nAddresses = NATUtils::GetLocalIPAddresses((unsigned long *)&MyAddress, 10);
    if (nAddresses > 0)
    {
        IPAddress[0] = MyAddress[0].S_un.S_un_b.s_b1;
        IPAddress[1] = MyAddress[0].S_un.S_un_b.s_b2;
        IPAddress[2] = MyAddress[0].S_un.S_un_b.s_b3;
        IPAddress[3] = MyAddress[0].S_un.S_un_b.s_b4;
    }

    // schedule to render on UI thread every 30 milliseconds
    UINT renderTimer = SetTimer(hWnd, ID_RENDERTIMER, 30, NULL);

    return true;
}


// Windows message processing function.
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    int wmId, wmEvent;
    PAINTSTRUCT ps;
    HDC hdc;

    switch (message)
    {
    case WM_COMMAND:
        wmId = LOWORD(wParam);
        wmEvent = HIWORD(wParam);
        // Parse the menu selections:
        switch (wmId)
        {
        case IDM_CONNECT:
            DialogBox(hInst, (LPCTSTR)IDD_NATNET, hWnd, (DLGPROC)NatNetDlgProc);
            break;
        case IDM_EXIT:
            DestroyWindow(hWnd);
            break;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        break;

    case WM_TIMER:
        if (wParam == ID_RENDERTIMER)
            Update(hWnd);
        break;

    case WM_KEYDOWN:
    {
        bool bShift = (GetKeyState(VK_SHIFT) & 0x80) != 0;
        bool bCtrl = (GetKeyState(VK_CONTROL) & 0x80) != 0;
        switch (wParam)
        {
        case VK_UP:
            if (bCtrl)
                g_fRotX += 1;
            else if (bShift)
                g_fEyeY += 0.03f;
            else
                g_fEyeZ -= 0.03f;
            break;
        case VK_DOWN:
            if (bCtrl)
                g_fRotX -= 1;
            else if (bShift)
                g_fEyeY -= 0.03f;
            else
                g_fEyeZ += 0.03f;
            break;
        case VK_LEFT:
            if (bCtrl)
                g_fRotY += 1;
            else
                g_fEyeX -= 0.03f;
            break;
        case VK_RIGHT:
            if (bCtrl)
                g_fRotY -= 1;
            else
                g_fEyeX += 0.03f;
            break;
        case 'T':
        case 't':
            showText = !showText;
            break;
        }
        InvalidateRect(hWnd, NULL, TRUE);
    }
        break;

    case WM_PAINT:
        hdc = BeginPaint(hWnd, &ps);
        Update(hWnd);
        EndPaint(hWnd, &ps);
        break;

    case WM_SIZE:
    {
        int cx = LOWORD(lParam), cy = HIWORD(lParam);
        if (cx != 0 && cy != 0 && hWnd != nullptr)
        {
            GLfloat fFovy = 40.0f; // Field-of-view
            GLfloat fZNear = 1.0f;  // Near clipping plane
            GLfloat fZFar = 10000.0f;  // Far clipping plane

            HDC hDC = GetDC(hWnd);
            wglMakeCurrent(hDC, openGLRenderContext);

            // Calculate OpenGL viewport aspect
            RECT rv;
            GetClientRect(hWnd, &rv);
            GLfloat fAspect = (GLfloat)(rv.right - rv.left) / (GLfloat)(rv.bottom - rv.top);

            // Define OpenGL viewport
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(fFovy, fAspect, fZNear, fZFar);
            glViewport(rv.left, rv.top, rv.right - rv.left, rv.bottom - rv.top);
            glMatrixMode(GL_MODELVIEW);

            Update(hWnd);

            wglMakeCurrent(0, 0);
            ReleaseDC(hWnd, hDC);
        }
    }
        break;

    case WM_DESTROY:
    {
        HDC hDC = GetDC(hWnd);
        wglMakeCurrent(hDC, openGLRenderContext);
        natnetClient.Disconnect();
        wglMakeCurrent(0, 0);
        wglDeleteContext(openGLRenderContext);
        ReleaseDC(hWnd, hDC);
        PostQuitMessage(0);
    }
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    return 0;
}

// Update OGL window
void Update(HWND hwnd)
{
    HDC hDC = GetDC(hwnd);
    if (hDC)
    {
        wglMakeCurrent(hDC, openGLRenderContext);
        RenderOGLScene();
        SwapBuffers(hDC);
        wglMakeCurrent(0, 0);
    }
    ReleaseDC(hwnd, hDC);
}

void ConvertRHSPosZupToYUp(float& x, float& y, float& z)
{
    /*
    [RHS, Y-Up]     [RHS, Z-Up]

                          Y
     Y                 Z /
     |__ X             |/__ X
     /
    Z

    Xyup  =  Xzup
    Yyup  =  Zzup
    Zyup  =  -Yzup
    */
    float yOriginal = y;
    y = z;
    z = -yOriginal;
}

void ConvertRHSRotZUpToYUp(float& qx, float& qy, float& qz, float& qw)
{
    // -90 deg rotation about +X
    float qRx, qRy, qRz, qRw;
    float angle = -90.0f * MATH_PI / 180.0f;
    qRx = sin(angle / 2.0f);
    qRy = 0.0f;
    qRz = 0.0f;
    qRw = cos(angle / 2.0f);

    // rotate quat using quat multiply
    float qxNew, qyNew, qzNew, qwNew;
    qxNew = qw*qRx + qx*qRw + qy*qRz - qz*qRy;
    qyNew = qw*qRy - qx*qRz + qy*qRw + qz*qRx;
    qzNew = qw*qRz + qx*qRy - qy*qRx + qz*qRw;
    qwNew = qw*qRw - qx*qRx - qy*qRy - qz*qRz;

    qx = qxNew;
    qy = qyNew;
    qz = qzNew;
    qw = qwNew;
}

// Render OpenGL scene
void RenderOGLScene()
{
    GLfloat m[9];
    GLfloat v[3];
    float fRadius = 5.0f;

    // Setup OpenGL viewport
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear buffers
    glClearColor( 0.098f, 0.098f, 0.098f, 1.0f );
    glLoadIdentity(); // Load identity matrix
    GLfloat glfLightAmb[] = { .3f, .3f, .3f, 1.0f };
    GLfloat glfLightPos[] = { -4.0f, 4.0f, 4.0f, 0.5f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, glfLightAmb);
    glLightfv(GL_LIGHT1, GL_POSITION, glfLightPos);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    glPushMatrix();


    // Draw timecode
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glPushMatrix();
    glTranslatef(2400.f, -1750.f, -5000.0f);
    glPrinter.Print(0.0f, 0.0f, szTimecode);
    glPopMatrix();

    // Position and rotate the camera
    glTranslatef(g_fEyeX * -1000, g_fEyeY * -1000, g_fEyeZ * -1000);
    glRotatef(g_fRotY, 0, 1, 0);
    glRotatef(g_fRotX, 1, 0, 0);
    

    // Draw reference axis triad
    // x
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(300, 0, 0);
    // y
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 300, 0);
    // z
    glColor3f(0.0f, 0.5294f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 300);
    glEnd();

    // Draw grid
    glLineWidth(1.0f);
    OpenGLDrawingFunctions::DrawGrid();

    // Draw rigid bodies
    float textX = -3200.0f;
    float textY = 2700.0f;
    GLfloat x, y, z;
    Quat q;
    EulerAngles ea;
    int order;

    for (size_t i = 0; i < rigidBodies.Count(); i++)
    {
        // RigidBody position
        std::tie(x, y, z) = rigidBodies.GetCoordinates(i);
        // convert to millimeters
        x *= unitConversion;
        y *= unitConversion;
        z *= unitConversion;

        // RigidBody orientation
        GLfloat qx, qy, qz, qw;
        std::tie(qx, qy, qz, qw) = rigidBodies.GetQuaternion(i);
        q.x = qx;
        q.y = qy;
        q.z = qz;
        q.w = qw;

        // If Motive is streaming Z-up, convert to this renderer's Y-up coordinate system
        if (upAxis==2)
        {
            // convert position
            ConvertRHSPosZupToYUp(x, y, z);
            // convert orientation
            ConvertRHSRotZUpToYUp(q.x, q.y, q.z, q.w);
        }

        // Convert Motive quaternion output to Euler angles
        // Motive coordinate conventions : X(Pitch), Y(Yaw), Z(Roll), Relative, RHS
        order = EulOrdXYZr;
        ea = Eul_FromQuat(q, order);

        ea.x = NATUtils::RadiansToDegrees(ea.x);
        ea.y = NATUtils::RadiansToDegrees(ea.y);
        ea.z = NATUtils::RadiansToDegrees(ea.z);

        // Draw RigidBody as cube
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushMatrix();

        glTranslatef(x, y, z);

        // source is Y-Up (default)
        glRotatef(ea.x, 1.0f, 0.0f, 0.0f);
        glRotatef(ea.y, 0.0f, 1.0f, 0.0f);
        glRotatef(ea.z, 0.0f, 0.0f, 1.0f);

        /*
        // alternate Z-up conversion - convert only Euler rotation interpretation
        //  Yyup  =  Zzup
        //  Zyup  =  -Yzup
        glRotatef(ea.x, 1.0f, 0.0f, 0.0f);
        glRotatef(ea.y, 0.0f, 0.0f, 1.0f);
        glRotatef(ea.z, 0.0f, -1.0f, 0.0f);
        */

        OpenGLDrawingFunctions::DrawCube(100.0f);
        glPopMatrix();
        glPopAttrib();

        if (showText)
        {
            glColor4f(1.0f, 1.0f, 1.0f, 0.2f);
            std::string rigidBodyName = mapIDToName.at(rigidBodies.ID(i));
            glPrinter.Print(textX, textY, "%s (Pitch: %3.1f, Yaw: %3.1f, Roll: %3.1f)", rigidBodyName.c_str(), ea.x, ea.y, ea.z);
            textY -= 100.0f;
        }

    }

	glPushAttrib(GL_ALL_ATTRIB_BITS);
	for (size_t i = 0; i < markerPositions.LabeledMarkerPositionCount(); i++)
	{
		const sMarker& markerData = markerPositions.GetLabeledMarker(i);

		// Set color dependent on marker params for labeled/unlabeled
		if ((markerData.params & 0x10) != 0) 
			glColor4f(0.8f, 0.4f, 0.0f, 0.8f);
		else
			glColor4f(0.8f, 0.8f, 0.8f, 0.8f);

		v[0] = markerData.x * unitConversion;
		v[1] = markerData.y * unitConversion;
		v[2] = markerData.z * unitConversion;
		fRadius = markerData.size * unitConversion;

		// If Motive is streaming Z-up, convert to this renderer's Y-up coordinate system
		if (upAxis == 2)
		{
			ConvertRHSPosZupToYUp(v[0], v[1], v[2]);
		}

		glPushMatrix();
		glTranslatef(v[0], v[1], v[2]);
		OpenGLDrawingFunctions::DrawSphere(1, fRadius);
		glPopMatrix();

	}
	glPopAttrib();

    // Done rendering a frame. The NatNet callback function DataHandler
    // will set render to true when it receives another frame of data.
    render = false;

}

// Callback for the connect-to-NatNet dialog. Gets the server and local IP 
// addresses and attempts to initialize the NatNet client.
LRESULT CALLBACK NatNetDlgProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    char szBuf[512];
    switch (message)
    {
    case WM_INITDIALOG:
        SetDlgItemText(hDlg, IDC_EDIT1, _itoa(IPAddress[0], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT2, _itoa(IPAddress[1], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT3, _itoa(IPAddress[2], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT4, _itoa(IPAddress[3], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT5, _itoa(IPAddress[0], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT6, _itoa(IPAddress[1], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT7, _itoa(IPAddress[2], szBuf, 10));
        SetDlgItemText(hDlg, IDC_EDIT8, _itoa(IPAddress[3], szBuf, 10));
        SendDlgItemMessage( hDlg, IDC_COMBO_CONNTYPE, CB_ADDSTRING, 0, (LPARAM)TEXT( "Multicast" ) );
        SendDlgItemMessage( hDlg, IDC_COMBO_CONNTYPE, CB_ADDSTRING, 0, (LPARAM)TEXT( "Unicast" ) );
        SendDlgItemMessage( hDlg, IDC_COMBO_CONNTYPE, CB_SETCURSEL, 0, 0 );
        return true;

    case WM_COMMAND:
        switch (LOWORD(wParam))
        {
        case IDC_CONNECT:
        {
            char szMyIPAddress[30], szServerIPAddress[30];
            char ip1[5], ip2[5], ip3[5], ip4[5];
            GetDlgItemText(hDlg, IDC_EDIT1, ip1, 4);
            GetDlgItemText(hDlg, IDC_EDIT2, ip2, 4);
            GetDlgItemText(hDlg, IDC_EDIT3, ip3, 4);
            GetDlgItemText(hDlg, IDC_EDIT4, ip4, 4);
            sprintf_s(szMyIPAddress, 30, "%s.%s.%s.%s", ip1, ip2, ip3, ip4);

            GetDlgItemText(hDlg, IDC_EDIT5, ip1, 4);
            GetDlgItemText(hDlg, IDC_EDIT6, ip2, 4);
            GetDlgItemText(hDlg, IDC_EDIT7, ip3, 4);
            GetDlgItemText(hDlg, IDC_EDIT8, ip4, 4);
            sprintf_s(szServerIPAddress, 30, "%s.%s.%s.%s", ip1, ip2, ip3, ip4);

            const ConnectionType connType = (ConnectionType)SendDlgItemMessage( hDlg, IDC_COMBO_CONNTYPE, CB_GETCURSEL, 0, 0 );

            // Try and initialize the NatNet client.
            if (InitNatNet( szMyIPAddress, szServerIPAddress, connType ) == false)
            {
                natnetClient.Disconnect();
                MessageBox(hDlg, "Failed to connect", "", MB_OK);
            }
        }
        case IDOK:
        case IDCANCEL:
            EndDialog(hDlg, LOWORD(wParam));
            return true;
        }
    }
    return false;
}

// Initialize the NatNet client with client and server IP addresses.
bool InitNatNet( LPSTR szIPAddress, LPSTR szServerIPAddress, ConnectionType connType )
{
    unsigned char ver[4];
    NatNet_GetVersion(ver);

    // Set callback handlers
    // Callback for NatNet messages.
    NatNet_SetLogCallback( MessageHandler );
    // this function will receive data from the server
    natnetClient.SetFrameReceivedCallback(DataHandler);

    sNatNetClientConnectParams connectParams;
    connectParams.connectionType = connType;
    connectParams.localAddress = szIPAddress;
    connectParams.serverAddress = szServerIPAddress;
    int retCode = natnetClient.Connect( connectParams );
    if (retCode != ErrorCode_OK)
    {
        //Unable to connect to server.
        return false;
    }
    else
    {
        // Print server info
        sServerDescription ServerDescription;
        memset(&ServerDescription, 0, sizeof(ServerDescription));
        natnetClient.GetServerDescription(&ServerDescription);
        if (!ServerDescription.HostPresent)
        {
            //Unable to connect to server. Host not present
            return false;
        }
    }

    // Retrieve RigidBody description from server
    sDataDescriptions* pDataDefs = NULL;
    retCode = natnetClient.GetDataDescriptionList(&pDataDefs);
    if (retCode != ErrorCode_OK || ParseRigidBodyDescription(pDataDefs) == false)
    {
        //Unable to retrieve RigidBody description
        //return false;
    }
    NatNet_FreeDescriptions( pDataDefs );
    pDataDefs = NULL;

    // example of NatNet general message passing. Set units to millimeters
    // and get the multiplicative conversion factor in the response.
    void* response;
    int nBytes;
    retCode = natnetClient.SendMessageAndWait("UnitsToMillimeters", &response, &nBytes);
    if (retCode == ErrorCode_OK)
    {
        unitConversion = *(float*)response;
    }

    retCode = natnetClient.SendMessageAndWait("UpAxis", &response, &nBytes);
    if (retCode == ErrorCode_OK)
    {
        upAxis = *(long*)response;
    }

    return true;
}

bool ParseRigidBodyDescription(sDataDescriptions* pDataDefs)
{
    mapIDToName.clear();

    if (pDataDefs == NULL || pDataDefs->nDataDescriptions <= 0)
        return false;

    // preserve a "RigidBody ID to Rigid Body Name" mapping, which we can lookup during data streaming
    int iSkel = 0;
    for (int i = 0, j = 0; i < pDataDefs->nDataDescriptions; i++)
    {
        if (pDataDefs->arrDataDescriptions[i].type == Descriptor_RigidBody)
        {
            sRigidBodyDescription *pRB = pDataDefs->arrDataDescriptions[i].Data.RigidBodyDescription;
            mapIDToName[pRB->ID] = std::string(pRB->szName);
        }
        else if (pDataDefs->arrDataDescriptions[i].type == Descriptor_Skeleton)
        {
            sSkeletonDescription *pSK = pDataDefs->arrDataDescriptions[i].Data.SkeletonDescription;
            for (int i = 0; i < pSK->nRigidBodies; i++)
            {
                // Note: Within FrameOfMocapData, skeleton rigid body ids are of the form:
                //   parent skeleton ID   : high word (upper 16 bits of int)
                //   rigid body id        : low word  (lower 16 bits of int)
                // 
                // However within DataDescriptions they are not, so apply that here for correct lookup during streaming
                int id = pSK->RigidBodies[i].ID | (pSK->skeletonID << 16);
                mapIDToName[id] = std::string(pSK->RigidBodies[i].szName);
            }
            iSkel++;
        }
        else
            continue;
    }

    return true;
}

// [Optional] Handler for NatNet messages. 
void NATNET_CALLCONV MessageHandler(Verbosity msgType, const char* msg)
{
    //	printf("\n[SampleClient] Message received: %s\n", msg);
}

// NatNet data callback function. Stores rigid body and marker data in the file level 
// variables markerPositions, and rigidBodies and sets the file level variable render
// to true. This signals that we have a frame ready to render.
void DataHandler(sFrameOfMocapData* data, void* pUserData)
{
    int mcount = min(MarkerPositionCollection::MAX_MARKER_COUNT, data->MocapData->nMarkers);
    markerPositions.SetMarkerPositions(data->MocapData->Markers, mcount);

    // Marker Data
    markerPositions.SetLabledMarkers(data->LabeledMarkers, data->nLabeledMarkers);

	// nOtherMarkers is deprecated
    // mcount = min(MarkerPositionCollection::MAX_MARKER_COUNT, data->nOtherMarkers);
    // markerPositions.AppendMarkerPositions(data->OtherMarkers, mcount);

    // rigid bodies
    int rbcount = min(RigidBodyCollection::MAX_RIGIDBODY_COUNT, data->nRigidBodies);
    rigidBodies.SetRigidBodyData(data->RigidBodies, rbcount);

    // skeleton segment (bones) as collection of rigid bodies
    for (int s = 0; s < data->nSkeletons; s++)
    {
        rigidBodies.AppendRigidBodyData(data->Skeletons[s].RigidBodyData, data->Skeletons[s].nRigidBodies);
    }

    // timecode
    NatNetClient* pClient = (NatNetClient*)pUserData;
    int hour, minute, second, frame, subframe;
    NatNet_DecodeTimecode( data->Timecode, data->TimecodeSubframe, &hour, &minute, &second, &frame, &subframe );
    // decode timecode into friendly string
    NatNet_TimecodeStringify( data->Timecode, data->TimecodeSubframe, szTimecode, 128 );

    render = true;
}
