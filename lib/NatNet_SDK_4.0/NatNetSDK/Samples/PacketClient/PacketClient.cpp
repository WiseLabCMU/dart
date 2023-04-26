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

/*

PacketClient.cpp

Decodes NatNet packets directly.

Usage [optional]:

	PacketClient [ServerIP] [LocalIP]

	[ServerIP]			IP address of server ( defaults to local machine)
	[LocalIP]			IP address of client ( defaults to local machine)

*/
/*
* WSA Error codes:
* https://docs.microsoft.com/en-us/windows/win32/winsock/windows-sockets-error-codes-2
*/

#include <stdio.h>
#include <inttypes.h>
#include <tchar.h>
#include <conio.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <map>
#include <assert.h>
#include <chrono>
#include <thread>
#include <vector>


#pragma warning( disable : 4996 )

#ifdef VDEBUG
#undef VDEBUG
#endif
// #define VDEBUG

#define MAX_NAMELENGTH              256
#define MAX_ANALOG_CHANNELS          32

// NATNET message ids
#define NAT_CONNECT                 0 
#define NAT_SERVERINFO              1
#define NAT_REQUEST                 2
#define NAT_RESPONSE                3
#define NAT_REQUEST_MODELDEF        4
#define NAT_MODELDEF                5
#define NAT_REQUEST_FRAMEOFDATA     6
#define NAT_FRAMEOFDATA             7
#define NAT_MESSAGESTRING           8
#define NAT_DISCONNECT              9
#define NAT_KEEPALIVE               10
#define NAT_UNRECOGNIZED_REQUEST    100
#define UNDEFINED                   999999.9999


#define MAX_PACKETSIZE				100000	// max size of packet (actual packet size is dynamic)

// This should match the multicast address listed in Motive's streaming settings.
#define MULTICAST_ADDRESS		"239.255.42.99"

// Requested size for socket
#define OPTVAL_REQUEST_SIZE 0x10000

// NatNet Command channel
#define PORT_COMMAND            1510

// NatNet Data channel
#define PORT_DATA  			    1511                

SOCKET gCommandSocket;
SOCKET gDataSocket;
//in_addr gServerAddress;
sockaddr_in gHostAddr;

int gNatNetVersion[4] = { 0,0,0,0 };
int gNatNetVersionServer[4] = { 0,0,0,0 };
int gServerVersion[4] = { 0,0,0,0 };
char gServerName[MAX_NAMELENGTH] = { 0 };
bool gCanChangeBitstream = false;

// Compiletime flag for unicast/multicast
//gUseMulticast = true  : Use Multicast
//gUseMulticast = false : Use Unicast
bool gUseMulticast = true;
bool gPausePlayback = false;
int gCommandResponse = 0;
int gCommandResponseSize = 0;
unsigned char gCommandResponseString[MAX_PATH];
int gCommandResponseCode = 0;

typedef struct sParsedArgs
{
    char    szMyIPAddress[128] = "127.0.0.1";
    char    szServerIPAddress[128] = "127.0.0.1";
    in_addr myAddress;
    in_addr serverAddress;

    in_addr multiCastAddress;
    bool    useMulticast = true;
} sParsedArgs;


// sender
typedef struct
{
    char szName[MAX_NAMELENGTH];            // sending app's name
    unsigned char Version[4];               // sending app's version [major.minor.build.revision]
    unsigned char NatNetVersion[4];         // sending app's NatNet version [major.minor.build.revision]

} sSender;

typedef struct
{
    unsigned short iMessage;                // message ID (e.g. NAT_FRAMEOFDATA)
    unsigned short nDataBytes;              // Num bytes in payload
    union
    {
        unsigned char  cData[MAX_PACKETSIZE];
        char           szData[MAX_PACKETSIZE];
        unsigned long  lData[MAX_PACKETSIZE / 4];
        float          fData[MAX_PACKETSIZE / 4];
        sSender        Sender;
    } Data;                                 // Payload incoming from NatNet Server

} sPacket;

typedef struct sConnectionOptions
{
    bool subscribedDataOnly;
    uint8_t BitstreamVersion[4];
#if defined(__cplusplus)
    sConnectionOptions() : subscribedDataOnly(false), BitstreamVersion{ 0,0,0,0 } {}
#endif
} sConnectionOptions;


// Communications functions
bool IPAddress_StringToAddr(char *szNameOrAddress, struct in_addr *Address);
int GetLocalIPAddresses(unsigned long Addresses[], int nMax);
int SendCommand(char * szCOmmand);

// Packet unpacking functions
char * Unpack(char * pPacketIn);
char* UnpackPacketHeader(char* ptr, int& messageID, int& nBytes, int& nBytesTotal);
// Frame data
char * UnpackFrameData(char* inptr, int nBytes, int major, int minor);
char * UnpackFramePrefixData(char * ptr, int major, int minor);
char * UnpackMarkersetData(char * ptr, int major, int minor);
char * UnpackRigidBodyData(char * ptr, int major, int minor);
char * UnpackSkeletonData(char * ptr, int major, int minor);
char * UnpackLabeledMarkerData(char* ptr, int major, int minor);
char * UnpackForcePlateData(char * ptr, int major, int minor);
char * UnpackDeviceData(char * ptr, int major, int minor);
char * UnpackFrameSuffixData(char * ptr, int major, int minor);
// Descriptions
char * UnpackDescription(char * inptr, int nBytes, int major, int minor);
char * UnpackMarkersetDescription(char * ptr,  char* targetPtr,int major, int minor);
char * UnpackRigidBodyDescription(char * ptr,  char* targetPtr,int major, int minor);
char * UnpackSkeletonDescription(char * ptr,  char* targetPtr,int major, int minor);
char * UnpackForcePlateDescription(char * ptr,  char* targetPtr,int major, int minor);
char * UnpackDeviceDescription(char * ptr,  char* targetPtr,int major, int minor);
char * UnpackCameraDescription(char * ptr,  char* targetPtr,int major, int minor);

std::map<int, std::string> wsaErrors = {
    { 10004, " WSAEINTR: Interrupted function call."},
    { 10009, " WSAEBADF: File handle is not valid."},
    { 10013, " WSAEACCESS: Permission denied."},
    { 10014, " WSAEFAULT: Bad address."},
    { 10022, " WSAEINVAL: Invalid argument."},
    { 10024, " WSAEMFILE: Too many open files."},
    { 10035, " WSAEWOULDBLOCK: Resource temporarily unavailable."},
    { 10036, " WSAEINPROGRESS: Operation now in progress."},
    { 10037, " WSAEALREADY: Operation already in progress."},
    { 10038, " WSAENOTSOCK: Socket operation on nonsocket."},
    { 10039, " WSAEDESTADDRREQ Destination address required."},
    { 10040, " WSAEMSGSIZE: Message too long."},
    { 10041, " WSAEPROTOTYPE: Protocol wrong type for socket."},
    { 10047, " WSAEAFNOSUPPORT: Address family not supported by protocol family."},
    { 10048, " WSAEADDRINUSE: Address already in use."},
    { 10049, " WSAEADDRNOTAVAIL: Cannot assign requested address."},
    { 10050, " WSAENETDOWN: Network is down."},
    { 10051, " WSAEWSAENETUNREACH: Network is unreachable."},
    { 10052, " WSAENETRESET: Network dropped connection on reset."},
    { 10053, " WSAECONNABORTED: Software caused connection abort."},
    { 10054, " WSAECONNRESET: Connection reset by peer."},
    { 10060, " WSAETIMEDOUT: Connection timed out."},
    { 10093, " WSANOTINITIALIZED: Successful WSAStartup not yet performed."}
};


bool SetNatNetVersion(int major, int minor)
{
    sPacket* PacketOut = new sPacket();
    if (PacketOut)
    {
        // send NAT_MESSAGESTRING (will respond on the "Command Listener" thread)
        //strcpy_s(szRequest, "TestRequest");
        char szRequest[512];
        sprintf(szRequest, "Bitstream,%1.1d.%1.1d", major, minor);
        PacketOut->iMessage = NAT_REQUEST;
        PacketOut->nDataBytes = (short)strlen(szRequest) + 1;
        strcpy(PacketOut->Data.szData, szRequest);
        int nTries = 3;
        int iRet = SOCKET_ERROR;
        while (nTries--)
        {
            iRet = sendto(gCommandSocket, (char*)PacketOut, 4 + PacketOut->nDataBytes, 0, (sockaddr*)&gHostAddr, sizeof(gHostAddr));
            if (iRet != SOCKET_ERROR)
                break;
        }
        printf("Command: %s returned value: %d%s\n", szRequest, iRet, (iRet == SOCKET_ERROR) ? " SOCKET_ERROR" : "");

        // make bitstream switch by requesting mocap frame
        if (iRet != SOCKET_ERROR){
            char szCommand[512];
            int returnCode;
            std::vector<std::string> commandVec{
                "TimelinePlay",
                "TimelineStop",
                "SetPlaybackCurrentFrame,0",
                "TimelineStop"
            };

            for (int i = 0; i < commandVec.size(); ++i) {
                strcpy_s(szCommand, commandVec[i].c_str());
                returnCode = SendCommand(szCommand);
                printf("Command: %s -  returnCode: %d\n", szCommand, returnCode);
            }


        }


        // cleanup
        delete PacketOut;
        PacketOut = nullptr;

        if (iRet == SOCKET_ERROR)
        {
            return false;
        }
        gNatNetVersion[0] = major;
        gNatNetVersion[1] = minor;
        gNatNetVersion[2] = 0;
        gNatNetVersion[3] = 0;

    }

    return true;
}

std::string GetWSAErrorString(int errorValue)
{
    // Additional values can be found in Winsock2.h or
    // https://docs.microsoft.com/en-us/windows/win32/winsock/windows-sockets-error-codes-2

    std::string errorString = std::to_string(errorValue);
    // loop over entries in map
    auto mapItr = wsaErrors.begin();
    for (; mapItr != wsaErrors.end(); ++mapItr)
    {
        if (mapItr->first == errorValue) {
            errorString += mapItr->second;
            return errorString;
        }
    }
// If it gets here, the code is unknown, so show the reference link.																		
    errorString += std::string(" Please see: https:\/\/docs.microsoft.com\/en-us\/windows\/win32\/winsock\/windows-sockets-error-codes-2");
    return errorString;
}

/*
* MakeAlnum
* For now, make sure the string is printable ascii.  
*/ 
void MakeAlnum(char* szName, int len)
{
    int i = 0, i_max = len;
    szName[len - 1] = 0;
    while ((i < len) && (szName[i] != 0))
    {
        if (szName[i] == 0)
        {
            break;
        }
        if (isalnum(szName[i]) == 0)
        {
            szName[i] = ' ';
        }
        ++i;
    }
}

/*
* CommandListenThread
* Manage the command channel
*/
DWORD WINAPI CommandListenThread(void* dummy)
{
    DWORD retValue = 0;
    int addr_len;
    int nDataBytesReceived;
    char str[MAX_NAMELENGTH];
    sockaddr_in TheirAddress;
    sPacket* PacketIn = new sPacket();
    sPacket* PacketOut = new sPacket();
    addr_len = sizeof(struct sockaddr);

    if (PacketIn && PacketOut)
    {
        printf("[PacketClient CLTh] CommandListenThread Started\n");
        while (1)
        {
            // Send a Keep Alive message to Motive (required for Unicast transmission only)
            if (!gUseMulticast)
            {
                PacketOut->iMessage = NAT_KEEPALIVE;
                PacketOut->nDataBytes = 0;
                int iRet = sendto(gCommandSocket, (char*)PacketOut, 4 + PacketOut->nDataBytes, 0, (sockaddr*)&gHostAddr, sizeof(gHostAddr));
                if (iRet == SOCKET_ERROR)
                {
                    printf("[PacketClient CLTh] sendto failure   (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
                }
            }

            // blocking with timeout
            nDataBytesReceived = recvfrom(gCommandSocket, (char*)PacketIn, sizeof(sPacket),
                0, (struct sockaddr*)&TheirAddress, &addr_len);

            if ((nDataBytesReceived == 0))
            {
                continue;
            }
            else if (nDataBytesReceived == SOCKET_ERROR)
            {
                if (WSAGetLastError() != 10060)// Ignore normal timeout failures
                {
                    printf("[PacketClient CLTh] recvfrom failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
                }
                continue;
            }
            // debug - print message
            sprintf(str, "[PacketClient CLTh] Received command from %d.%d.%d.%d: Command=%d, nDataBytes=%d",
                TheirAddress.sin_addr.S_un.S_un_b.s_b1, TheirAddress.sin_addr.S_un.S_un_b.s_b2,
                TheirAddress.sin_addr.S_un.S_un_b.s_b3, TheirAddress.sin_addr.S_un.S_un_b.s_b4,
                (int)PacketIn->iMessage, (int)PacketIn->nDataBytes);

            printf("%s\n", str);
            // handle command
            switch (PacketIn->iMessage)
            {
            case NAT_SERVERINFO: // 1
                strcpy_s(gServerName, PacketIn->Data.Sender.szName);
                for (int i = 0; i < 4; i++)
                {
                    gNatNetVersionServer[i] = (int)PacketIn->Data.Sender.NatNetVersion[i];
                    gServerVersion[i] = (int)PacketIn->Data.Sender.Version[i];
                }
                if ((gNatNetVersion[0] == 0)&&(gNatNetVersion[1] == 0)) {
                    for (int i = 0; i < 4; i++)
                    {
                        gNatNetVersion[i] = gNatNetVersionServer[i];
                    }
                    if ((gNatNetVersionServer[0] >= 4) && (!gUseMulticast))
                    {
                        gCanChangeBitstream = true;
                    }
                }
                
                printf("[PacketClient CLTh]  NatNet Server Info\n");
                printf("[PacketClient CLTh]    Sending Application Name: %s\n", gServerName);
                printf("[PacketClient CLTh]    NatNetVersion %d %d %d %d\n", 
                    gNatNetVersion[0], gNatNetVersion[1], gNatNetVersion[2], gNatNetVersion[3]);
                printf("[PacketClient CLTh]    ServerVersion %d %d %d %d\n",
                    gServerVersion[0], gServerVersion[1], gServerVersion[2], gServerVersion[3]);
                break;
            case NAT_RESPONSE: // 3
                gCommandResponseSize = PacketIn->nDataBytes;
                if (gCommandResponseSize == 4)
                    memcpy(&gCommandResponse, &PacketIn->Data.lData[0], gCommandResponseSize);
                else
                {
                    memcpy(&gCommandResponseString[0], &PacketIn->Data.cData[0], gCommandResponseSize);
                    printf("[PacketClient CLTh]    Response : %s\n", gCommandResponseString);
                    gCommandResponse = 0;   // ok
                }
                break;
            case NAT_MODELDEF: //5
                Unpack((char*)PacketIn);
                break;
            case NAT_FRAMEOFDATA: // 7
                Unpack((char*)PacketIn);
                break;
            case NAT_UNRECOGNIZED_REQUEST: //100
                printf("[PacketClient CLTh]    Received iMessage 100 = 'unrecognized request'\n");
                gCommandResponseSize = 0;
                gCommandResponse = 1;       // err
                break;
            case NAT_MESSAGESTRING: //8
                printf("[PacketClient CLTh]    Received message: %s\n", PacketIn->Data.szData);
                break;
            default:
                printf("[PacketClient CLTh]    Received unknown command %d\n",
                    PacketIn->iMessage);
            }
        }// end of while
    }
    else {
        printf("[PacketClient CLTh] CommandListenThread Start FAILURE\n");
        retValue = 1;
    }
    if (PacketIn)
    {
        delete PacketIn;
        PacketIn = nullptr;
    }
    if (PacketOut)
    {
        delete PacketOut;
        PacketOut = nullptr;
    }
    return retValue;
}

// Data listener thread. Listens for incoming bytes from NatNet
DWORD WINAPI DataListenThread(void* dummy)
{
    const int baseDataBytes = 48 * 1024;
    char* szData = nullptr;
    int nDataBytes = 0;
    szData = new char[baseDataBytes];
    if (szData)
    {
        nDataBytes = baseDataBytes;
    }
    else {
        printf("[PacketClient DLTh] DataListenThread Start FAILURE memory allocation\n");
    }
    int addr_len = sizeof(struct sockaddr);
    sockaddr_in TheirAddress;
    printf("[PacketClient DLTh] DataListenThread Started\n");

    while (1)
    {
        // Block until we receive a datagram from the network (from anyone including ourselves)
        int nDataBytesReceived = recvfrom(gDataSocket, szData, nDataBytes, 0, (sockaddr *)&TheirAddress, &addr_len);
        // Once we have bytes recieved Unpack organizes all the data
        if (nDataBytesReceived>0)
        {
            Unpack(szData);
        }
        else if (nDataBytesReceived < 0)
        {
            int wsaLastError = WSAGetLastError();
            printf("[PacketClient DLTh] gDataSocket failure (error: %d)\n", nDataBytesReceived);
            printf("[PacketClient DLTh] WSAError (error: %s)\n", GetWSAErrorString(wsaLastError).c_str());
            if (wsaLastError == 10040)
            {
                // peek at truncated data, determine better buffer size
                int messageID = 0;
                int nBytes = 0;
                int nBytesTotal = 0;
                UnpackPacketHeader(szData, messageID, nBytes, nBytesTotal);
                printf("[PacketClient DLTh] messageID %d nBytes %d nBytesTotal %d\n",
                    messageID, nBytes, nBytesTotal);
                if (nBytesTotal <= MAX_PACKETSIZE) {
                    int newSize = nBytesTotal + 10000;
                    newSize = min(newSize, (int)MAX_PACKETSIZE);
                    char* szDataNew = new char[newSize];
                    if (szDataNew) {
                        printf("[PacketClient DLTh] Resizing data buffer from %d bytes to %d bytes",
                            nDataBytes, newSize);
                        if (szData) {
                            delete[] szData;
                        }
                        szData = szDataNew;
                        nDataBytes = newSize;
                        szDataNew = nullptr;
                        newSize = 0;
                    }
                    else
                    {
                        printf("PacketClient DLTh] Data buffer size failure have %d bytes but need %d bytes", nDataBytes, nBytesTotal);
                    }
                }
                else
                {
                    printf("PacketClient DLTh] Data buffer size failure have %d bytes but need %d bytes", nDataBytes, nBytesTotal);
                }

            }
        }
    }
    if (szData)
    {
        delete[] szData;
        szData = nullptr;
        nDataBytes = 0;
    }
    return 0;
}

SOCKET CreateCommandSocket(unsigned long IP_Address, unsigned short uPort, int optval,
    bool useMulticast)
{
    int retval = SOCKET_ERROR;
    struct sockaddr_in my_addr;
    static unsigned long ivalue = 0x0;
    static unsigned long bFlag = 0x0;
    int nlengthofsztemp = 64;
    SOCKET sockfd = -1;
    int optval_size = sizeof(int);
    ivalue = 1;
    int bufSize = optval;

    int protocol = 0;

    if (!useMulticast)
    {
        protocol = IPPROTO_UDP;
    }
    // Create a datagram socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, protocol)) == INVALID_SOCKET)
    {
        printf("[PacketClient Main] gCommandSocket create failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        return -1;
    }

    // bind socket
    memset(&my_addr, 0, sizeof(my_addr));
    my_addr.sin_family = AF_INET;
    my_addr.sin_port = htons(uPort);
    my_addr.sin_addr.S_un.S_addr = IP_Address;

    if (bind(sockfd, (struct sockaddr*)&my_addr, sizeof(struct sockaddr)) == SOCKET_ERROR)
    {
        printf("[PacketClient Main] gCommandSocket bind failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        closesocket(sockfd);
        return -1;
    }

    if (useMulticast)
    {
        // set to broadcast mode
        if (setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, (char*)&ivalue, sizeof(ivalue)) == SOCKET_ERROR)
        {
            // error - should show setsockopt error.
            closesocket(sockfd);
            return -1;
        }

        // set a read timeout to allow for sending keep_alive message
        int timeout = 2000;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);
    }
    retval = getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char*)&optval, &optval_size);
    if (retval == SOCKET_ERROR)
    {
        // error
        printf("[PacketClient Main] gCommandSocket get options  SO_RCVBUF failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        closesocket(sockfd);
        return -1;
    }
    if (optval != OPTVAL_REQUEST_SIZE)
    {
        // err - actual size...
        printf("[PacketClient Main] gCommandSocket Receive Buffer size = %d requested %d\n",
            optval, OPTVAL_REQUEST_SIZE);
    }
    if (useMulticast)
    {
        // [optional] set to non-blocking
        //u_long iMode=1;
        //ioctlsocket(gCommandSocket,FIONBIO,&iMode); 
        // set buffer
        retval = setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char*)&optval, 4);
        if (retval == SOCKET_ERROR)
        {
            // error
            printf("[PacketClient Main] gCommandSocket set options SO_RCVBUF failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            closesocket(sockfd);
            return -1;
        }
    }
    // Unicast case
    else
    {
        // set a read timeout to allow for sending keep_alive message for unicast clients
        int timeout = 2000;
        setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof timeout);

        // allow multiple clients on same machine to use multicast group address/port
        int value = 1;
        int retval = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char*)&value, sizeof(value));
        if (retval == -1)
        {
            printf("[PacketClient Main] gCommandSocket setsockopt failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            closesocket(sockfd);
            return -1;
        }


        // set user-definable send buffer size
        int defaultBufferSize=0;
        socklen_t optval_size = 4;
        retval = getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char*)&defaultBufferSize, &optval_size);
        retval = setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char*)&bufSize, sizeof(bufSize));
        if (retval == -1)
        {
            printf("[PacketClient Main] gCommandSocket user send buffer failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            closesocket(sockfd);
            return -1;
        }
        int confirmValue = 0;
        getsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, (char*)&confirmValue, &optval_size);
        if (confirmValue != bufSize)
        {
            // not fatal, but notify user requested size is not valid
            printf("[PacketClient Main] gCommandSocket buffer smaller than expected %d instead of %d\n",
                confirmValue, bufSize);
        }

        // Set "Don't Fragment" bit in IP header to false (0).
        // note : we want fragmentation support since our packets are over the standard ethernet MTU (~1500 bytes).
        int optval2;
        socklen_t optlen = sizeof(int);
        int iRet = getsockopt(sockfd, IPPROTO_IP, IP_DONTFRAGMENT, (char*)&optval2, &optlen);
        optval2 = 0;
        iRet = setsockopt(sockfd, IPPROTO_IP, IP_DONTFRAGMENT, (char*)&optval2, sizeof(optval2));
        if (iRet == -1)
        {
            printf("[PacketClient Main] gCommandSocket Don't fragment request failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            closesocket(sockfd);
            return -1;
        }
        iRet = getsockopt(sockfd, IPPROTO_IP, IP_DONTFRAGMENT, (char*)&optval2, &optlen);

    }
    

    return sockfd;
}

SOCKET CreateDataSocket(unsigned long socketIPAddress, unsigned short uUnicastPort, int optval,
    bool useMulticast, unsigned long multicastIPAddress, unsigned short uMulticastPort)
{
    int retval = SOCKET_ERROR;
    static unsigned long ivalue = 0x0;
    static unsigned long bFlag = 0x0;
    int nlengthofsztemp = 64;
    SOCKET sockfd = -1;
    int optval_size = sizeof(int);
    int value = 1;
    // create the socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd == SOCKET_ERROR)
    {
        printf("[PacketClient Main] gDataSocket socket allocation failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        return -1;
    }
    // allow multiple clients on same machine to use address/port
    retval = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char*)&value, sizeof(value));
    if (retval == SOCKET_ERROR)
    {
        printf("[PacketClient Main] gDataSocket SO_REUSEADDR setsockopt failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        closesocket(sockfd);
        return -1;
    }

    if (useMulticast)
    {
        // Bind socket to address/port								  
        struct sockaddr_in MySocketAddress;
        memset(&MySocketAddress, 0, sizeof(MySocketAddress));
        MySocketAddress.sin_family = AF_INET;
        MySocketAddress.sin_port = htons(uMulticastPort);
        MySocketAddress.sin_addr.S_un.S_addr = socketIPAddress;
        if (bind(sockfd, (struct sockaddr*)&MySocketAddress, sizeof(struct sockaddr)) == SOCKET_ERROR)
        {
            printf("[PacketClient Main] gDataSocket bind failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            closesocket(sockfd);
            return -1;
        }

        // If Motive is transmitting data in Multicast, must join multicast group
        in_addr MyAddress, MultiCastAddress;
        MyAddress.S_un.S_addr = socketIPAddress;
        MultiCastAddress.S_un.S_addr = multicastIPAddress;

        struct ip_mreq Mreq;
        Mreq.imr_multiaddr = MultiCastAddress;
        Mreq.imr_interface = MyAddress;
        retval = setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&Mreq, sizeof(Mreq));
        if (retval == SOCKET_ERROR)
        {
            printf("[PacketClient Main] gDataSocket join failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            WSACleanup();
            return -1;
        }
    }
    //Unicast case
    else
    {
        // bind it
        struct sockaddr_in MyAddr;
        memset(&MyAddr, 0, sizeof(MyAddr));
        MyAddr.sin_family = AF_INET;
        MyAddr.sin_port = htons(uUnicastPort);
        MyAddr.sin_addr.S_un.S_addr = socketIPAddress;

        if (bind(sockfd, (struct sockaddr*)&MyAddr, sizeof(sockaddr_in)) == -1)
        {
            printf("[PacketClient Main] gDataSocket bind failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
            closesocket(sockfd);
            return -1;
        }

        char str[INET_ADDRSTRLEN];
        in_addr MyAddress, MultiCastAddress;
        MyAddress.S_un.S_addr = socketIPAddress;
        MultiCastAddress.S_un.S_addr = multicastIPAddress;
        inet_ntop(AF_INET, &(MultiCastAddress), str, INET_ADDRSTRLEN);
        if (strcmp(str, "255.255.255.255") == 0)
        {
            // client has indicated it wants to receive broadcast - do not join a multicast group
        }
        else
        {
            struct ip_mreq stMreq;
            stMreq.imr_multiaddr = MultiCastAddress;
            stMreq.imr_interface = MyAddress;
            retval = setsockopt(sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&stMreq, sizeof(stMreq));
            if (retval == -1)
            {
                printf("[PacketClient Main] gDataSocket setsockopt failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
                closesocket(sockfd);
                return -1;
            }
        }

    }



        // create a 1MB buffer
        retval = setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char*)&optval, 4);
        if (retval == SOCKET_ERROR)
        {
            printf("[PacketClient Main] gDataSocket setsockopt failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        }
        retval = getsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, (char*)&optval, &optval_size);
        if (retval == SOCKET_ERROR)
        {
            printf("[PacketClient Main] CreateDataSocket getsockopt failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        }
        if (optval != OPTVAL_REQUEST_SIZE)
        {
            printf("[PacketClient Main] gDataSocket ReceiveBuffer size = %d requested %d\n",
                optval, OPTVAL_REQUEST_SIZE);
        }


    
    // return working gDataSocket
    return sockfd;
}
void PrintConfiguration(const char * szMyIPAddress, const char * szServerIPAddress, bool useMulticast)
{
    printf("Connection Configuration:\n");
    printf("  Client:          %s\n", szMyIPAddress);
    printf("  Server:          %s\n", szServerIPAddress);
    printf("  Command Port:    %d\n", PORT_COMMAND);
    printf("  Data Port:       %d\n", PORT_DATA);

    if (useMulticast)
	{
        printf("  Using Multicast\n");
        printf("  Multicast Group: %s\n", MULTICAST_ADDRESS);
	}
    else
	{
        printf("  Using Unicast\n");
	}
    printf("  NatNet Server Info\n");
    printf("    Application Name %s\n", gServerName);
    printf("    NatNetVersion  %d %d %d %d\n",
        gNatNetVersionServer[0], gNatNetVersionServer[1],
        gNatNetVersionServer[2], gNatNetVersionServer[3]);
    printf("    ServerVersion  %d %d %d %d\n",
        gServerVersion[0], gServerVersion[1],
        gServerVersion[2], gServerVersion[3]);
    printf("  NatNet Bitstream Requested\n");
    printf("    NatNetVersion  %d %d %d %d\n",
        gNatNetVersion[0], gNatNetVersion[1],
        gNatNetVersion[2], gNatNetVersion[3]);
    printf("    Can Change Bitstream Version = %s\n", (gCanChangeBitstream) ? "true" : "false");
}


void PrintCommands(bool canChangeBitstream)
{
    printf("Commands:\n"
        "Return Data from Motive\n"
        "  s  send data descriptions\n"
        "  r  resume/start frame playback\n"
        "  p  pause frame playback\n"
        "     pause may require several seconds\n"
        "     depending on the frame data size\n"
        "Change Working Range\n"
        "  o  reset Working Range to: start/current/end frame 0/0/end of take\n"
        "  w  set Working Range to: start/current/end frame 1/100/1500\n"
        "Change NatNet data stream version (Unicast only)\n"
        "  3 Request NatNet 3.1 data stream (Unicast only)\n"
        "  4 Request NatNet 4.0 data stream (Unicast only)\n"
        "c  print configuration\n"
        "h  print commands\n"
        "q  quit\n"
        "\n"
        "NOTE: Motive frame playback will respond differently in\n"
        "       Endpoint, Loop, and Bounce playback modes.\n"
        "\n"
        "EXAMPLE: PacketClient [serverIP [ clientIP [ Multicast/Unicast]]]\n"
        "         PacketClient \"192.168.10.14\" \"192.168.10.14\" Multicast\n"
        "         PacketClient \"127.0.0.1\" \"127.0.0.1\" u\n"
        "\n"
    );
}


bool MyParseArgs(int argc, char* argv[], sParsedArgs &parsedArgs)
{
    bool retval = true;

    // Process arguments
    // server address
    if (argc > 1)
    {
        strcpy_s(parsedArgs.szServerIPAddress, argv[1]);	// specified on command line
        retval = IPAddress_StringToAddr(parsedArgs.szServerIPAddress, &parsedArgs.serverAddress);
    }
    // pull IP address from local IP addy
    else
    {
        // default to loopback
        retval = IPAddress_StringToAddr(parsedArgs.szServerIPAddress, &parsedArgs.serverAddress);
        // attempt to get address from local environment
        //GetLocalIPAddresses((unsigned long*)&parsedArgs.serverAddress, 1);
        // formatted print back to parsedArgs
        sprintf_s(parsedArgs.szServerIPAddress, "%d.%d.%d.%d",
            parsedArgs.serverAddress.S_un.S_un_b.s_b1,
            parsedArgs.serverAddress.S_un.S_un_b.s_b2,
            parsedArgs.serverAddress.S_un.S_un_b.s_b3,
            parsedArgs.serverAddress.S_un.S_un_b.s_b4);
    }

    if (retval == false)
        return retval;

    // client address
    if (argc > 2)
    {
        strcpy_s(parsedArgs.szMyIPAddress, argv[2]);	// specified on command line
        retval = IPAddress_StringToAddr(parsedArgs.szMyIPAddress, &parsedArgs.myAddress);
    }
    // pull IP address from local IP addy
    else
    {
        // default to loopback
        retval = IPAddress_StringToAddr(parsedArgs.szMyIPAddress, &parsedArgs.myAddress);
        // attempt to get IP from environment
        //GetLocalIPAddresses((unsigned long*)&parsedArgs.myAddress, 1);
        // print back to szMyIPAddress
        sprintf_s(parsedArgs.szMyIPAddress, "%d.%d.%d.%d",
            parsedArgs.myAddress.S_un.S_un_b.s_b1,
            parsedArgs.myAddress.S_un.S_un_b.s_b2,
            parsedArgs.myAddress.S_un.S_un_b.s_b3,
            parsedArgs.myAddress.S_un.S_un_b.s_b4);
    }
    if (retval == false)
        return retval;


    // unicast/multicast
    if ((argc > 3) && strlen(argv[3]))
    {
        char firstChar = toupper(argv[3][0]);
        switch (firstChar)
        {
        case 'M':
            parsedArgs.useMulticast = true;
            break;
        case 'U':
            parsedArgs.useMulticast = false;
            break;
        default:
            parsedArgs.useMulticast = true;
            break;
        }
    }
    return retval;
}

int main(int argc, char * argv[])
{
    int retval = SOCKET_ERROR;
    sParsedArgs parsedArgs;

    WSADATA wsaData; 
    int optval = OPTVAL_REQUEST_SIZE;
    int optval_size = 4;

    // Command Listener Attributes
    SECURITY_ATTRIBUTES commandListenSecurityAttribs;
    commandListenSecurityAttribs.nLength = sizeof(SECURITY_ATTRIBUTES);
    commandListenSecurityAttribs.lpSecurityDescriptor = NULL;
    commandListenSecurityAttribs.bInheritHandle = TRUE;
    DWORD commandListenThreadID;
    HANDLE commandListenThreadHandle;

    // Data Listener Attributes
    SECURITY_ATTRIBUTES dataListenThreadSecurityAttribs;
    dataListenThreadSecurityAttribs.nLength = sizeof(SECURITY_ATTRIBUTES);
    dataListenThreadSecurityAttribs.lpSecurityDescriptor = NULL;
    dataListenThreadSecurityAttribs.bInheritHandle = TRUE;
    DWORD dataListenThread_ID;
    HANDLE dataListenThread_Handle;

    // Start up winsock
    if (WSAStartup(0x202, &wsaData) == SOCKET_ERROR)
    {
        printf("[PacketClient Main] WSAStartup failed (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        WSACleanup();
        return 0;
    }


    if (MyParseArgs(argc, argv, parsedArgs) == false)
    {
        return -1;
    }
    gUseMulticast = parsedArgs.useMulticast;

    // multicast address - hard coded to MULTICAST_ADDRESS define above.
    parsedArgs.multiCastAddress.S_un.S_addr = inet_addr(MULTICAST_ADDRESS);

    // create "Command" socket
    int commandPort = 0;
    gCommandSocket = CreateCommandSocket(parsedArgs.myAddress.S_un.S_addr,commandPort, optval, gUseMulticast);
    if(gCommandSocket == -1)
    {
        // error
        printf("[PacketClient Main] gCommandSocket create failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        WSACleanup();
        return -1;
    }
    printf("[PacketClient Main] gCommandSocket started\n");

    // create the gDataSocket
    int dataPort = 0;
    gDataSocket = CreateDataSocket(parsedArgs.myAddress.S_un.S_addr,dataPort, optval,
        parsedArgs.useMulticast, parsedArgs.multiCastAddress.S_un.S_addr, PORT_DATA);
    if (gDataSocket == -1)
    {
        printf("[PacketClient Main] gDataSocket create failure (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        closesocket(gCommandSocket);
        WSACleanup();
        return -1;				  
    }
    printf("[PacketClient Main] gDataSocket started\n");

    // startup our "Command Listener" thread
    commandListenThreadHandle = CreateThread(&commandListenSecurityAttribs, 0, CommandListenThread, NULL, 0, &commandListenThreadID);
    printf("[PacketClient Main] CommandListenThread started\n");
    
    // startup our "Data Listener" thread
    dataListenThread_Handle = CreateThread( &dataListenThreadSecurityAttribs, 0, DataListenThread, NULL, 0, &dataListenThread_ID);
    printf("[PacketClient Main] DataListenThread started\n");

    // server address for commands
    memset(&gHostAddr, 0, sizeof(gHostAddr));
    gHostAddr.sin_family = AF_INET;        
    gHostAddr.sin_port = htons(PORT_COMMAND); 
    gHostAddr.sin_addr = parsedArgs.serverAddress;

    // send initial connect request
    sPacket* PacketOut = new sPacket;
    sSender sender;
    sConnectionOptions connectOptions;
    PacketOut->iMessage = NAT_CONNECT;
    PacketOut->nDataBytes = sizeof(sSender) + sizeof(connectOptions) + 4;
    memset(&sender, 0, sizeof(sender));
    memcpy(&PacketOut->Data, &sender, (int)sizeof(sSender));

    // [optional] Custom connection options
    /*
    connectOptions.subscribedDataOnly = true;
    connectOptions.BitstreamVersion[0] = 2;
    connectOptions.BitstreamVersion[1] = 2;
    connectOptions.BitstreamVersion[2] = 0;
    connectOptions.BitstreamVersion[3] = 0;
    */
    memcpy(&PacketOut->Data.cData[(int)sizeof(sSender)], &connectOptions, sizeof(connectOptions));

    int nTries = 3;
    int iRet = SOCKET_ERROR;
    while (nTries--)
    {
        iRet = sendto(gCommandSocket, (char *)PacketOut, 4 + PacketOut->nDataBytes, 0, (sockaddr *)&gHostAddr, sizeof(gHostAddr));
        if(iRet != SOCKET_ERROR)
            break;
    }
    if (iRet == SOCKET_ERROR)
    {
        printf("[PacketClient Main] gCommandSocket sendto error (error: %s)\n", GetWSAErrorString(WSAGetLastError()).c_str());
        return -1;
    }

    // just to make things look more orderly on startup
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    printf("[PacketClient Main] Started\n\n");
    PrintConfiguration(parsedArgs.szMyIPAddress, parsedArgs.szServerIPAddress, parsedArgs.useMulticast);
    PrintCommands(gCanChangeBitstream);

    int c;
    char szRequest[512] = { 0 };
    bool bExit = false;
    nTries = 3;
    iRet = SOCKET_ERROR;
    std::string errorString;
    while (!bExit)
    {
        c =_getch();
        switch(c)
        {
        case 's':
            // send NAT_REQUEST_MODELDEF command to server (will respond on the "Command Listener" thread)
            PacketOut->iMessage = NAT_REQUEST_MODELDEF;
            PacketOut->nDataBytes = 0;
            nTries = 3;
            iRet = SOCKET_ERROR;
            while (nTries--)
            {
                iRet = sendto(gCommandSocket, (char *)PacketOut, 4 + PacketOut->nDataBytes, 0, (sockaddr *)&gHostAddr, sizeof(gHostAddr));
                if(iRet != SOCKET_ERROR)
                    break;
            }
            printf("Command: NAT_REQUEST_MODELDEF returned value: %d%s\n", iRet, (iRet==-1)?" SOCKET_ERROR":"");
            break;
        case 'p':
        {
            char szCommand[512];
            sprintf(szCommand, "TimelineStop");
            printf("Command: %s - ", szCommand);
            int returnCode = SendCommand(szCommand);
            printf(" returnCode: %d\n", returnCode);
        }

        break;
        case 'r':
        {
            char szCommand[512];
            sprintf(szCommand, "TimelinePlay");
            int returnCode = SendCommand(szCommand);
            printf("Command: %s -  returnCode: %d\n", szCommand, returnCode);
        }

        break;
        case 'h':
            PrintCommands(gCanChangeBitstream);
            break;
        case 'c':
            PrintConfiguration(parsedArgs.szMyIPAddress, parsedArgs.szServerIPAddress, parsedArgs.useMulticast);
            break;
        case 'o':
        {
            char szCommand[512];
            long startFrameNum = 0;
            long endFrameNum = 100000;
            int returnCode;
            std::vector<std::string> commandVec {
                "TimelineStop",
                "SetPlaybackStartFrame,0",
                "SetPlaybackCurrentFrame,0",
                "SetPlaybackStopFrame,1000000",
                "SetPlaybackLooping,0",
                "TimelineStop"
            };

            for (int i = 0; i < commandVec.size(); ++i) {
                strcpy_s(szCommand, commandVec[i].c_str());
                returnCode = SendCommand(szCommand);
                printf("Command: %s -  returnCode: %d\n", szCommand, returnCode);

            }

        }
            break;
        case 'w':
            {
                char szCommand[512];
                int returnCode;
                std::vector<std::string> commandVec{
                    "TimelineStop",
                    "SetPlaybackStartFrame,10",
                    "SetPlaybackCurrentFrame,100",
                    "SetPlaybackStopFrame,1500",
                    "SetPlaybackLooping,0",
                    "TimelineStop"
                };

                for (int i = 0; i < commandVec.size(); ++i) {
                    strcpy_s(szCommand, commandVec[i].c_str());
                    returnCode = SendCommand(szCommand);
                    printf("Command: %s -  returnCode: %d\n", szCommand, returnCode);
                }


            }
            break;
        case '3':
            {
                if ( gCanChangeBitstream)
                {
                    SetNatNetVersion(3, 1);
                }
                else
                {
                    printf("Bitstream changes allowed for Unicast with NatNetServer >= 4 only\n");
                }
            }
            break;
        case '4':
            {
                if ( gCanChangeBitstream)
                {
                    SetNatNetVersion(4,0);
                }
                else
                {
                    printf("Bitstream changes allowed for Unicast with NatNetServer >= 4 only\n");
                }
            }
        break;
        case 'q':
            bExit = true;		
            break;	
        default:
            break;
        }
    }

    return 0;
}

// Send a command to Motive.  
int SendCommand(char * szCommand)
{
    // reset global result
    gCommandResponse = -1;

    // format command packet
    sPacket * commandPacket = new sPacket();
    strcpy(commandPacket->Data.szData, szCommand);
    commandPacket->iMessage = NAT_REQUEST;
    commandPacket->nDataBytes = (short)strlen(commandPacket->Data.szData) + 1;

    // send command, and wait (a bit) for command response to set global response var in CommandListenThread
    int iRet = sendto(gCommandSocket, (char *)commandPacket, 4 + commandPacket->nDataBytes, 0, (sockaddr *)&gHostAddr, sizeof(gHostAddr));
    if(iRet == SOCKET_ERROR)
    {
        printf("Socket error sending command\n");
    }
    else
    {
        int waitTries = 5;
        while (waitTries--)
        {
            if(gCommandResponse != -1)
                break;
            Sleep(30);
        }

        if(gCommandResponse == -1)
        {
            printf("Command response not received (timeout)\n");
        }
        else if(gCommandResponse == 0)
        {
            printf("Command response received with success\n");
        }
        else if(gCommandResponse > 0)
        {
            printf("Command response received with errors\n");
        }
        else
        {
            printf("Command response unknown value=%d\n", gCommandResponse);
        }
    }

    return gCommandResponse;
}

// Convert IP address string to address
bool IPAddress_StringToAddr(char *szNameOrAddress, struct in_addr *Address)
{
	int retVal;
	struct sockaddr_in saGNI;
	char hostName[MAX_NAMELENGTH];
	char servInfo[MAX_NAMELENGTH];
	u_short port;
	port = 0;

	// Set up sockaddr_in structure which is passed to the getnameinfo function
	saGNI.sin_family = AF_INET;
	saGNI.sin_addr.s_addr = inet_addr(szNameOrAddress);
	saGNI.sin_port = htons(port);

	// getnameinfo in WS2tcpip is protocol independent and resolves address to ANSI host name
	if ((retVal = getnameinfo((SOCKADDR *)&saGNI, sizeof(sockaddr), hostName, MAX_NAMELENGTH, servInfo, MAX_NAMELENGTH, NI_NUMERICSERV)) != 0)
	{
        // Returns error if getnameinfo failed
        printf("[PacketClient Main] GetHostByAddr failed. Error #: %ld\n", WSAGetLastError());
		return false;
	}

    Address->S_un.S_addr = saGNI.sin_addr.S_un.S_addr;
	
    return true;
}

// get ip addresses on local host
int GetLocalIPAddresses(unsigned long Addresses[], int nMax)
{
    unsigned long  NameLength = 128;
    char szMyName[1024];
    struct addrinfo aiHints;
	struct addrinfo *aiList = NULL;
    struct sockaddr_in addr;
    int retVal = 0;
    char * port = "0";
    
    if(GetComputerName(szMyName, &NameLength) != TRUE)
    {
        printf("[PacketClient Main] get computer name  failed. Error #: %ld\n", WSAGetLastError());
        return 0;       
    };

	memset(&aiHints, 0, sizeof(aiHints));
	aiHints.ai_family = AF_INET;
	aiHints.ai_socktype = SOCK_DGRAM;
	aiHints.ai_protocol = IPPROTO_UDP;

    // Take ANSI host name and translates it to an address
	if ((retVal = getaddrinfo(szMyName, port, &aiHints, &aiList)) != 0) 
	{
        printf("[PacketClient Main] getaddrinfo failed. Error #: %ld\n", WSAGetLastError());
        return 0;
	}

    memcpy(&addr, aiList->ai_addr, aiList->ai_addrlen);
    freeaddrinfo(aiList);
    Addresses[0] = addr.sin_addr.S_un.S_addr;

    return 1;
}

// Funtion that assigns a time code values to 5 variables passed as arguments
// Requires an integer from the packet as the timecode and timecodeSubframe
bool DecodeTimecode(unsigned int inTimecode, unsigned int inTimecodeSubframe, int* hour, int* minute, int* second, int* frame, int* subframe)
{
	bool bValid = true;

	*hour = (inTimecode>>24)&255;
	*minute = (inTimecode>>16)&255;
	*second = (inTimecode>>8)&255;
	*frame = inTimecode&255;
	*subframe = inTimecodeSubframe;

	return bValid;
}

// Takes timecode and assigns it to a string
bool TimecodeStringify(unsigned int inTimecode, unsigned int inTimecodeSubframe, char *Buffer, int BufferSize)
{
	bool bValid;
	int hour, minute, second, frame, subframe;
	bValid = DecodeTimecode(inTimecode, inTimecodeSubframe, &hour, &minute, &second, &frame, &subframe);

	sprintf_s(Buffer,BufferSize,"%2d:%2d:%2d:%2d.%d",hour, minute, second, frame, subframe);
	for(unsigned int i=0; i<strlen(Buffer); i++)
		if(Buffer[i]==' ')
			Buffer[i]='0';

	return bValid;
}

void DecodeMarkerID(int sourceID, int* pOutEntityID, int* pOutMemberID)
{
    if (pOutEntityID)
        *pOutEntityID = sourceID >> 16;

    if (pOutMemberID)
        *pOutMemberID = sourceID & 0x0000ffff;
}


// *********************************************************************
//
//  UnpackDescription:
//      Receives pointer to byes of a data description
//
// *********************************************************************

char* UnpackDescription(char* inptr, int nBytes, int major, int minor)
{
    char* ptr = inptr;
    char* targetPtr = ptr + nBytes;
    long long nBytesProcessed = (long long)ptr - (long long)inptr;
    // number of datasets
    int nDatasets = 0; memcpy(&nDatasets, ptr, 4); ptr += 4;
    printf("Dataset Count : %d\n", nDatasets);
#ifdef VDEBUG
    int datasetCounts[7] = { 0,0,0,0,0,0,0 };
#endif
    bool errorDetected = false;
    for (int i = 0; i < nDatasets; i++)
    {
        printf("Dataset %d\n", i);
#ifdef VDEBUG
        int nBytesUsed = (long long)ptr - (long long)inptr;
        int nBytesRemaining = nBytes - nBytesUsed;
        printf("Bytes Decoded: %d Bytes Remaining: %d)\n",
            nBytesUsed, nBytesRemaining);
#endif

        // Determine type and advance
        // The next type entry is inaccurate 
        // if data descriptions are out of date
        int type = 0; memcpy(&type, ptr, 4); ptr += 4;
#ifdef VDEBUG
        if ((0 <= type) && (type <= 5))
        {
            datasetCounts[type] += 1;
        }
        else
        {
            datasetCounts[6] += 1;
        }
#endif

        switch (type)
        {
        case 0: // Markerset
        {
            printf("Type: 0 Markerset\n");
            ptr = UnpackMarkersetDescription(ptr, targetPtr, major, minor);
        }
        break;
        case 1: // rigid body
            printf("Type: 1 Rigid Body\n");
            ptr = UnpackRigidBodyDescription(ptr, targetPtr, major, minor);
            break;
        case 2: // skeleton
            printf("Type: 2 Skeleton\n");
            ptr = UnpackSkeletonDescription(ptr, targetPtr, major, minor);
            break;
        case 3: // force plate
            printf("Type: 3 Force Plate\n");
            ptr = UnpackForcePlateDescription(ptr, targetPtr, major, minor);
            break;
        case 4: // device
            printf("Type: 4 Device\n");
            ptr = UnpackDeviceDescription(ptr, targetPtr, major, minor);
            break;
        case 5: // camera
            printf("Type: 5 Camera\n");
            ptr = UnpackCameraDescription(ptr, targetPtr, major, minor);
            break;
        default: // unknown type
            printf("Type: %d UNKNOWN\n", type);
            printf("ERROR: Type decode failure\n");
            errorDetected = true;
            break;
        }
        if (errorDetected)
        {
            printf("ERROR: Stopping decode\n");
            break;
        }
        if (ptr > targetPtr)
        {
            printf("UnpackDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
            return ptr;
        }
        printf("\t%d datasets processed of %d\n", (i+1), nDatasets);
        printf("\t%lld bytes processed of %d\n", ((long long)ptr- (long long)inptr), nBytes);
    }   // next dataset

#ifdef VDEBUG
    printf("Cnt Type    Description\n");
    for (int i = 0; i < 7; ++i)
    {
        printf("%3.3d ", datasetCounts[i]);
        switch (i)
        {
        case 0: // Markerset
            printf("Type: 0 Markerset\n");
            break;
        case 1: // rigid body
            printf("Type: 1 rigid body\n");
            break;
        case 2: // skeleton
            printf("Type: 2 skeleton\n");
            break;
        case 3: // force plate
            printf("Type: 3 force plate\n");
            break;
        case 4: // device
            printf("Type: 4 device\n");
            break;
        case 5: // camera
            printf("Type: 5 camera\n");
            break;
        default:
            printf("Type: %d UNKNOWN\n", i);
            break;
        }
    }
#endif
    return ptr;
}


//
// UnpackMarkersetDescription
// (sMarkerSetDescription)
//
char * UnpackMarkersetDescription(char *ptr, char * targetPtr, int major,int minor)
{
    // name
    char szName[MAX_NAMELENGTH];
    strcpy_s(szName, ptr);
    int nDataBytes = (int)strlen(szName) + 1;
    ptr += nDataBytes;
    MakeAlnum(szName, MAX_NAMELENGTH);
    printf("Markerset Name: %s\n", szName);

    // marker data
    int nMarkers = 0; memcpy(&nMarkers, ptr, 4); ptr += 4;
    printf("Marker Count : %d\n", nMarkers);

    for (int j = 0; j < nMarkers; j++)
    {
        char szName[MAX_NAMELENGTH];
        strcpy_s(szName, ptr);
        int nDataBytes = (int)strlen(ptr) + 1;
        ptr += nDataBytes;
        MakeAlnum(szName,MAX_NAMELENGTH);
        printf("  %3.1d Marker Name: %s\n", j, szName);
        if (ptr > targetPtr)
        {
            printf("UnpackMarkersetDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
            return ptr;
        }
    }

    return ptr;
}


//
// UnpackRigidBodyDescription
// (sRigidBodyDescription)
//
char * UnpackRigidBodyDescription(char * inptr,  char* targetPtr,int major, int minor)
{
    char* ptr = inptr;
    int nBytes = 0; // common scratch variable
    if ((major >= 2) || (major == 0))
    {
        // RB name
        char szName[MAX_NAMELENGTH];
        strcpy_s(szName, ptr);
        ptr += strlen(ptr) + 1;
        MakeAlnum(szName, MAX_NAMELENGTH);
        printf("  Rigid Body Name: %s\n", szName);
    }

    int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
    printf("  RigidBody ID   : %d\n", ID);

    int parentID = 0; memcpy(&parentID, ptr, 4); ptr += 4;
    printf("  Parent ID      : %d\n", parentID);

    // Offsets
    float xoffset = 0; memcpy(&xoffset, ptr, 4); ptr += 4;
    float yoffset = 0; memcpy(&yoffset, ptr, 4); ptr += 4;
    float zoffset = 0; memcpy(&zoffset, ptr, 4); ptr += 4;
    printf("  Position       : %3.2f, %3.2f, %3.2f\n", xoffset, yoffset, zoffset);

    if (ptr > targetPtr)
    {
        printf("UnpackRigidBodyDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
        return ptr;
    }

    if ((major >= 3) || (major == 0))
    {
        int nMarkers = 0; memcpy(&nMarkers, ptr, 4); ptr += 4;
        printf("  Number of Markers : %d\n", nMarkers);
        if (nMarkers > 16000)
        {
            int nBytesProcessed = (int)(targetPtr - ptr);
            printf("UnpackRigidBodyDescription: UNPACK ERROR DETECTED: STOPPING DECODE at %d processed\n",
                nBytesProcessed);
            printf("                           Unreasonable number of markers\n");
            return targetPtr + 4;
        }

        if (nMarkers > 0) {

            printf("  Marker Positions:\n");
            char* ptr2 = ptr + (nMarkers * sizeof(float) * 3);
            char* ptr3 = ptr2 + (nMarkers * sizeof(int));
            for (int markerIdx = 0; markerIdx < nMarkers; ++markerIdx)
            {
                float xpos, ypos, zpos;
                int32_t label;
                char szMarkerNameUTF8[MAX_NAMELENGTH] = { 0 };
                char szMarkerName[MAX_NAMELENGTH] = { 0 };
                // marker positions
                memcpy(&xpos, ptr, 4); ptr += 4;
                memcpy(&ypos, ptr, 4); ptr += 4;
                memcpy(&zpos, ptr, 4); ptr += 4;

                // Marker Required activeLabels
                memcpy(&label, ptr2, 4); ptr2 += 4;

                // Marker Name
                szMarkerName[0] = 0;
                if ((major >= 4) || (major == 0)) {
                    strcpy_s(szMarkerName, ptr3);
                    ptr3 += strlen(ptr3) + 1;
                }

                printf("    %3.1d Marker Label: %3.1d Position: %6.6f %6.6f %6.6f %s\n",
                    markerIdx, label, xpos, ypos, zpos, szMarkerName);
                if (ptr3 > targetPtr)
                {
                    printf("UnpackRigidBodyDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
                    return ptr3;
                }
            }
            ptr = ptr3; // advance to the end of the labels & marker names
        }
    }

    if (ptr > targetPtr)
    {
        printf("UnpackRigidBodyDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
        return ptr;
    }
    printf("UnpackRigidBodyDescription processed %lld bytes\n", ((long long)ptr - (long long)inptr));
    return ptr;
}


//
// UnpackSkeletonDescription
//
char * UnpackSkeletonDescription(char * ptr,  char* targetPtr,int major, int minor) 
{
    char szName[MAX_NAMELENGTH];
    // Name
    strcpy_s(szName, ptr);
    ptr += strlen(ptr) + 1;
    MakeAlnum(szName, MAX_NAMELENGTH);
    printf("Name: %s\n", szName);

    // ID
    int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
    printf("ID : %d\n", ID);

    // # of RigidBodies
    int nRigidBodies = 0; memcpy(&nRigidBodies, ptr, 4); ptr += 4;
    printf("RigidBody (Bone) Count : %d\n", nRigidBodies);

    if (ptr > targetPtr)
    {
        printf("UnpackSkeletonDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
        return ptr;
    }

    for (int i = 0; i < nRigidBodies; i++)
    {
        printf("Rigid Body (Bone) %d:\n", i);
        ptr = UnpackRigidBodyDescription(ptr, targetPtr, major, minor);
        if (ptr > targetPtr)
        {
            printf("UnpackSkeletonDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
            return ptr;
        }
    }
    return ptr;
}



char * UnpackForcePlateDescription(char * ptr,  char* targetPtr,int major, int minor)
{
    if ((major >= 3)||(major == 0))
    {
        // ID
        int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
        printf("ID : %d\n", ID);

        // Serial Number
        char strSerialNo[128];
        strcpy_s(strSerialNo, ptr);
        ptr += strlen(ptr) + 1;
        printf("Serial Number : %s\n", strSerialNo);

        // Dimensions
        float fWidth = 0; memcpy(&fWidth, ptr, 4); ptr += 4;
        printf("Width : %3.2f\n", fWidth);

        float fLength = 0; memcpy(&fLength, ptr, 4); ptr += 4;
        printf("Length : %3.2f\n", fLength);

        // Origin
        float fOriginX = 0; memcpy(&fOriginX, ptr, 4); ptr += 4;
        float fOriginY = 0; memcpy(&fOriginY, ptr, 4); ptr += 4;
        float fOriginZ = 0; memcpy(&fOriginZ, ptr, 4); ptr += 4;
        printf("Origin : %3.2f,  %3.2f,  %3.2f\n", fOriginX, fOriginY, fOriginZ);

        // Calibration Matrix
        const int kCalMatX = 12;
        const int kCalMatY = 12;
        float fCalMat[kCalMatX][kCalMatY];
        printf("Cal Matrix\n");
        for (int calMatX = 0; calMatX < kCalMatX; ++calMatX)
        {
            printf("  ");
            for (int calMatY = 0; calMatY < kCalMatY; ++calMatY)
            {
                memcpy(&fCalMat[calMatX][calMatY], ptr, 4); ptr += 4;
                printf("%3.3e ", fCalMat[calMatX][calMatY]);
            }
            printf("\n");
        }

        // Corners
        const int kCornerX = 4;
        const int kCornerY = 3;
        float fCorners[kCornerX][kCornerY] = { {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0} };
        printf("Corners\n");
        for (int cornerX = 0; cornerX < kCornerX; ++cornerX)
        {
            printf("  ");
            for (int cornerY = 0; cornerY < kCornerY; ++cornerY)
            {
                memcpy(&fCorners[cornerX][cornerY], ptr, 4); ptr += 4;
                printf("%3.3e ", fCorners[cornerX][cornerY]);
            }
            printf("\n");
        }

        // Plate Type
        int iPlateType = 0; memcpy(&iPlateType, ptr, 4); ptr += 4;
        printf("Plate Type : %d\n", iPlateType);

        // Channel Data Type
        int iChannelDataType = 0; memcpy(&iChannelDataType, ptr, 4); ptr += 4;
        printf("Channel Data Type : %d\n", iChannelDataType);

        // Number of Channels
        int nChannels = 0; memcpy(&nChannels, ptr, 4); ptr += 4;
        printf("  Number of Channels : %d\n", nChannels);
        if (ptr > targetPtr)
        {
            printf("UnpackSkeletonDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
            return ptr;
        }

        for (int chNum = 0; chNum < nChannels; ++chNum)
        {
            char szName[MAX_NAMELENGTH];
            strcpy_s(szName, ptr);
            int nDataBytes = (int)strlen(szName) + 1;
            ptr += nDataBytes;
            printf("    Channel Name %d: %s\n",chNum,  szName);
            if (ptr > targetPtr)
            {
                printf("UnpackSkeletonDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
                return ptr;
            }
        }
    }
    return ptr;
}


char * UnpackDeviceDescription(char * ptr,  char* targetPtr,int major, int minor)
{
    if ((major >= 3) ||(major == 0))
    {
        int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
        printf("ID : %d\n", ID);

        // Name
        char strName[128];
        strcpy_s(strName, ptr);
        ptr += strlen(ptr) + 1;
        printf("Device Name :       %s\n", strName);

        // Serial Number
        char strSerialNo[128];
        strcpy_s(strSerialNo, ptr);
        ptr += strlen(ptr) + 1;
        printf("Serial Number :     %s\n", strSerialNo);

        int iDeviceType = 0; memcpy(&iDeviceType, ptr, 4); ptr += 4;
        printf("Device Type :        %d\n", iDeviceType);

        int iChannelDataType = 0; memcpy(&iChannelDataType, ptr, 4); ptr += 4;
        printf("Channel Data Type : %d\n", iChannelDataType);

        int nChannels = 0; memcpy(&nChannels, ptr, 4); ptr += 4;
        printf("Number of Channels : %d\n", nChannels);
        char szChannelName[MAX_NAMELENGTH];

        if (ptr > targetPtr)
        {
            printf("UnpackDeviceDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
            return ptr;
        }

        for (int chNum = 0; chNum < nChannels; ++chNum) {
            strcpy_s(szChannelName, ptr);
            ptr += strlen(ptr) + 1;
            printf("  Channel Name %d:     %s\n", chNum, szChannelName);
            if (ptr > targetPtr)
            {
                printf("UnpackDeviceDescription: UNPACK ERROR DETECTED: STOPPING DECODE\n");
                return ptr;
            }
        }
    }

    return ptr;
}

//
// UnpackCameraDescription
//
char * UnpackCameraDescription(char * ptr,  char* targetPtr,int major, int minor)
{

    // Name
    char szName[MAX_NAMELENGTH];
    strcpy_s(szName, ptr);
    ptr += strlen(ptr) + 1;
    MakeAlnum(szName, MAX_NAMELENGTH);
    printf("Camera Name  : %s\n", szName);

    // Pos
    float cameraPosition[3];
    memcpy(cameraPosition+0, ptr, 4); ptr += 4;
    memcpy(cameraPosition+1, ptr, 4); ptr += 4;
    memcpy(cameraPosition+2, ptr, 4); ptr += 4;
    printf("  Position   : %3.2f, %3.2f, %3.2f\n",
        cameraPosition[0], cameraPosition[1],
        cameraPosition[2]);

    // Ori
    float cameraOriQuat[4]; // x, y, z, w
    memcpy(cameraOriQuat + 0, ptr, 4); ptr += 4;
    memcpy(cameraOriQuat + 1, ptr, 4); ptr += 4;
    memcpy(cameraOriQuat + 2, ptr, 4); ptr += 4;
    memcpy(cameraOriQuat + 3, ptr, 4); ptr += 4;
    printf("  Orientation: %3.2f, %3.2f, %3.2f, %3.2f\n", 
        cameraOriQuat[0], cameraOriQuat[1], 
        cameraOriQuat[2], cameraOriQuat[3] );

    return ptr;
}


char * UnpackFrameData(char * inptr, int nBytes, int major, int minor)
{
    char * ptr = inptr;
    printf("MoCap Frame Begin\n---------------- - \n" );

    ptr = UnpackFramePrefixData(ptr, major, minor);

    ptr = UnpackMarkersetData(ptr, major, minor);

    ptr = UnpackRigidBodyData(ptr, major, minor);

    ptr = UnpackSkeletonData(ptr, major, minor);

    ptr = UnpackLabeledMarkerData(ptr, major, minor);

    ptr = UnpackForcePlateData(ptr, major, minor);

    ptr = UnpackDeviceData(ptr, major, minor);

    ptr = UnpackFrameSuffixData(ptr, major, minor);
    printf("MoCap Frame End\n---------------- - \n" );
        return ptr;
}

char* UnpackFramePrefixData(char* ptr, int major, int minor)
{
    // Next 4 Bytes is the frame number
    int frameNumber = 0; memcpy(&frameNumber, ptr, 4); ptr += 4;
    printf("Frame # : %d\n", frameNumber);
    return ptr;
}



char* UnpackMarkersetData(char* ptr, int major, int minor)
{
    // First 4 Bytes is the number of data sets (markersets, rigidbodies, etc)
    int nMarkerSets = 0; memcpy(&nMarkerSets, ptr, 4); ptr += 4;
    printf("Marker Set Count : %3.1d\n", nMarkerSets);

    // Loop through number of marker sets and get name and data
    for (int i = 0; i < nMarkerSets; i++)
    {
        // Markerset name
        char szName[MAX_NAMELENGTH];
        strcpy_s(szName, ptr);
        int nDataBytes = (int)strlen(szName) + 1;
        ptr += nDataBytes;
        MakeAlnum(szName, MAX_NAMELENGTH);
        printf("Model Name       : %s\n", szName);

        // marker data
        int nMarkers = 0; memcpy(&nMarkers, ptr, 4); ptr += 4;
        printf("Marker Count     : %3.1d\n", nMarkers);

        for (int j = 0; j < nMarkers; j++)
        {
            float x = 0; memcpy(&x, ptr, 4); ptr += 4;
            float y = 0; memcpy(&y, ptr, 4); ptr += 4;
            float z = 0; memcpy(&z, ptr, 4); ptr += 4;
            printf("  Marker %3.1d : [x=%3.2f,y=%3.2f,z=%3.2f]\n", j, x, y, z);
        }
    }

    // Loop through unlabeled markers
    int nOtherMarkers = 0; memcpy(&nOtherMarkers, ptr, 4); ptr += 4;
    // OtherMarker list is Deprecated
    printf("Unlabeled Markers Count : %d\n", nOtherMarkers);
    for (int j = 0; j < nOtherMarkers; j++)
    {
        float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
        float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
        float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;

        // Deprecated
        printf("  Marker %3.1d : [x=%3.2f,y=%3.2f,z=%3.2f]\n", j, x, y, z);
    }

    return ptr;
}


char* UnpackRigidBodyData(char* ptr, int major, int minor)
{
    // Loop through rigidbodies
    int nRigidBodies = 0;
    memcpy(&nRigidBodies, ptr, 4); ptr += 4;
    printf("Rigid Body Count : %3.1d\n", nRigidBodies);
    for (int j = 0; j < nRigidBodies; j++)
    {
        // Rigid body position and orientation 
        int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
        float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
        float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
        float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
        float qx = 0; memcpy(&qx, ptr, 4); ptr += 4;
        float qy = 0; memcpy(&qy, ptr, 4); ptr += 4;
        float qz = 0; memcpy(&qz, ptr, 4); ptr += 4;
        float qw = 0; memcpy(&qw, ptr, 4); ptr += 4;
        printf("  RB: %3.1d ID : %3.1d\n",j, ID);
        printf("    pos: [%3.2f,%3.2f,%3.2f]\n", x, y, z);
        printf("    ori: [%3.2f,%3.2f,%3.2f,%3.2f]\n", qx, qy, qz, qw);

        // Marker positions removed as redundant (since they can be derived from RB Pos/Ori plus initial offset) in NatNet 3.0 and later to optimize packet size
        if (major < 3)
        {
            // Associated marker positions
            int nRigidMarkers = 0;  memcpy(&nRigidMarkers, ptr, 4); ptr += 4;
            printf("Marker Count: %d\n", nRigidMarkers);
            int nBytes = nRigidMarkers * 3 * sizeof(float);
            float* markerData = (float*)malloc(nBytes);
            memcpy(markerData, ptr, nBytes);
            ptr += nBytes;

            // NatNet Version 2.0 and later
            if (major >= 2)
            {
                // Associated marker IDs
                nBytes = nRigidMarkers * sizeof(int);
                int* markerIDs = (int*)malloc(nBytes);
                memcpy(markerIDs, ptr, nBytes);
                ptr += nBytes;

                // Associated marker sizes
                nBytes = nRigidMarkers * sizeof(float);
                float* markerSizes = (float*)malloc(nBytes);
                memcpy(markerSizes, ptr, nBytes);
                ptr += nBytes;

                for (int k = 0; k < nRigidMarkers; k++)
                {
                    printf("  Marker %d: id=%d  size=%3.1f  pos=[%3.2f,%3.2f,%3.2f]\n",
					k, markerIDs[k], markerSizes[k],
					markerData[k * 3], markerData[k * 3 + 1], markerData[k * 3 + 2]);
                }

                if (markerIDs)
                    free(markerIDs);
                if (markerSizes)
                    free(markerSizes);

            }
            // Print marker positions for all rigid bodies
            else
            {
                int k3;
                for (int k = 0; k < nRigidMarkers; k++)
                {
                    k3 = k * 3;
                    printf("  Marker %d: pos = [%3.2f,%3.2f,%3.2f]\n",
					k, markerData[k3], markerData[k3 + 1], markerData[k3 + 2]);
                }
            }

            if (markerData)
                free(markerData);
        }

        // NatNet version 2.0 and later
        if ((major >= 2) ||(major == 0))
        {
            // Mean marker error
            float fError = 0.0f; memcpy(&fError, ptr, 4); ptr += 4;
            printf("    Mean marker err: %3.2f\n", fError);
        }

        // NatNet version 2.6 and later
        if (((major == 2) && (minor >= 6)) || (major > 2) || (major == 0))
        {
            // params
            short params = 0; memcpy(&params, ptr, 2); ptr += 2;
            bool bTrackingValid = params & 0x01; // 0x01 : rigid body was successfully tracked in this frame
            printf("    Tracking Valid: %s\n", (bTrackingValid) ? "True" : "False");
        }

    } // Go to next rigid body


    return ptr;
}


char* UnpackSkeletonData(char* ptr, int major, int minor)
{
    // Skeletons (NatNet version 2.1 and later)
    if (((major == 2) && (minor > 0)) || (major > 2))
    {
        int nSkeletons = 0;
        memcpy(&nSkeletons, ptr, 4); ptr += 4;
        printf("Skeleton Count : %d\n", nSkeletons);

        // Loop through skeletons
        for (int j = 0; j < nSkeletons; j++)
        {
            // skeleton id
            int skeletonID = 0;
            memcpy(&skeletonID, ptr, 4); ptr += 4;
            printf("  Skeleton %d ID=%d : BEGIN\n", j, skeletonID);

            // Number of rigid bodies (bones) in skeleton
            int nRigidBodies = 0;
            memcpy(&nRigidBodies, ptr, 4); ptr += 4;
            printf("  Rigid Body Count : %d\n", nRigidBodies);

            // Loop through rigid bodies (bones) in skeleton
            for (int k = 0; k < nRigidBodies; k++)
            {
                // Rigid body position and orientation
                int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
                float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
                float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
                float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
                float qx = 0; memcpy(&qx, ptr, 4); ptr += 4;
                float qy = 0; memcpy(&qy, ptr, 4); ptr += 4;
                float qz = 0; memcpy(&qz, ptr, 4); ptr += 4;
                float qw = 0; memcpy(&qw, ptr, 4); ptr += 4;
                printf("    RB: %3.1d ID : %3.1d\n",k, ID);
                printf("      pos: [%3.2f,%3.2f,%3.2f]\n", x, y, z);
                printf("      ori: [%3.2f,%3.2f,%3.2f,%3.2f]\n", qx, qy, qz, qw);

                // Mean marker error (NatNet version 2.0 and later)
                if (major >= 2)
                {
                    float fError = 0.0f; memcpy(&fError, ptr, 4); ptr += 4;
                    printf("    Mean marker error: %3.2f\n", fError);
                }

                // Tracking flags (NatNet version 2.6 and later)
                if (((major == 2) && (minor >= 6)) || (major > 2) || (major == 0))
                {
                    // params
                    short params = 0; memcpy(&params, ptr, 2); ptr += 2;
                    bool bTrackingValid = params & 0x01; // 0x01 : rigid body was successfully tracked in this frame
                }
            } // next rigid body
            printf("  Skeleton %d ID=%d : END\n", j, skeletonID);

        } // next skeleton
    }

    return ptr;
}


char* UnpackLabeledMarkerData(char* ptr, int major, int minor)
{
    // labeled markers (NatNet version 2.3 and later)
// labeled markers - this includes all markers: Active, Passive, and 'unlabeled' (markers with no asset but a PointCloud ID)
    if (((major == 2) && (minor >= 3)) || (major > 2))
    {
        int nLabeledMarkers = 0;
        memcpy(&nLabeledMarkers, ptr, 4); ptr += 4;
        printf("Labeled Marker Count : %d\n", nLabeledMarkers);

        // Loop through labeled markers
        for (int j = 0; j < nLabeledMarkers; j++)
        {
            // id
            // Marker ID Scheme:
            // Active Markers:
            //   ID = ActiveID, correlates to RB ActiveLabels list
            // Passive Markers: 
            //   If Asset with Legacy Labels
            //      AssetID 	(Hi Word)
            //      MemberID	(Lo Word)
            //   Else
            //      PointCloud ID
            int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;
            int modelID, markerID;
            DecodeMarkerID(ID, &modelID, &markerID);


            // x
            float x = 0.0f; memcpy(&x, ptr, 4); ptr += 4;
            // y
            float y = 0.0f; memcpy(&y, ptr, 4); ptr += 4;
            // z
            float z = 0.0f; memcpy(&z, ptr, 4); ptr += 4;
            // size
            float size = 0.0f; memcpy(&size, ptr, 4); ptr += 4;

            // NatNet version 2.6 and later
            if (((major == 2) && (minor >= 6)) || (major > 2) || (major == 0))
            {
                // marker params
                short params = 0; memcpy(&params, ptr, 2); ptr += 2;
                bool bOccluded = (params & 0x01) != 0;     // marker was not visible (occluded) in this frame
                bool bPCSolved = (params & 0x02) != 0;     // position provided by point cloud solve
                bool bModelSolved = (params & 0x04) != 0;  // position provided by model solve
                if ((major >= 3) || (major == 0))
                {
                    bool bHasModel = (params & 0x08) != 0;     // marker has an associated asset in the data stream
                    bool bUnlabeled = (params & 0x10) != 0;    // marker is 'unlabeled', but has a point cloud ID
                    bool bActiveMarker = (params & 0x20) != 0; // marker is an actively labeled LED marker
                }

            }

            // NatNet version 3.0 and later
            float residual = 0.0f;
            if ((major >= 3) || (major == 0))
            {
                // Marker residual
                memcpy(&residual, ptr, 4); ptr += 4;
            }

            printf("  ID  : [MarkerID: %d] [ModelID: %d]\n", markerID, modelID);
            printf("    pos : [%3.2f,%3.2f,%3.2f]\n", x, y, z);
            printf("    size: [%3.2f]\n", size);
            printf("    err:  [%3.2f]\n", residual);
        }
    }
    return ptr;
}


char* UnpackForcePlateData(char* ptr, int major, int minor)
{
    // Force Plate data (NatNet version 2.9 and later)
    if (((major == 2) && (minor >= 9)) || (major > 2))
    {
        int nForcePlates;
        const int kNFramesShowMax = 4;
        memcpy(&nForcePlates, ptr, 4); ptr += 4;
        printf("Force Plate Count: %d\n", nForcePlates);
        for (int iForcePlate = 0; iForcePlate < nForcePlates; iForcePlate++)
        {
            // ID
            int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;

            // Channel Count
            int nChannels = 0; memcpy(&nChannels, ptr, 4); ptr += 4;

            printf("Force Plate %3.1d ID: %3.1d Num Channels: %3.1d\n", iForcePlate, ID, nChannels);

            // Channel Data
            for (int i = 0; i < nChannels; i++)
            {
                printf("  Channel %d : ", i);
                int nFrames = 0; memcpy(&nFrames, ptr, 4); ptr += 4;
                printf("  %3.1d Frames - Frame Data: ", nFrames);

                // Force plate frames
                int nFramesShow = min(nFrames, kNFramesShowMax);
                for (int j = 0; j < nFrames; j++)
                {
                    float val = 0.0f;  memcpy(&val, ptr, 4); ptr += 4;
                    if(j < nFramesShow)
                        printf("%3.2f   ", val);
                }
                if (nFramesShow < nFrames)
                {
                    printf(" showing %3.1d of %3.1d frames", nFramesShow, nFrames);
                }
                printf("\n");
            }
        }
    }

    return ptr;
}


char* UnpackDeviceData(char* ptr, int major, int minor)
{
    // Device data (NatNet version 3.0 and later)
    if (((major == 2) && (minor >= 11)) || (major > 2))
    {
        const int kNFramesShowMax = 4;
        int nDevices;
        memcpy(&nDevices, ptr, 4); ptr += 4;
        printf("Device Count: %d\n", nDevices);
        for (int iDevice = 0; iDevice < nDevices; iDevice++)
        {
            // ID
            int ID = 0; memcpy(&ID, ptr, 4); ptr += 4;

            // Channel Count
            int nChannels = 0; memcpy(&nChannels, ptr, 4); ptr += 4;

            printf("Device %3.1d      ID: %3.1d Num Channels: %3.1d\n",iDevice, ID,nChannels);

            // Channel Data
            for (int i = 0; i < nChannels; i++)
            {
                printf("  Channel %d : ", i);
                int nFrames = 0; memcpy(&nFrames, ptr, 4); ptr += 4;
                printf("  %3.1d Frames - Frame Data: ", nFrames);
                // Device frames
                int nFramesShow = min(nFrames, kNFramesShowMax);
                for (int j = 0; j < nFrames; j++)
                {
                    float val = 0.0f;  memcpy(&val, ptr, 4); ptr += 4;
                    if (j < nFramesShow)
                        printf("%3.2f   ", val);
                }
                if (nFramesShow < nFrames)
                {
                    printf(" showing %3.1d of %3.1d frames", nFramesShow, nFrames);
                }
                printf("\n");
            }
        }
    }

    return ptr;
}


char* UnpackFrameSuffixData(char* ptr, int major, int minor)
{

    // software latency (removed in version 3.0)
    if (major < 3)
    {
        float softwareLatency = 0.0f; memcpy(&softwareLatency, ptr, 4);	ptr += 4;
        printf("software latency : %3.3f\n", softwareLatency);
    }

    // timecode
    unsigned int timecode = 0; 	memcpy(&timecode, ptr, 4);	ptr += 4;
    unsigned int timecodeSub = 0; memcpy(&timecodeSub, ptr, 4); ptr += 4;
    char szTimecode[128] = "";
    TimecodeStringify(timecode, timecodeSub, szTimecode, 128);

    // timestamp
    double timestamp = 0.0f;

    // NatNet version 2.7 and later - increased from single to double precision
    if (((major == 2) && (minor >= 7)) || (major > 2))
    {
        memcpy(&timestamp, ptr, 8); ptr += 8;
    }
    else
    {
        float fTemp = 0.0f;
        memcpy(&fTemp, ptr, 4); ptr += 4;
        timestamp = (double)fTemp;
    }
    printf("Timestamp : %3.3f\n", timestamp);

    // high res timestamps (version 3.0 and later)
    if ((major >= 3) || (major == 0))
    {
        uint64_t cameraMidExposureTimestamp = 0;
        memcpy(&cameraMidExposureTimestamp, ptr, 8); ptr += 8;
        printf("Mid-exposure timestamp         : %" PRIu64"\n", cameraMidExposureTimestamp);

        uint64_t cameraDataReceivedTimestamp = 0;
        memcpy(&cameraDataReceivedTimestamp, ptr, 8); ptr += 8;
        printf("Camera data received timestamp : %" PRIu64"\n", cameraDataReceivedTimestamp);

        uint64_t transmitTimestamp = 0;
        memcpy(&transmitTimestamp, ptr, 8); ptr += 8;
        printf("Transmit timestamp             : %" PRIu64"\n", transmitTimestamp);
    }

    // frame params
    short params = 0;  memcpy(&params, ptr, 2); ptr += 2;
    bool bIsRecording = (params & 0x01) != 0;                  // 0x01 Motive is recording
    bool bTrackedModelsChanged = (params & 0x02) != 0;         // 0x02 Actively tracked model list has changed


    // end of data tag
    int eod = 0; memcpy(&eod, ptr, 4); ptr += 4;
    /*End Packet*/

    return ptr;
}

char * UnpackPacketHeader(char * ptr, int &messageID, int& nBytes, int& nBytesTotal)
{
    // First 2 Bytes is message ID
    memcpy(&messageID, ptr, 2); ptr += 2;

    // Second 2 Bytes is the size of the packet
    memcpy(&nBytes, ptr, 2); ptr += 2;
    nBytesTotal = nBytes + 4;
    return ptr;
}


// *********************************************************************
//
//  Unpack:
//      Receives pointer to bytes that represent a packet of data
//
//      There are lots of print statements that show what
//      data is being stored
//
//      Most memcpy functions will assign the data to a variable.
//      Use this variable at your descretion. 
//      Variables created for storing data do not exceed the 
//      scope of this function. 
//
// *********************************************************************
char * Unpack(char * pData)
{
    // Checks for NatNet Version number. Used later in function. 
    // Packets may be different depending on NatNet version.
    int major = gNatNetVersion[0];
    int minor = gNatNetVersion[1];

    char *ptr = pData;

    printf("Begin Packet\n-------\n");
    printf("NatNetVersion %d %d %d %d\n", 
        gNatNetVersion[0], gNatNetVersion[1],
        gNatNetVersion[2], gNatNetVersion[3] );


    int messageID = 0;
    int nBytes = 0;
    int nBytesTotal = 0;
    ptr = UnpackPacketHeader(ptr, messageID, nBytes, nBytesTotal);

    switch (messageID)
    {
    case NAT_CONNECT:
        printf("Message ID  : %d NAT_CONNECT\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_SERVERINFO:
        printf("Message ID  : %d NAT_SERVERINFO\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_REQUEST:
        printf("Message ID  : %d NAT_REQUEST\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_RESPONSE:
        printf("Message ID  : %d NAT_RESPONSE\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_REQUEST_MODELDEF:
        printf("Message ID  : %d NAT_REQUEST_MODELDEF\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_MODELDEF:
    // Data Descriptions
    {
        printf("Message ID  : %d NAT_MODELDEF\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        ptr = UnpackDescription(ptr, nBytes, major, minor);
    }
    break;
    case NAT_REQUEST_FRAMEOFDATA:
        printf("Message ID  : %d NAT_REQUEST_FRAMEOFDATA\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_FRAMEOFDATA:
        // FRAME OF MOCAP DATA packet
    {
        printf("Message ID  : %d NAT_FRAMEOFDATA\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        ptr = UnpackFrameData(ptr, nBytes, major, minor);
    }
    break;
    case NAT_MESSAGESTRING:
        printf("Message ID  : %d NAT_MESSAGESTRING\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_DISCONNECT:
        printf("Message ID  : %d NAT_DISCONNECT\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_KEEPALIVE:
        printf("Message ID  : %d NAT_KEEPALIVE\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    case NAT_UNRECOGNIZED_REQUEST:
        printf("Message ID  : %d NAT_UNRECOGNIZED_REQUEST\n", messageID);
        printf("Packet Size : %d\n", nBytes);
        break;
    default:
    {
        printf("Unrecognized Packet Type.\n");
        printf("Message ID  : %d\n", messageID);
        printf("Packet Size : %d\n", nBytes);
    }
    break;
    }
    
    printf("End Packet\n-------------\n");
    
    // check for full packet processing
    long long nBytesProcessed = (long long)ptr - (long long)pData;
    if (nBytesTotal != nBytesProcessed) {
        printf("WARNING: %d expected but %lld bytes processed\n",
            nBytesTotal, nBytesProcessed);
        if (nBytesTotal > nBytesProcessed) {
            int count = 0, countLimit = 8*25;// put on 8 byte boundary
            printf("Sample of remaining bytes:\n");
            char* ptr_start = ptr;
            int nCount = nBytesProcessed;
            char tmpChars[9] = { "        " };
            int charPos = ((long long)ptr % 8);
            char tmpChar;
            // add spaces for first row
            if (charPos > 0)
            {
                for (int i = 0; i < charPos; ++i)
                {
                    printf("   ");
                    if (i == 4)
                    {
                        printf("    ");
                    }
                }
            }
            countLimit = countLimit - (charPos+1);
            while (nCount < nBytesTotal)
            {
                tmpChar = ' ';
                if (isalnum(*ptr)) {
                    tmpChar = *ptr;
                }
                tmpChars[charPos] = tmpChar;
                printf("%2.2x ", (unsigned char)*ptr);
                ptr += 1;
                charPos = (long long)ptr % 8;
                if (charPos == 0)
                {
                    printf("    ");
                    for (int i = 0; i < 8; ++i)
                    {
                        printf("%c", tmpChars[i]);
                    }
                    printf("\n");
                }
                else if (charPos == 4)
                {
                    printf("    ");
                }
                if (++count > countLimit)
                {
                    break;
                }
                ++nCount;
            }
            if ((long long)ptr % 8)
            {
                printf("\n");
            }
        }
    }
    // return the beginning of the possible next packet
    // assuming no additional termination
    ptr = pData + nBytesTotal;
    return ptr;
}
