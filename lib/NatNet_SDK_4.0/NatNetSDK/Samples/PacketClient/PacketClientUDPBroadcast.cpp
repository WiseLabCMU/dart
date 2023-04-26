/* 
Copyright © 2013 NaturalPoint Inc.

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

Listen to Motive Record Broacast messages

*/

#include <stdio.h>
#include <tchar.h>
#include <conio.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma warning( disable : 4996 )

#define MAX_NAMELENGTH              256
#define MAX_PACKETSIZE				100000	// max size of packet (actual packet size is dynamic)
#define SIO_RCVALL _WSAIOW(IOC_VENDOR,1)

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
        unsigned long  lData[MAX_PACKETSIZE/4];
        float          fData[MAX_PACKETSIZE/4];
        sSender        Sender;
    } Data;                                 // Payload

} sPacket;


bool IPAddress_StringToAddr(char *szNameOrAddress, struct in_addr *Address);
int GetLocalIPAddresses(unsigned long Addresses[], int nMax);

#define PORT_COMMAND            1512
#define PORT_DATA  			    1511                // Default multicast group

SOCKET CommandSocket;
in_addr ServerAddress;
sockaddr_in HostAddr;  

// command response listener thread
DWORD WINAPI CommandListenThread(void* dummy)
{
    int addr_len;
    int nDataBytesReceived;
    char str[256];
    sockaddr_in TheirAddress;
    sPacket PacketIn;
    addr_len = sizeof(struct sockaddr);

	char sz[MAX_PACKETSIZE];
	char sz2[MAX_PACKETSIZE];

    while (1)
    {
        // blocking
		nDataBytesReceived = recvfrom( CommandSocket,(char *)sz, MAX_PACKETSIZE, 0, (struct sockaddr *)&TheirAddress, &addr_len);

		int ipoffset = 0; // udp
		//int ipoffset = 28; // raw / ipheader
		if (nDataBytesReceived < (ipoffset+10))
			continue;

		if( (strncmp(&(sz[ipoffset]), "<?xml", 5)!=0) && (strncmp(&(sz[ipoffset]), "Motiv", 5)!=0) )
			continue;

		strcpy(sz2, &sz[ipoffset]);
		printf("\nReceived Broadcast from %d.%d.%d.%d: %s\n",
			TheirAddress.sin_addr.S_un.S_un_b.s_b1, TheirAddress.sin_addr.S_un.S_un_b.s_b2,
			TheirAddress.sin_addr.S_un.S_un_b.s_b3, TheirAddress.sin_addr.S_un.S_un_b.s_b4,
			sz2);

		strcpy(sz, "");
		strcpy(sz2, "");
    }

    return 0;
}

SOCKET CreateCommandSocket(unsigned long IP_Address, unsigned short uPort)
{
    struct sockaddr_in my_addr;     
    static unsigned long ivalue;
    static unsigned long bFlag;
    int nlengthofsztemp = 64;  
    SOCKET sockfd;

    // Create a blocking, datagram socket
	//if ((sockfd=socket(AF_INET, SOCK_DGRAM, 0)) == INVALID_SOCKET)
	if ((sockfd=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == INVALID_SOCKET)
	//if ((sockfd=socket(AF_INET, SOCK_RAW, IPPROTO_IP)) == INVALID_SOCKET)
    {
		printf("[PacketClient] WSAStartup failed (error: %d)\n", WSAGetLastError());
        return -1;
    }

	// allow multiple clients on same machine to use address/port
	int value = 1;
	int retval = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, (char*)&value, sizeof(value));
	if (retval == SOCKET_ERROR)
	{
		closesocket(DataSocket);
		return -1;
	}



    // bind socket
    memset(&my_addr, 0, sizeof(my_addr));
    my_addr.sin_family = AF_INET;
    //my_addr.sin_port = htons(uPort);
    //my_addr.sin_addr.S_un.S_addr = IP_Address;
	my_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	my_addr.sin_port = htons(PORT_COMMAND); // Want to receive broadcasts to this port number on the LAN.

    if (bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) == SOCKET_ERROR)
    {
		printf("[PacketClient] Bind Failed (error: %d)\n", WSAGetLastError());
        closesocket(sockfd);
        return -1;
    }

	// testing - listen to ALL broadcast messages
	unsigned int  optval = 1;
	DWORD dwBytesRet;
	DWORD  dwIoControlCode= SIO_RCVALL;
	if (WSAIoctl(sockfd, dwIoControlCode, &optval, sizeof(optval), NULL, 0, &dwBytesRet, NULL, NULL) == SOCKET_ERROR)
	{
		printf("[PacketClient] ReceiveAll Failed (error: %d)\n", WSAGetLastError());
	}
    
	// set to broadcast mode
    ivalue = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_BROADCAST, (char *)&ivalue, sizeof(ivalue)) == SOCKET_ERROR)
    {
        closesocket(sockfd);
        return -1;
    }


    return sockfd;
}

int main(int argc, char* argv[])
{
    int retval;
    char szMyIPAddress[128] = "";
    char szServerIPAddress[128] = "";
    in_addr MyAddress, MultiCastAddress;
    WSADATA WsaData; 
    int optval = 0x100000;
    int optval_size = 4;

    if (WSAStartup(0x202, &WsaData) == SOCKET_ERROR)
    {
		printf("[PacketClient] WSAStartup failed (error: %d)\n", WSAGetLastError());
        WSACleanup();
        return 0;
    }

	// server address
	if(argc>1)
	{
		strcpy_s(szServerIPAddress, argv[1]);	// specified on command line
	    retval = IPAddress_StringToAddr(szServerIPAddress, &ServerAddress);
	}
	else
	{
        GetLocalIPAddresses((unsigned long *)&ServerAddress, 1);
        sprintf_s(szServerIPAddress, "%d.%d.%d.%d", ServerAddress.S_un.S_un_b.s_b1, ServerAddress.S_un.S_un_b.s_b2, ServerAddress.S_un.S_un_b.s_b3, ServerAddress.S_un.S_un_b.s_b4);
	}

    // client address
	if(argc>2)
	{
		strcpy_s(szMyIPAddress, argv[2]);	// specified on command line
	    retval = IPAddress_StringToAddr(szMyIPAddress, &MyAddress);
	}
	else
	{
        GetLocalIPAddresses((unsigned long *)&MyAddress, 1);
        sprintf_s(szMyIPAddress, "%d.%d.%d.%d", MyAddress.S_un.S_un_b.s_b1, MyAddress.S_un.S_un_b.s_b2, MyAddress.S_un.S_un_b.s_b3, MyAddress.S_un.S_un_b.s_b4);
    }

    // create "Command" socket
	int port = 0;
	//int port = 1510;
    CommandSocket = CreateCommandSocket(MyAddress.S_un.S_addr,port);
    if(CommandSocket == -1)
    {
        // error
    }
    else
    {
        // [optional] set to non-blocking
        //u_long iMode=1;
        //ioctlsocket(CommandSocket,FIONBIO,&iMode); 
        // set buffer
        setsockopt(CommandSocket, SOL_SOCKET, SO_RCVBUF, (char *)&optval, 4);
        getsockopt(CommandSocket, SOL_SOCKET, SO_RCVBUF, (char *)&optval, &optval_size);
        if (optval != 0x100000)
        {
            // err - actual size...
        }
        // startup our "Command Listener" thread
        SECURITY_ATTRIBUTES security_attribs;
        security_attribs.nLength = sizeof(SECURITY_ATTRIBUTES);
        security_attribs.lpSecurityDescriptor = NULL;
        security_attribs.bInheritHandle = TRUE;
        DWORD CommandListenThread_ID;
        HANDLE CommandListenThread_Handle;
        CommandListenThread_Handle = CreateThread( &security_attribs, 0, CommandListenThread, NULL, 0, &CommandListenThread_ID);
    }


    while (!bExit)
    {
        c =_getch();
        switch(c)
        {
        case 'q':
            bExit = true;		
            break;	
        default:
            break;
        }
    }

    return 0;
}


// convert ipp address string to addr
bool IPAddress_StringToAddr(char *szNameOrAddress, struct in_addr *Address)
{
	int retVal;
	struct sockaddr_in saGNI;
	char hostName[256];
	char servInfo[256];
	u_short port;
	port = 0;

	// Set up sockaddr_in structure which is passed to the getnameinfo function
	saGNI.sin_family = AF_INET;
	saGNI.sin_addr.s_addr = inet_addr(szNameOrAddress);
	saGNI.sin_port = htons(port);

	// getnameinfo
	if ((retVal = getnameinfo((SOCKADDR *)&saGNI, sizeof(sockaddr), hostName, 256, servInfo, 256, NI_NUMERICSERV)) != 0)
	{
        printf("[PacketClient] GetHostByAddr failed. Error #: %ld\n", WSAGetLastError());
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
    char* port = "0";
    
    if(GetComputerName(szMyName, &NameLength) != TRUE)
    {
        printf("[PacketClient] get computer name  failed. Error #: %ld\n", WSAGetLastError());
        return 0;       
    };

	memset(&aiHints, 0, sizeof(aiHints));
	aiHints.ai_family = AF_INET;
	aiHints.ai_socktype = SOCK_DGRAM;
	aiHints.ai_protocol = IPPROTO_UDP;
	if ((retVal = getaddrinfo(szMyName, port, &aiHints, &aiList)) != 0) 
	{
        printf("[PacketClient] getaddrinfo failed. Error #: %ld\n", WSAGetLastError());
        return 0;
	}

    memcpy(&addr, aiList->ai_addr, aiList->ai_addrlen);
    freeaddrinfo(aiList);
    Addresses[0] = addr.sin_addr.S_un.S_addr;

    return 1;
}

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


