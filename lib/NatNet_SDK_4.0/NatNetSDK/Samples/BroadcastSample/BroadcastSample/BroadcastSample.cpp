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

/*
BroadcastClient.cpp

Simple application illustrating how to use remote record trigger in Motive using
XML formatted UDP broadcast packets.
*/

#include <stdio.h>
#include <tchar.h>
#include <conio.h>
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma warning( disable : 4996 )

bool IPAddress_StringToAddr(char *szNameOrAddress, struct in_addr *Address);
int GetLocalIPAddresses(unsigned long Addresses[], int nMax);

// This is the port Motive is sending/listening commands
#define PORT_COMMAND_XML        1512                

SOCKET senderSocket;
SOCKET receiverSocket;

unsigned long ParsePID(char* buf)
{
    int pid = 0;
    char* p = strstr(buf, "ProcessID");
    if(p)
    {
        strlen (p);
        while ( (*p!='=') && (*p!='\0') )
            p++;
        if(*p=='=')
        {
            p++;
            p++;
            sscanf(p, "%d", &pid);
        }
    }
    return (unsigned long)pid;

}

DWORD WINAPI ReceiverListenThread(void* dummy)
{
    char  szData[20000];
    int addr_len = sizeof(struct sockaddr);
    sockaddr_in TheirAddress;

    while (1)
    {
        // Block until we receive a datagram from the network (from anyone including ourselves)
        int nDataBytesReceived = recvfrom(receiverSocket, szData, sizeof(szData), 0, (sockaddr *)&TheirAddress, &addr_len);

        if( (nDataBytesReceived == 0) || (nDataBytesReceived == SOCKET_ERROR) )
            continue;

        unsigned long thisPID = GetCurrentProcessId();
        unsigned long senderPID = ParsePID(szData);
        if (thisPID == senderPID)
        {
            // Received from self - ignoring
            continue;
        }

        // print received data
        printf("\nRECEIVED [from %d.%d.%d.%d:%d] : %s\n",
            TheirAddress.sin_addr.S_un.S_un_b.s_b1, TheirAddress.sin_addr.S_un.S_un_b.s_b2,
            TheirAddress.sin_addr.S_un.S_un_b.s_b3, TheirAddress.sin_addr.S_un.S_un_b.s_b4,
            ntohs(TheirAddress.sin_port), szData);
    }

    return 0;
}

int main(int argc, char* argv[])
{
    int retval;
    WSADATA WsaData; 
    int value = 1;

    if (WSAStartup(0x202, &WsaData) == SOCKET_ERROR)
    {
		printf("[PacketClient] WSAStartup failed (error: %d)\n", WSAGetLastError());
        WSACleanup();
        return 0;
    }


    // CREATE SENDER SOCKET
    struct sockaddr_in sender_addr;     
    in_addr senderAddress;
    char szSenderAddress[128] = "";
    // address
    if(argc>1)
    {
        strcpy_s(szSenderAddress, argv[1]);	// specified on command line
        retval = IPAddress_StringToAddr(szSenderAddress, &senderAddress);
    }
    else
    {
        GetLocalIPAddresses((unsigned long *)&senderAddress, 1);
    }
    sprintf_s(szSenderAddress, "%d.%d.%d.%d", senderAddress.S_un.S_un_b.s_b1, senderAddress.S_un.S_un_b.s_b2, senderAddress.S_un.S_un_b.s_b3, senderAddress.S_un.S_un_b.s_b4);
    printf("Sender Address: %s\n", szSenderAddress);


    // create a blocking, datagram socket
    if ((senderSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == INVALID_SOCKET)	// AF_INTE=IPV4, SOCK_DGRAM=UDP
    {
        printf("[PacketClient] create failed (error: %d)\n", WSAGetLastError());
        return 0;
    }
    // allow re-use
    retval = setsockopt(senderSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&value, sizeof(value));
    if (retval == SOCKET_ERROR)
    {
        printf("[PacketClient] (error: %d)\n", WSAGetLastError());
        closesocket(senderSocket);
        return 0;
    }
    // bind socket
    memset(&sender_addr, 0, sizeof(sender_addr));
    sender_addr.sin_family = AF_INET;
    sender_addr.sin_port = htons(PORT_COMMAND_XML);
    sender_addr.sin_addr = senderAddress;
    //my_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    //my_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(senderSocket, (struct sockaddr *)&sender_addr, sizeof(struct sockaddr)) == SOCKET_ERROR)
    {
        printf("[PacketClient] (error: %d)\n", WSAGetLastError());
        closesocket(senderSocket);
        return 0;
    }
    // set to broadcast mode
    value = 1;
    if (setsockopt(senderSocket, SOL_SOCKET, SO_BROADCAST, (char *)&value, sizeof(value)) == SOCKET_ERROR)
    {
        printf("[PacketClient] (error: %d)\n", WSAGetLastError());
        closesocket(senderSocket);
        return 0;
    }
    // END SENDER SOCKET

 

    // CREATE RECEIVER SOCKET
    struct sockaddr_in receive_addr;     
    in_addr receiveAddress;
    char szReceiveAddress[128] = "";
    // receiver address
	if(argc>2)
	{
		strcpy_s(szReceiveAddress, argv[2]);	// specified on command line
	    retval = IPAddress_StringToAddr(szReceiveAddress, &receiveAddress);
	}
	else
	{
        GetLocalIPAddresses((unsigned long *)&receiveAddress, 1);
        sprintf_s(szReceiveAddress, "%d.%d.%d.%d", receiveAddress.S_un.S_un_b.s_b1, receiveAddress.S_un.S_un_b.s_b2, receiveAddress.S_un.S_un_b.s_b3, receiveAddress.S_un.S_un_b.s_b4);
    }
    sprintf_s(szReceiveAddress, "%d.%d.%d.%d", receiveAddress.S_un.S_un_b.s_b1, receiveAddress.S_un.S_un_b.s_b2, receiveAddress.S_un.S_un_b.s_b3, receiveAddress.S_un.S_un_b.s_b4);
    printf("Receiver Address: %s\n", szReceiveAddress);

    // create a blocking, datagram socket
    if ((receiverSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == INVALID_SOCKET)	// AF_INTE=IPV4, SOCK_DGRAM=UDP
    {
        printf("[PacketClient] create failed (error: %d)\n", WSAGetLastError());
        return 0;
    }
    // allow multiple clients on same machine to re-use address/port
    retval = setsockopt(receiverSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&value, sizeof(value));
    if (retval == SOCKET_ERROR)
    {
        printf("[PacketClient] (error: %d)\n", WSAGetLastError());
        closesocket(receiverSocket);
        return 0;
    }
    // bind socket
    memset(&receive_addr, 0, sizeof(receive_addr));
    receive_addr.sin_family = AF_INET;
    receive_addr.sin_port = htons(PORT_COMMAND_XML);
    receive_addr.sin_addr = receiveAddress;
    //my_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    //my_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(receiverSocket, (struct sockaddr *)&receive_addr, sizeof(struct sockaddr)) == SOCKET_ERROR)
    {
        printf("[PacketClient] (error: %d)\n", WSAGetLastError());
        closesocket(receiverSocket);
        return 0;
    }
    // set to broadcast mode
    if (setsockopt(receiverSocket, SOL_SOCKET, SO_BROADCAST, (char *)&value, sizeof(value)) == SOCKET_ERROR)
    {
        printf("[PacketClient] (error: %d)\n", WSAGetLastError());
        closesocket(receiverSocket);
        return 0;
    }
    // END RECEIVER SOCKET

    
    // startup our "receiver listener" thread
    SECURITY_ATTRIBUTES security_attribs;
    security_attribs.nLength = sizeof(SECURITY_ATTRIBUTES);
    security_attribs.lpSecurityDescriptor = NULL;
    security_attribs.bInheritHandle = TRUE;
    DWORD DataListenThread_ID;
    HANDLE DataListenThread_Handle;
    DataListenThread_Handle = CreateThread( &security_attribs, 0, ReceiverListenThread, NULL, 0, &DataListenThread_ID);

    // destination address
    sockaddr_in ToAddr;
    memset(&ToAddr, 0, sizeof(ToAddr));
    ToAddr.sin_family = AF_INET;        
    ToAddr.sin_port = htons(PORT_COMMAND_XML); 
    ToAddr.sin_addr.s_addr = htonl(INADDR_BROADCAST);


    // XML message format:
    // <?xml version="1.0" encoding="UTF-8" standalone="no" ?><CaptureStart><TimeCode VALUE="12 13 14 14 0 0 1 1"/><Name VALUE="RemoteTriggerTest_take01"/><Notes VALUE=""/><Description VALUE=""/><DatabasePath VALUE="S:/shared/testfolder/"/><PacketID VALUE="0"/></CaptureStart>
    
    // use pid to allow listener to ignore packets from self  
    unsigned long pid = GetCurrentProcessId();
    char szMessage[512];

    printf("\nCommands:\nx : Send XML command to Motive to start recording.\nz : Send XML command to Motive to stop recording.\nq : exit");

    printf("\n\nListening...\n");

    int c;
    bool bExit = false;

    while (!bExit)
    {
        c =_getch();
        switch(c)
        {
        case 'x':
            // broadcast XML formatted "record start" command
            // standard format
            //sprintf(szMessage, "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?><CaptureStart><TimeCode VALUE=\"12 13 14 15 0 0 1 1\"/><Name VALUE=\"RemoteTriggerTest_take01\"/><Notes VALUE=\"\"/><Description VALUE=\"\"/><DatabasePath VALUE=\"S:/shared/testfolder/\"/><PacketID VALUE=\"0\"/><ProcessID VALUE=\"%d\"/></CaptureStart>", pid);
            // no-header format
            sprintf(szMessage, "<CaptureStart><TimeCode VALUE=\"\"/><Name VALUE=\"RemoteTriggerTest_take01\"/><Notes VALUE=\"\"/><Description VALUE=\"\"/><DatabasePath VALUE=\"S:/shared/testfolder/\"/><PacketID VALUE=\"0\"/><ProcessID VALUE=\"%d\"/></CaptureStart>", pid);
            retval = sendto(receiverSocket, (char *)szMessage, 4 + strlen(szMessage), 0, (sockaddr *)&ToAddr, sizeof(ToAddr));
            if(retval != SOCKET_ERROR)
            {
                printf("\nSENT (to %d.%d.%d.%d:%d) : %s\n",
                    ToAddr.sin_addr.S_un.S_un_b.s_b1, ToAddr.sin_addr.S_un.S_un_b.s_b2,
                    ToAddr.sin_addr.S_un.S_un_b.s_b3, ToAddr.sin_addr.S_un.S_un_b.s_b4,
                    ntohs(ToAddr.sin_port), szMessage);
            }
            break;	
        case 'z':
            // broadcast XML formatted "record stop" command
            sprintf(szMessage, "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?><CaptureStop><TimeCode VALUE=\"12 13 14 15 0 0 1 1\"/><Name VALUE=\"RemoteTriggerTest_take01\"/><Notes VALUE=\"\"/><Description VALUE=\"\"/><DatabasePath VALUE=\"S:/shared/testfolder/\"/><PacketID VALUE=\"0\"/><ProcessID VALUE=\"%d\"/></CaptureStop>", pid);
            retval = sendto(receiverSocket, (char *)szMessage, 4 + strlen(szMessage), 0, (sockaddr *)&ToAddr, sizeof(ToAddr));
            if(retval != SOCKET_ERROR)
            {
                printf("\nSENT (to %d.%d.%d.%d:%d) : %s\n",
                    ToAddr.sin_addr.S_un.S_un_b.s_b1, ToAddr.sin_addr.S_un.S_un_b.s_b2,
                    ToAddr.sin_addr.S_un.S_un_b.s_b3, ToAddr.sin_addr.S_un.S_un_b.s_b4,
                    ntohs(ToAddr.sin_port), szMessage);
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
