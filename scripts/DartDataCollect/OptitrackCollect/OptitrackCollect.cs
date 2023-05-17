using System.Collections;
using System.Text.Json;
using NatNetML;


/* OptitrackCollect.cs
 * 
 * This program is a sample console application which uses the managed NatNet assembly (NatNetML.dll) for receiving NatNet data
 * from a tracking server (e.g. Motive) and outputting them in every 200 mocap frames. This is provided mainly for demonstration purposes,
 * and thus, the program is designed at its simpliest approach. The program connects to a server application at a localhost IP address
 * (127.0.0.1) using Multicast connection protocol.
 *  
 * You may integrate this program into your applications if needed. This program is not designed to take account for latency/frame build up
 * when tracking a high number of assets. For more robust and comprehensive use of the NatNet assembly, refer to the provided WinFormSample project.
 * 
 *  Note: The NatNet .NET assembly is derived from the native NatNetLib.dll library, so make sure the NatNetLib.dll is linked to your application
 *        along with the NatNetML.dll file.  
 * 
 *  List of Output Data:
 *  ====================
 *      - Markers Data : Prints out total number of markers reconstructed in the scene.
 *      - Rigid Body Data : Prints out position and orientation data
 *      - Skeleton Data : Prints out only the position of the hip segment
 *      - Force Plate Data : Prints out only the first subsample data per each mocap frame
 * 
 */


OptitrackCollect.OptitrackCollect.Main(args);


namespace OptitrackCollect
{
    public class OptitrackCollect
    {
        /*  [NatNet] Network connection configuration    */
        private static NatNetClientML? mNatNet;    // The client instance

        /*  List for saving each of datadescriptors */
        private static List<DataDescriptor> mDataDescriptor = new();

        /*  Lists and Hashtables for saving data descriptions   */
        private static readonly Hashtable mHtSkelRBs = new();
        private static readonly List<RigidBody> mRigidBodies = new();
        private static readonly List<Skeleton> mSkeletons = new();
        private static readonly List<ForcePlate> mForcePlates = new();
        private static readonly List<Device> mDevices = new();
        private static readonly List<Camera> mCameras = new();

        /*  boolean value for detecting change in asset */
        private static bool mAssetChanged = false;

        /*  for writing to output files */
        private static StreamWriter? mTrajWriter;

        public static void Main(string[] args)
        {
            string strLocalIP = "127.0.0.1";   // Local IP address (string)
            string strServerIP = "127.0.0.1";  // Server IP address (string)
            string strOutputDir = @".\";
            ConnectionType connectionType = ConnectionType.Multicast; // Multicast or Unicast mode

            Console.WriteLine("OptitrackCollect managed client application starting...\n");
            if (args.Length == 0)
            {
                Console.WriteLine("  command line options: \n");
                Console.WriteLine("  OptitrackCollect [server_ip_address [client_ip_address [Unicast/Multicast [output_dir]]]] \n");
                Console.WriteLine("  Examples: \n");
                Console.WriteLine(@"    OptitrackCollect 127.0.0.1 127.0.0.1 Unicast C:\Users\Administrator\Desktop\dartdata\dataset0" + "\n");
                Console.WriteLine(@"    OptitrackCollect 127.0.0.1 127.0.0.1 m C:\Users\Administrator\Desktop\dartdata\dataset0" + "\n");
                Console.WriteLine("\n");
            }
            else
            {
                strServerIP = args[0];
                if (args.Length > 1)
                {
                    strLocalIP = args[1];
                    if (args.Length > 2)
                    {
                        connectionType = ConnectionType.Multicast; // Multicast or Unicast mode
                        string res = args[2][..1];
                        string res2 = res.ToLower();
                        if (res2 == "u")
                            connectionType = ConnectionType.Unicast;
                        if (args.Length > 3)
                            strOutputDir = args[3];
                    }
                }
            }

            string cmdline = "OptitrackCollect " + strServerIP + " " + strLocalIP + " ";
            if (connectionType == ConnectionType.Multicast)
                cmdline += "Multicast";
            else
                cmdline += "Unicast";
            Console.WriteLine("Using: " + cmdline + "\n");

            /*  [NatNet] Initialize client object and connect to the server  */
            // Initialize a NatNetClient object and connect to a server.
            ConnectToServer(strServerIP, strLocalIP, connectionType);

            Console.WriteLine("============================ SERVER DESCRIPTOR ================================\n");
            /*  [NatNet] Confirming Server Connection. Instantiate the server descriptor object and obtain the server description. */
            bool connectionConfirmed = FetchServerDescriptor();    // To confirm connection, request server description data

            if (connectionConfirmed)                         // Once the connection is confirmed.
            {
                Console.WriteLine("============================= DATA DESCRIPTOR =================================\n");
                Console.WriteLine("Now Fetching the Data Descriptor.\n");
                FetchDataDescriptor();                  //Fetch and parse data descriptor

                Console.WriteLine("============================= FRAME OF DATA ===================================\n");
                Console.WriteLine("Now Fetching the Frame Data\n");

                /*  [NatNet] Assigning a event handler function for fetching frame data each time a frame is received   */
                mNatNet!.OnFrameReady += new FrameReadyEventHandler(FetchFrameData);

                Console.WriteLine("Success: Data Port Connected \n");

                Console.WriteLine("======================== STREAMING IN (PRESS ESC TO EXIT) =====================\n");
            }

            /*  Open files for writing output */
            Directory.CreateDirectory(strOutputDir);
            using (mTrajWriter = File.CreateText(Path.Combine(strOutputDir, "optitrack.txt")))
            {
                //Process.Start("cmd", "/c echo hello");
                while (!(Console.KeyAvailable && Console.ReadKey().Key == ConsoleKey.Escape))
                {
                    // Continuously listening for Frame data
                    // Enter ESC to exit

                    // Exception handler for updated assets list.
                    if (mAssetChanged)
                    {
                        Console.WriteLine("\n===============================================================================\n");
                        Console.WriteLine("Change in the list of the assets. Refetching the descriptions");

                        /*  Clear out existing lists */
                        mDataDescriptor.Clear();
                        mHtSkelRBs.Clear();
                        mRigidBodies.Clear();
                        mSkeletons.Clear();
                        mForcePlates.Clear();

                        /* [NatNet] Re-fetch the updated list of descriptors  */
                        FetchDataDescriptor();
                        Console.WriteLine("===============================================================================\n");
                        mAssetChanged = false;
                    }
                }
                /*  [NatNet] Disabling data handling function   */
                mNatNet!.OnFrameReady -= FetchFrameData;

                /*  Clearing Saved Descriptions */
                mRigidBodies.Clear();
                mSkeletons.Clear();
                mHtSkelRBs.Clear();
                mForcePlates.Clear();
                mNatNet.Disconnect();
            }
        }

        /// <summary>
        /// [NatNet] parseFrameData will be called when a frame of Mocap
        /// data has is received from the server application.
        ///
        /// Note: This callback is on the network service thread, so it is
        /// important to return from this function quickly as possible 
        /// to prevent incoming frames of data from buffering up on the
        /// network socket.
        ///
        /// Note: "data" is a reference structure to the current frame of data.
        /// NatNet re-uses this same instance for each incoming frame, so it should
        /// not be kept (the values contained in "data" will become replaced after
        /// this callback function has exited).
        /// </summary>
        /// <param name="data">The actual frame of mocap data</param>
        /// <param name="client">The NatNet client instance</param>
        private static void FetchFrameData(FrameOfMocapData data, NatNetClientML client)
        {
            /*  Exception handler for cases where assets are added or removed.
                Data description is re-obtained in the main function so that contents
                in the frame handler is kept minimal. */
            if (data.bTrackingModelsChanged || data.nRigidBodies != mRigidBodies.Count || data.nSkeletons != mSkeletons.Count || data.nForcePlates != mForcePlates.Count)
                mAssetChanged = true;

            /*  Processing and ouputting frame data every 200th frame.
                This conditional statement is included in order to simplify the program output */
            if (data.iFrame % 100 == 0)
            {
                if (!data.bRecording)
                    Console.WriteLine("Frame #{0} Received:", data.iFrame);
                else if (data.bRecording)
                    Console.WriteLine("[Recording] Frame #{0} Received:", data.iFrame);

                ProcessFrameData(data, true);
            }
            else
            {
                ProcessFrameData(data, false);
            }
        }

        private static void ProcessFrameData(FrameOfMocapData data, bool doPrint)
        {
            /*  Parsing Rigid Body Frame Data   */
            for (int i = 0; i < mRigidBodies.Count; i++)
            {
                int rbID = mRigidBodies[i].ID;              // Fetching rigid body IDs from the saved descriptions

                for (int j = 0; j < data.nRigidBodies; j++)
                {
                    if (rbID == data.RigidBodies[j].ID)      // When rigid body ID of the descriptions matches rigid body ID of the frame data.
                    {
                        RigidBody rb = mRigidBodies[i];                // Saved rigid body descriptions
                        RigidBodyData rbData = data.RigidBodies[j];    // Received rigid body descriptions

                        if (rbData.Tracked)
                        {
                            if (doPrint)
                            {
                                Console.WriteLine("\tRigidBody ({0}):", rb.Name);
                                Console.WriteLine("\t\tpos ({0:N3}, {1:N3}, {2:N3})", rbData.x, rbData.y, rbData.z);
                                Console.WriteLine("\t\tori ({0:N3}, {1:N3}, {2:N3}, {3:N3})", rbData.qx, rbData.qy, rbData.qz, rbData.qw);
                            }
                            double timeDelaySeconds = mNatNet!.SecondsSinceHostTimestamp(data.CameraMidExposureTimestamp);
                            DateTime trajtime = DateTime.UtcNow - TimeSpan.FromSeconds(timeDelaySeconds);
                            OptitrackPoint optitrackPoint = new
                            (
                                object_id: rb.Name,
                                data: new
                                (
                                    position: new
                                    (
                                        x: rbData.x,
                                        y: rbData.y,
                                        z: rbData.z
                                    ),
                                    rotation: new
                                    (
                                        x: rbData.qx,
                                        y: rbData.qy,
                                        z: rbData.qz,
                                        w: rbData.qw
                                    ),
                                    timestamp: (trajtime - DateTime.UnixEpoch).TotalSeconds
                                )
                            );
                            mTrajWriter!.WriteLine(JsonSerializer.Serialize(optitrackPoint));
                        }
                        else if (doPrint)
                        {
                            Console.WriteLine("\t{0} is not tracked in current frame", rb.Name);
                        }
                    }
                }
            }

            /* Parsing Skeleton Frame Data  */
            for (int i = 0; i < mSkeletons.Count; i++)      // Fetching skeleton IDs from the saved descriptions
            {
                int sklID = mSkeletons[i].ID;

                for (int j = 0; j < data.nSkeletons; j++)
                {
                    if (sklID == data.Skeletons[j].ID)      // When skeleton ID of the description matches skeleton ID of the frame data.
                    {
                        Skeleton skl = mSkeletons[i];              // Saved skeleton descriptions
                        SkeletonData sklData = data.Skeletons[j];  // Received skeleton frame data

                        if (doPrint)
                        {
                            Console.WriteLine("\tSkeleton ({0}):", skl.Name);
                            Console.WriteLine("\t\tSegment count: {0}", sklData.nRigidBodies);
                        }

                        /*  Now, for each of the skeleton segments  */
                        for (int k = 0; k < sklData.nRigidBodies; k++)
                        {
                            RigidBodyData boneData = sklData.RigidBodies[k];

                            /*  Decoding skeleton bone ID   */
                            int skeletonID = HighWord(boneData.ID);
                            int rigidBodyID = LowWord(boneData.ID);
                            int uniqueID = skeletonID * 1000 + rigidBodyID;
                            int key = uniqueID.GetHashCode();

                            RigidBody? bone = (RigidBody?)mHtSkelRBs[key];   //Fetching saved skeleton bone descriptions

                            //Outputting only the hip segment data for the purpose of this sample.
                            if (k == 0 && doPrint)
                                Console.WriteLine("\t\t{0:N3}: pos({1:N3}, {2:N3}, {3:N3})", bone!.Name, boneData.x, boneData.y, boneData.z);
                        }
                    }
                }
            }

            /*  Parsing Force Plate Frame Data  */
            for (int i = 0; i < mForcePlates.Count; i++)
            {
                int fpID = mForcePlates[i].ID;                  // Fetching force plate IDs from the saved descriptions

                for (int j = 0; j < data.nForcePlates; j++)
                {
                    if (fpID == data.ForcePlates[j].ID)         // When force plate ID of the descriptions matches force plate ID of the frame data.
                    {
                        ForcePlate fp = mForcePlates[i];                // Saved force plate descriptions
                        ForcePlateData fpData = data.ForcePlates[i];    // Received forceplate frame data

                        if (doPrint)
                            Console.WriteLine("\tForce Plate ({0}):", fp.Serial);

                        // Here we will be printing out only the first force plate "subsample" (index 0) that was collected with the mocap frame.
                        for (int k = 0; k < fpData.nChannels; k++)
                        {
                            if (doPrint)
                                Console.WriteLine("\t\tChannel {0}: {1}", fp.ChannelNames[k], fpData.ChannelData[k].Values[0]);
                        }
                    }
                }
            }
            if (doPrint)
                Console.WriteLine("\n");
        }

        private static void ConnectToServer(string serverIPAddress, string localIPAddress, ConnectionType connectionType)
        {
            /*  [NatNet] Instantiate the client object  */
            mNatNet = new NatNetClientML();

            /*  [NatNet] Checking verions of the NatNet SDK library  */
            int[] verNatNet = mNatNet.NatNetVersion(); // Saving NatNet SDK version number
            Console.WriteLine("NatNet SDK Version: {0}.{1}.{2}.{3}", verNatNet[0], verNatNet[1], verNatNet[2], verNatNet[3]);

            /*  [NatNet] Connecting to the Server    */

            NatNetClientML.ConnectParams connectParams = new()
            {
                ConnectionType = connectionType,
                ServerAddress = serverIPAddress,
                LocalAddress = localIPAddress
            };

            Console.WriteLine("\nConnecting...");
            Console.WriteLine("\tServer IP Address: {0}", serverIPAddress);
            Console.WriteLine("\tLocal IP address : {0}", localIPAddress);
            Console.WriteLine("\tConnection Type  : {0}", connectionType);
            Console.WriteLine("\n");

            mNatNet.Connect(connectParams);
        }

        private static bool FetchServerDescriptor()
        {
            ServerDescription m_ServerDescriptor = new();
            int errorCode = mNatNet!.GetServerDescription(m_ServerDescriptor);

            if (errorCode == 0)
            {
                Console.WriteLine("Success: Connected to the server\n");
                ParseSeverDescriptor(m_ServerDescriptor);
                return true;
            }
            else
            {
                Console.WriteLine("Error: Failed to connect. Check the connection settings.");
                Console.WriteLine("Program terminated (Enter ESC to exit)");
                return false;
            }
        }

        private static void ParseSeverDescriptor(ServerDescription server)
        {
            Console.WriteLine("Server Info:");
            Console.WriteLine("\tHost               : {0}", server.HostComputerName);
            Console.WriteLine("\tApplication Name   : {0}", server.HostApp);
            Console.WriteLine("\tApplication Version: {0}.{1}.{2}.{3}", server.HostAppVersion[0], server.HostAppVersion[1], server.HostAppVersion[2], server.HostAppVersion[3]);
            Console.WriteLine("\tNatNet Version     : {0}.{1}.{2}.{3}\n", server.NatNetVersion[0], server.NatNetVersion[1], server.NatNetVersion[2], server.NatNetVersion[3]);
        }

        private static void FetchDataDescriptor()
        {
            /*  [NatNet] Fetch Data Descriptions. Instantiate objects for saving data descriptions and frame data    */
            bool result = mNatNet!.GetDataDescriptions(out mDataDescriptor);
            if (result)
            {
                Console.WriteLine("Success: Data Descriptions obtained from the server.");
                ParseDataDescriptor(mDataDescriptor);
            }
            else
            {
                Console.WriteLine("Error: Could not get the Data Descriptions");
            }
            Console.WriteLine("\n");
        }

        private static void ParseDataDescriptor(List<DataDescriptor> description)
        {
            //  [NatNet] Request a description of the Active Model List from the server. 
            //  This sample will list only names of the data sets, but you can access 
            int numDataSet = description.Count;
            Console.WriteLine("Total {0} data sets in the capture:", numDataSet);

            for (int i = 0; i < numDataSet; ++i)
            {
                int dataSetType = description[i].type;
                // Parse Data Descriptions for each data sets and save them in the delcared lists and hashtables for later uses.
                switch (dataSetType)
                {
                    case (int)DataDescriptorType.eMarkerSetData:
                        MarkerSet mkset = (MarkerSet)description[i];
                        Console.WriteLine("\tMarkerSet ({0})", mkset.Name);
                        break;


                    case (int)DataDescriptorType.eRigidbodyData:
                        RigidBody rb = (RigidBody)description[i];
                        Console.WriteLine("\tRigidBody ({0})", rb.Name);

                        // Saving Rigid Body Descriptions
                        mRigidBodies.Add(rb);
                        break;


                    case (int)DataDescriptorType.eSkeletonData:
                        Skeleton skl = (Skeleton)description[i];
                        Console.WriteLine("\tSkeleton ({0}), Bones:", skl.Name);

                        //Saving Skeleton Descriptions
                        mSkeletons.Add(skl);

                        // Saving Individual Bone Descriptions
                        for (int j = 0; j < skl.nRigidBodies; j++)
                        {

                            Console.WriteLine("\t\t{0}. {1}", j + 1, skl.RigidBodies[j].Name);
                            int uniqueID = skl.ID * 1000 + skl.RigidBodies[j].ID;
                            int key = uniqueID.GetHashCode();
                            mHtSkelRBs.Add(key, skl.RigidBodies[j]); //Saving the bone segments onto the hashtable
                        }
                        break;


                    case (int)DataDescriptorType.eForcePlateData:
                        ForcePlate fp = (ForcePlate)description[i];
                        Console.WriteLine("\tForcePlate ({0})", fp.Serial);

                        // Saving Force Plate Channel Names
                        mForcePlates.Add(fp);

                        for (int j = 0; j < fp.ChannelCount; j++)
                            Console.WriteLine("\t\tChannel {0}: {1}", j + 1, fp.ChannelNames[j]);
                        break;

                    case (int)DataDescriptorType.eDeviceData:
                        Device dd = (Device)description[i];
                        Console.WriteLine("\tDeviceData ({0})", dd.Serial);

                        // Saving Device Data Channel Names
                        mDevices.Add(dd);

                        for (int j = 0; j < dd.ChannelCount; j++)
                            Console.WriteLine("\t\tChannel {0}: {1}", j + 1, dd.ChannelNames[j]);
                        break;

                    case (int)DataDescriptorType.eCameraData:
                        // Saving Camera Names
                        Camera camera = (Camera)description[i];
                        Console.WriteLine("\tCamera: ({0})", camera.Name);

                        // Saving Force Plate Channel Names
                        mCameras.Add(camera);
                        break;


                    default:
                        // When a Data Set does not match any of the descriptions provided by the SDK.
                        Console.WriteLine("\tError: Invalid Data Set - dataSetType = " + dataSetType);
                        break;
                }
            }
        }

        private static int LowWord(int number)
        {
            return number & 0xFFFF;
        }

        private static int HighWord(int number)
        {
            return (number >> 16) & 0xFFFF;
        }
    } // End. ManagedClient class
} // End. NatNetML Namespace
