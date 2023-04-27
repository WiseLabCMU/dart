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

using System;
using System.IO;
using System.Collections.Generic;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Net;
using System.Threading;
using System.Runtime.InteropServices;

using NatNetML;
using System.Reflection;

using System.Net.NetworkInformation;
using System.Text;

/*
 *
 * Simple C# .NET sample showing how to use the NatNet managed assembly (NatNETML.dll).
 * 
 * It is designed to illustrate using NatNet.  There are some inefficiencies to keep the
 * code as simple to read as possible.
 * 
 * Sections marked with a [NatNet] are NatNet related and should be implemented in your code.
 * 
 * This sample uses the Microsoft Chart Controls for Microsoft .NET for graphing, which
 * requires the following assemblies:
 *   - System.Windows.Forms.DataVisualization.Design.dll
 *   - System.Windows.Forms.DataVisualization.dll
 * Make sure you have these in your path when building and redistributing.
 * 
 */


namespace WinFormTestApp
{

    public partial class Form1 : Form
    {
        // Helper class for discovering NatNet servers.
        private NatNetServerDiscovery m_Discovery = new NatNetServerDiscovery();

        // [NatNet] Our NatNet object
        private NatNetML.NatNetClientML m_NatNet;

        // [NatNet] Our NatNet Frame of Data object
        private NatNetML.FrameOfMocapData m_FrameOfData = new NatNetML.FrameOfMocapData();

        // Time that has passed since the NatNet server transmitted m_FrameOfData.
        private double m_FrameOfDataTransitLatency;

        // Total Latency : Time between mid-camera-exposure to client available
        private double m_TotalLatency;

        // [NatNet] Description of the Active Model List from the server (e.g. Motive)
        NatNetML.ServerDescription desc = new NatNetML.ServerDescription();

        // [NatNet] Queue holding our incoming mocap frames the NatNet server (e.g. Motive)
        private Queue<NatNetML.FrameOfMocapData> m_FrontQueue = new Queue<NatNetML.FrameOfMocapData>();
        private Queue<NatNetML.FrameOfMocapData> m_BackQueue = new Queue<NatNetML.FrameOfMocapData>();
        private static object FrontQueueLock = new object();
        private static object BackQueueLock = new object();

        // Records the age of each frame in m_FrameQueue at the time it arrived.
        private Queue<double> m_FrameTransitLatencies = new Queue<double>();
        private Queue<double> m_TotalLatencies = new Queue<double>();

        // data grid
        Hashtable htMarkers = new Hashtable();
        
        List<RigidBody> mRigidBodies = new List<RigidBody>();
        Hashtable htRigidBodies = new Hashtable();
        Hashtable htRigidBodyMarkers = new Hashtable();
        
        Hashtable htSkels = new Hashtable();
        Hashtable htSkelRBs = new Hashtable();

        List<ForcePlate> mForcePlates = new List<ForcePlate>();
        Hashtable htForcePlates = new Hashtable();
        
        List<Device> mDevices = new List<Device>();
        Hashtable htDevices = new Hashtable();
        private int mLastRowCount;
        private int minGridHeight;

        // graph
        const int GraphFrames = 10000;
        int m_iLastFrameNumber = 0;
        const int maxSeriesCount = 10;

        // frame timing information
        double m_fLastFrameTimestamp = 0.0f;
        QueryPerfCounter m_FramePeriodTimer = new QueryPerfCounter();
        QueryPerfCounter m_ProcessingTimer = new QueryPerfCounter();
        private double interframeDuration;
        private int droppedFrameIndicator = 0;

        // server information
        string mServerIP = "";
        double m_ServerFramerate = 1.0f;
        float m_ServerToMillimeters = 1.0f;
        int m_UpAxis = 1;   // 0=x, 1=y, 2=z (Y default)
        int mAnalogSamplesPerMocpaFrame = 0;
        int mDroppedFrames = 0;
        int mLastFrame = 0;
        int mUIBusyCount = 0;
        bool mNeedTrackingListUpdate = false;

        // UI updating
        private delegate void OutputMessageCallback(string strMessage);
        private bool mPaused = false;
        delegate void UpdateUICallback();
        bool mApplicationRunning = true;
        Thread UIUpdateThread;

        // polling
        delegate void PollCallback();
        Thread pollThread;
        bool mPolling = false;

        // ping time
        delegate void PingCallback();
        Thread UpdatePingTimeThread;
        private double mLastPingTimeMs;

        // recording
        bool mRecording = false;
        TextWriter mWriter;

        // auto-connect
        bool mWantAutoconnect = false;
        bool mServerDetected= false;
        bool mServerEstablished = false;
        string mDetectedLocalIP = "";
        string mDetectedServerIP = "";


        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
             // Show available ip addresses of this machine
            String strMachineName = Dns.GetHostName();
            IPHostEntry ipHost = Dns.GetHostByName(strMachineName);
            foreach (IPAddress ip in ipHost.AddressList)
            {
                string strIP = ip.ToString();
                comboBoxLocal.Items.Add(strIP);
            }
            int selected = comboBoxLocal.Items.Add("127.0.0.1");
            //comboBoxLocal.SelectedItem = comboBoxLocal.Items[selected];


            // create NatNet client
            int iResult = CreateClient();

            // create graph
            chart1.Series.Clear();
            for (int i = 0; i < maxSeriesCount; i++)
            {
                System.Windows.Forms.DataVisualization.Charting.Series series = chart1.Series.Add("Series" + i.ToString());
                series.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.FastLine;
                chart1.Series[i].Points.Clear();
            }
            chart1.ChartAreas[0].CursorX.IsUserSelectionEnabled = true;

            // DataGrid 
            // enable double buffering on DataGridView to optimize cell redraws
            Type dgvType = dataGridView1.GetType();
            System.Reflection.PropertyInfo pi = dgvType.GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic);
            pi.SetValue(dataGridView1, true, null);
            // preserve height
            minGridHeight = dataGridView1.Height;

            // create and run an Update UI thread
            UpdateUICallback d = new UpdateUICallback(UpdateUI);
            UIUpdateThread = new Thread(() =>
            {
                while (mApplicationRunning)
                {
                    try
                    {
                        this.Invoke(d);
                        Thread.Sleep(15);
                    }
                    catch (System.Exception ex)
                    {
                        OutputMessage(ex.Message);
                        break;
                    }
                }
            });
            UIUpdateThread.Start();

            // (optional) create and run a polling thread for polling-driven data access option (instead of event callback driven )
            PollCallback pd = new PollCallback(PollData);
            pollThread = new Thread(() =>
            {
                while (mPolling)
                {
                    try
                    {
                        pd.Invoke();
                        Thread.Sleep(15);
                    }
                    catch (System.Exception ex)
                    {
                        OutputMessage(ex.Message);
                        break;
                    }
                }
            });

            /*
            // create and run a ping time thread
            PingCallback pingCallback = new PingCallback(UpdatePing);
            UpdatePingTimeThread = new Thread(() =>
            {
                while (mApplicationRunning)
                {
                    try
                    {
                        pingCallback.Invoke();
                        Thread.Sleep(30);
                    }
                    catch (System.Exception ex)
                    {
                        OutputMessage(ex.Message);
                        break;
                    }
                }
            });
            UpdatePingTimeThread.Start();
            */
            

            // Auto-connect to first detected Motive
            m_Discovery.OnServerDiscovered += delegate (NatNetML.DiscoveredServer server)
            {
                OutputMessage(String.Format(
                    "Discovered server: {0} {1}.{2} at {3} (local interface: {4})",
                    server.ServerDesc.HostApp,
                    server.ServerDesc.HostAppVersion[0],
                    server.ServerDesc.HostAppVersion[1],
                    server.ServerAddress,
                    server.LocalAddress
                ));

                if (!mServerDetected)
                {
                    mDetectedLocalIP = server.LocalAddress.ToString();
                    mDetectedServerIP = server.ServerAddress.ToString();
                    mServerDetected = true;
                }
            };
            m_Discovery.StartDiscovery();

        }

        /// <summary>
        /// Create a new NatNet client, which manages all communication with the NatNet server (e.g. Motive)
        /// </summary>
        /// <param name="iConnectionType">0 = Multicast, 1 = Unicast</param>
        /// <returns></returns>
        private int CreateClient()
        {
            // release any previous instance
            if (m_NatNet != null)
            {
                m_NatNet.Disconnect();
            }

            // [NatNet] create a new NatNet instance
            m_NatNet = new NatNetML.NatNetClientML();

            // [NatNet] set a "Frame Ready" callback function (event handler) handler that will be
            // called by NatNet when NatNet receives a frame of data from the server application
            m_NatNet.OnFrameReady += new NatNetML.FrameReadyEventHandler(m_NatNet_OnFrameReady);

            /*
            // [NatNet] for testing only - event signature format required by some types of .NET applications (e.g. MatLab)
            m_NatNet.OnFrameReady2 += new FrameReadyEventHandler2(m_NatNet_OnFrameReady2);
            */

            // [NatNet] print version info
            int[] ver = new int[4];
            ver = m_NatNet.NatNetVersion();
            String strVersion = String.Format("NatNet Version : {0}.{1}.{2}.{3}", ver[0], ver[1], ver[2], ver[3]);
            OutputMessage(strVersion);

            return 0;
        }

        /// <summary>
        /// Connect to a NatNet server (e.g. Motive)
        /// </summary>
        private void Connect()
        {
            if (comboBoxLocal.SelectedItem == null)
                return;

            if (textBoxServer.Text.Length == 0)
                return;

            // [NatNet] connect to a NatNet server
            int returnCode = 0;
            string strLocalIP = comboBoxLocal.SelectedItem.ToString();
            string strServerIP = textBoxServer.Text;

            NatNetClientML.ConnectParams connectParams = new NatNetClientML.ConnectParams();
            if(RadioUnicast.Checked)
            {
                connectParams.ConnectionType = ConnectionType.Unicast;
            }
            else if(RadioMulticast.Checked)
            {
                connectParams.ConnectionType = ConnectionType.Multicast;
            }
            else if(RadioBroadcast.Checked)
            {
                connectParams.ConnectionType = ConnectionType.Multicast;
                connectParams.MulticastAddress = "255.255.255.255";
            }
            connectParams.ServerAddress = strServerIP;
            connectParams.LocalAddress = strLocalIP;
            
            // Test: subscribed data only:
            //connectParams.SubscribedDataOnly = SubscribeOnlyCheckBox.Checked;
            
            // Test : requested bitstream version
            /*
            connectParams.BitstreamMajor = 1;
            connectParams.BitstreamMinor = 2;
            connectParams.BitstreamRevision = 3;
            connectParams.BitstreamBuild = 4;
            */


            returnCode = m_NatNet.Connect( connectParams );
            if (returnCode == 0)
            {
                OutputMessage( "Initialization Succeeded." );
            }
            else
            {
                OutputMessage("Error Initializing.");
                checkBoxConnect.Checked = false;
            }

            // [NatNet] validate the connection
            returnCode = m_NatNet.GetServerDescription(desc);
            if (returnCode == 0)
            {
                OutputMessage("Connection Succeeded.");
                OutputMessage("   Server App Name: " + desc.HostApp);
                OutputMessage(String.Format("   Server App Version: {0}.{1}.{2}.{3}", desc.HostAppVersion[0], desc.HostAppVersion[1], desc.HostAppVersion[2], desc.HostAppVersion[3]));
                OutputMessage(String.Format("   Server NatNet Version: {0}.{1}.{2}.{3}", desc.NatNetVersion[0], desc.NatNetVersion[1], desc.NatNetVersion[2], desc.NatNetVersion[3]));

                checkBoxConnect.Text = "Disconnect";
                mServerEstablished = true;
                mServerIP = String.Format("{0}.{1}.{2}.{3}", desc.HostComputerAddress[0], desc.HostComputerAddress[1], desc.HostComputerAddress[2], desc.HostComputerAddress[3]);


                // Tracking Tools and Motive report in meters - lets convert to millimeters
                if (desc.HostApp.Contains("TrackingTools") || desc.HostApp.Contains("Motive"))
                    m_ServerToMillimeters = 1000.0f;

                // [NatNet] [optional] Query mocap server for the current camera framerate
                int nBytes = 0;
                byte[] response = new byte[10000];
                int rc;
                rc = m_NatNet.SendMessageAndWait("FrameRate", out response, out nBytes);
                if (rc == 0)
                {
                    try
                    {
                        m_ServerFramerate = BitConverter.ToSingle(response, 0);
                        OutputMessage(String.Format("   Camera Framerate: {0}", m_ServerFramerate));
                    }
                    catch (System.Exception ex)
                    {
                        OutputMessage(ex.Message);
                    }
                }

                // [NatNet] [optional] Query mocap server for the current analog framerate
                rc = m_NatNet.SendMessageAndWait("AnalogSamplesPerMocapFrame", out response, out nBytes);
                if (rc == 0)
                {
                    try
                    {
                        mAnalogSamplesPerMocpaFrame = BitConverter.ToInt32(response, 0);
                        OutputMessage(String.Format("   Analog Samples Per Camera Frame: {0}", mAnalogSamplesPerMocpaFrame));
                    }
                    catch (System.Exception ex)
                    {
                        OutputMessage(ex.Message);
                    }
                }


                // [NatNet] [optional] Query mocap server for the current up axis
                rc = m_NatNet.SendMessageAndWait("UpAxis", out response, out nBytes);
                if (rc == 0)
                {
                    m_UpAxis = BitConverter.ToInt32(response, 0);
                }


                mDroppedFrames = 0;
                lock(FrontQueueLock)
                {
                    m_FrontQueue.Clear();
                }
                lock (BackQueueLock)
                {
                    m_BackQueue.Clear();
                }
            }
            else
            {
                OutputMessage("Error Connecting.");
                checkBoxConnect.Checked = false;
                checkBoxConnect.Text = "Connect";
            }

        }

        private void Disconnect()
        {
            // [NatNet] disconnect
            // optional : for unicast clients only - notify Motive we are disconnecting
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc;
            rc = m_NatNet.SendMessageAndWait("Disconnect", out response, out nBytes);
            if (rc == 0)
            {

            }
            // shutdown our client socket
            m_NatNet.Disconnect();
            checkBoxConnect.Text = "Connect";
        }

        private void checkBoxConnect_CheckedChanged(object sender, EventArgs e)
        {
            if (checkBoxConnect.Checked)
            {
                Connect();
            }
            else
            {
                Disconnect();
            }
        }

        private void OutputMessage(string strMessage)
        {
            if (mPaused)
                return;

            if(!mApplicationRunning)
                return;

            if (this.listView1.InvokeRequired)
            {
                // It's on a different thread, so use Invoke
                OutputMessageCallback d = new OutputMessageCallback(OutputMessage);
                this.Invoke(d, new object[] { strMessage });
            }
            else
            {
                // It's on the same thread, no need for Invoke
                DateTime d = DateTime.Now;
                String strTime = String.Format("{0}:{1}:{2}:{3}", d.Hour, d.Minute, d.Second, d.Millisecond);
                ListViewItem item = new ListViewItem(strTime, 0);
                item.SubItems.Add(strMessage);
                listView1.Items.Add(item);
            }
        }

        private RigidBody FindRB(int id, int parentID = -2)
        {
            foreach (RigidBody rb in mRigidBodies)
            {
                if (rb.ID == id)
                {
                    if(parentID != -2)
                    {
                        if(rb.parentID == parentID)
                            return rb;
                    }
                    else
                    {
                        return rb;
                    }
                }
            }
            return null;
        }

        /// <summary>
        /// Redraw the graph using the data of the selected cell in the spreadsheet
        /// </summary>
        /// <param name="iFrame">Frame ID of mocap data</param>
        private void UpdateChart(long iFrame)
        {
            // Lets only show 500 frames at a time
            iFrame = iFrame % GraphFrames;

            // clear graph if we've wrapped, allow for fudge
            if ((m_iLastFrameNumber - iFrame) > 400)
            {
                for (int i = 0; i < chart1.Series.Count; i++)
                    chart1.Series[i].Points.Clear();
            }

            for (int i = 0; i < dataGridView1.SelectedCells.Count; i++)
            {
                // for simple performance only graph maxSeriesCount lines
                if (i >= maxSeriesCount)
                    break;

                DataGridViewCell cell = dataGridView1.SelectedCells[i];
                if (cell.Value == null)
                    return;
                double dValue = 0.0f;
                if (!Double.TryParse(cell.Value.ToString(), out dValue))
                    return;
                chart1.Series[i].Points.AddXY(iFrame, (float)dValue);
            }

            // update red 'cursor' line
            chart1.ChartAreas[0].CursorX.SetCursorPosition(iFrame);

            m_iLastFrameNumber = (int)iFrame;
        }

        /// <summary>
        /// Update the spreadsheet.  
        /// Note: This refresh is quite slow and provided here only as a complete example. 
        /// In a production setting this would be optimized.
        /// </summary>
        private void UpdateDataGrid()
        {
            // update MarkerSet data
            for (int i = 0; i < m_FrameOfData.nMarkerSets; i++)
            {
                NatNetML.MarkerSetData ms = m_FrameOfData.MarkerSets[i];
                for (int j = 0; j < ms.nMarkers; j++)
                {
                    string strUniqueName = ms.MarkerSetName + j.ToString();
                    int key = strUniqueName.GetHashCode();
                    if (htMarkers.Contains(key))
                    {
                        int rowIndex = (int)htMarkers[key];
                        if (rowIndex >= 0)
                        {
                            dataGridView1.Rows[rowIndex].Cells[1].Value = ms.Markers[j].x * m_ServerToMillimeters;
                            dataGridView1.Rows[rowIndex].Cells[2].Value = ms.Markers[j].y * m_ServerToMillimeters;
                            dataGridView1.Rows[rowIndex].Cells[3].Value = ms.Markers[j].z * m_ServerToMillimeters;
                        }
                    }
                }
            }

            // update RigidBody data
            for (int i = 0; i < m_FrameOfData.nRigidBodies; i++)
            {
                NatNetML.RigidBodyData rb = m_FrameOfData.RigidBodies[i];
                int key = rb.ID.GetHashCode();
                int rowIndex = -1;
                if (htRigidBodies.ContainsKey(key))
                {
                    rowIndex = (int)htRigidBodies[key];
                    if (rowIndex >= 0)
                    {
                        bool tracked = rb.Tracked;
                        if (!tracked)
                        {
                            //OutputMessage("RigidBody not tracked in this frame.");
                        }

                        dataGridView1.Rows[rowIndex].Cells[1].Value = rb.x * m_ServerToMillimeters;
                        dataGridView1.Rows[rowIndex].Cells[2].Value = rb.y * m_ServerToMillimeters;
                        dataGridView1.Rows[rowIndex].Cells[3].Value = rb.z * m_ServerToMillimeters;

                        // Convert quaternion to eulers.  Motive coordinate conventions: X(Pitch), Y(Yaw), Z(Roll), Relative, RHS
                        float[] quat = new float[4] { rb.qx, rb.qy, rb.qz, rb.qw };
                        float[] eulers = new float[3];
                        eulers = NatNetClientML.QuatToEuler(quat, NATEulerOrder.NAT_XYZr);
                        double x = RadiansToDegrees(eulers[0]);     // convert to degrees
                        double y = RadiansToDegrees(eulers[1]);
                        double z = RadiansToDegrees(eulers[2]);

                        dataGridView1.Rows[rowIndex].Cells[4].Value = x;
                        dataGridView1.Rows[rowIndex].Cells[5].Value = y;
                        dataGridView1.Rows[rowIndex].Cells[6].Value = z;
                    }
                }
            }

            // update Skeleton data
            for (int i = 0; i < m_FrameOfData.nSkeletons; i++)
            {
                NatNetML.SkeletonData sk = m_FrameOfData.Skeletons[i];
                for (int j = 0; j < sk.nRigidBodies; j++)
                {
                    // note : skeleton rigid body ids are of the form:
                    // parent skeleton ID   : high word (upper 16 bits of int)
                    // rigid body id        : low word  (lower 16 bits of int)
                    NatNetML.RigidBodyData rb = sk.RigidBodies[j];
                    int skeletonID = HighWord(rb.ID);
                    int rigidBodyID = LowWord(rb.ID);
                    int uniqueID = GetUniqueRBKey(skeletonID, rigidBodyID);
                    int key = uniqueID.GetHashCode();
                    int rowIndex = -1;
                    if (htRigidBodies.ContainsKey(key))
                    {
                        rowIndex = (int)htRigidBodies[key];
                        if (rowIndex >= 0)
                        {
                            dataGridView1.Rows[rowIndex].Cells[1].Value = rb.x * m_ServerToMillimeters;
                            dataGridView1.Rows[rowIndex].Cells[2].Value = rb.y * m_ServerToMillimeters;
                            dataGridView1.Rows[rowIndex].Cells[3].Value = rb.z * m_ServerToMillimeters;

                            // Convert quaternion to eulers.  Motive coordinate conventions: X(Pitch), Y(Yaw), Z(Roll), Relative, RHS
                            float[] quat = new float[4] { rb.qx, rb.qy, rb.qz, rb.qw };
                            float[] eulers = new float[3];
                            eulers = NatNetClientML.QuatToEuler(quat, NATEulerOrder.NAT_XYZr);
                            double x = RadiansToDegrees(eulers[0]);     // convert to degrees
                            double y = RadiansToDegrees(eulers[1]);
                            double z = RadiansToDegrees(eulers[2]);

                            dataGridView1.Rows[rowIndex].Cells[4].Value = x;
                            dataGridView1.Rows[rowIndex].Cells[5].Value = y;
                            dataGridView1.Rows[rowIndex].Cells[6].Value = z;

                        }
                    }
                }
            }   // end skeleton update

            // update ForcePlate data
            if (htForcePlates.Count > 0)
            {
                for (int i = 0; i < m_FrameOfData.nForcePlates; i++)
                {
                    NatNetML.ForcePlateData fp = m_FrameOfData.ForcePlates[i];
                    int key = fp.ID.GetHashCode();
                    int rowIndex = (int)htForcePlates[key];
                    if (rowIndex >= 0)
                    {
                        for (int iChannel = 0; iChannel < fp.nChannels; iChannel++)
                        {
                            if (fp.ChannelData[iChannel].nFrames > 0)
                            {
                                int mocapAlignedSubsampleIndex = 0;
                                if (fp.ChannelData[iChannel].nFrames > 1)
                                {
                                    int id = fp.ChannelData[iChannel].nFrames / 2;
                                    int rem = fp.ChannelData[iChannel].nFrames % 2;
                                    mocapAlignedSubsampleIndex = (id + rem) - 1;
                                }

                                dataGridView1.Rows[rowIndex].Cells[iChannel + 1].Value = fp.ChannelData[iChannel].Values[mocapAlignedSubsampleIndex];
                            }
                        }
                    }
                }
            }

            // update Device data
            if (htDevices.Count > 0)
            {
                for (int i = 0; i < m_FrameOfData.nDevices; i++)
                {
                    NatNetML.DeviceData device = m_FrameOfData.Devices[i];
                    int key = device.ID.GetHashCode();
                    int rowIndex = (int)htDevices[key];
                    if (rowIndex >= 0)
                    {
                        int nChannels = Math.Min(dataGridView1.Rows[rowIndex].Cells.Count, device.nChannels);
                        for (int iChannel = 0; iChannel < nChannels; iChannel++)
                        {
                            if (device.ChannelData[iChannel].nFrames > 0)
                            {
                                int mocapAlignedSubsampleIndex = 0;
                                if (device.ChannelData[iChannel].nFrames > 1)
                                {
                                    int id = device.ChannelData[iChannel].nFrames / 2;
                                    int rem = device.ChannelData[iChannel].nFrames % 2;
                                    mocapAlignedSubsampleIndex = (id + rem) - 1;
                                }

                                try
                                {
                                    dataGridView1.Rows[rowIndex].Cells[iChannel + 1].Value = device.ChannelData[iChannel].Values[mocapAlignedSubsampleIndex];
                                }
                                catch (Exception e)
                                {
                                    // ok - likely cells is out of range
                                    string ex = e.Message;
                                }
                            }
                        }
                    }
                }
            }

            // update labeled markers data
            // remove previous dynamic marker list
            int currentRow = m_FrameOfData.nMarkerSets + htMarkers.Count + htRigidBodies.Count + htRigidBodyMarkers.Count + htForcePlates.Count + htDevices.Count + 1;
            int labeledCount = 0;
            if (LabeledMarkersCheckBox.Checked)
            {
                int assetID, memberID;
                string name;
                for (int i = 0; i < m_FrameOfData.nMarkers; i++)
                {
                    NatNetML.Marker m = m_FrameOfData.LabeledMarkers[i];

                    // Marker ID Scheme:
                    // Active Markers:
                    //   ID = ActiveID, correlates to RB ActiveLabels list
                    // Passive Markers: 
                    //   If Asset with Legacy Labels
                    //      AssetID 	(Hi Word)
                    //      MemberID	(Lo Word)
                    //   Else
                    //      PointCloud ID

                    bool activeMarker = false;
                    int activeKey = m.ID.GetHashCode();
                    if (htRigidBodyMarkers.Contains(activeKey))
                        activeMarker = true;

                    // marker parameters
                    bool bOccluded = (m.parameters & (1 << 0)) != 0;
                    bool bPCSolved = (m.parameters & (1 << 1)) != 0;
                    bool bModelSolved = (m.parameters & (1 << 2)) != 0;
                    bool bHasModel = (m.parameters & (1 << 3)) != 0;
                    bool bUnlabeled = (m.parameters & (1 << 4)) != 0;
                    bool bActive = (m.parameters & (1 << 5)) != 0;


                    if (bActive || activeMarker)
                    {
                        name = "Active Marker: " + m.ID;
                    }
                    else
                    {
                        if (bUnlabeled)
                        {
                            name = "Unlabeled Marker (PointCloud ID: " + m.ID + ")";
                        }
                        else
                        {
                            NatNetClientML.DecodeID(m.ID, out assetID, out memberID);
                            int key = assetID.GetHashCode();
                            if (htRigidBodies.Contains(key) || htSkels.Contains(key) || htSkelRBs.Contains(key))
                                name = "Passive Marker (AssetID: " + assetID + "  MemberID: " + memberID + ")";
                            else
                                name = "Passive Marker (PointCloud ID: " + m.ID + ")";
                        }
                    }

                    // expand grid if necessary
                    while (currentRow >= dataGridView1.RowCount)
                        dataGridView1.Rows.Add();

                    dataGridView1.Rows[currentRow].Cells[0].Value = name;
                    dataGridView1.Rows[currentRow].Cells[1].Value = m.x * m_ServerToMillimeters;
                    dataGridView1.Rows[currentRow].Cells[2].Value = m.y * m_ServerToMillimeters;
                    dataGridView1.Rows[currentRow].Cells[3].Value = m.z * m_ServerToMillimeters;
                    dataGridView1.Rows[currentRow].Cells[4].Value = m.residual * m_ServerToMillimeters;

                    // Active Markers : Also add to corresponding RigidBody Marker Row
                    if(activeMarker)
                    {
                        int rowIndex = (int)htRigidBodyMarkers[activeKey];
                        if (rowIndex >= 0)
                        {
                            dataGridView1.Rows[rowIndex].Cells[1].Value = m.x * m_ServerToMillimeters;
                            dataGridView1.Rows[rowIndex].Cells[2].Value = m.y * m_ServerToMillimeters;
                            dataGridView1.Rows[rowIndex].Cells[3].Value = m.z * m_ServerToMillimeters;
                        }
                    }

                    labeledCount++;
                    currentRow++;
                }

            }

            // clear any remaining rows ( e.g. from markers not present in this frame)
            if (LabeledMarkersCheckBox.Checked)
            {
                while (currentRow < dataGridView1.RowCount)
                {
                    dataGridView1.Rows[currentRow].Cells[0].Value = "";
                    dataGridView1.Rows[currentRow].Cells[1].Value = "";
                    dataGridView1.Rows[currentRow].Cells[2].Value = "";
                    dataGridView1.Rows[currentRow].Cells[3].Value = "";
                    dataGridView1.Rows[currentRow].Cells[4].Value = "";
                    currentRow++;
                }
            }

            // if rows not empty, add frame telemetry to grid, so its graphable
            if(dataGridView1.Rows.Count > 0)
            {
                // Update the interframe duration
                dataGridView1.Rows[0].Cells[7].Value = interframeDuration;
                // Update Frame drop detection
                dataGridView1.Rows[0].Cells[8].Value = droppedFrameIndicator;

                bool bMotiveHardwareLatenciesAvailable = m_FrameOfData.CameraMidExposureTimestamp != 0;
                double systemLatencyMs = -1.0f;
                double totalLatencyMs = -1.0f;
                if (bMotiveHardwareLatenciesAvailable)
                {
                    // Motive System latency ( Camera Photons -> Motive Transmit)
                    systemLatencyMs = (m_FrameOfData.TransmitTimestamp - m_FrameOfData.CameraMidExposureTimestamp) / (double)desc.HighResClockFrequency * 1000.0;

                    // Total latency ( Camera Photons -> Client Receive )
                    totalLatencyMs = m_TotalLatency;
                }
                dataGridView1.Rows[0].Cells[9].Value = systemLatencyMs.ToString("F3");
                dataGridView1.Rows[0].Cells[12].Value = totalLatencyMs.ToString("F3");

                // Motive Software latency ( Frame Group ->  Motive Trasmit )
                bool bMotiveLatenciesAvailable = m_FrameOfData.CameraDataReceivedTimestamp != 0;
                double softwareLatencyMs = bMotiveLatenciesAvailable ?
                    (m_FrameOfData.TransmitTimestamp - m_FrameOfData.CameraDataReceivedTimestamp) / (double)desc.HighResClockFrequency * 1000.0
                    : -1.0;
                dataGridView1.Rows[0].Cells[10].Value = softwareLatencyMs.ToString( "F3" );

                // Transmit latency ( Motive Transmit -> Client Receive )
                double transitLatencyMs = m_FrameOfDataTransitLatency;
                dataGridView1.Rows[0].Cells[11].Value = transitLatencyMs.ToString( "F3" );

                // Ping ( Client -> Server -> Client )
                // Test : overriding timecode as packet size
                this.Ping.HeaderText = "PacketSize(bytes)";
                uint packetSize = m_FrameOfData.Timecode;
                dataGridView1.Rows[0].Cells[13].Value = packetSize.ToString();
                //dataGridView1.Rows[0].Cells[13].Value = mLastPingTimeMs.ToString("F3");

            }

            if ( dataGridView1.Rows.Count != mLastRowCount )
            {
                mLastRowCount = dataGridView1.Rows.Count;
                int newHeight = (dataGridView1.CurrentRow.Height+1) * mLastRowCount + 5;
                newHeight = Math.Max(newHeight, minGridHeight);
                dataGridView1.Height = newHeight;
            }
        }

        int GetUniqueRBKey(int skeletonID, int rigidBodyID)
        {
            return (skeletonID+1) * 1000 + rigidBodyID;
        }

        /// <summary>
        /// [NatNet] Request a description of the Active Model List from the server (e.g. Motive) and build up a new spreadsheet  
        /// </summary>
        private void GetDataDescriptions()
        {
            mForcePlates.Clear();
            htForcePlates.Clear();
            mDevices.Clear();
            htDevices.Clear();
            mRigidBodies.Clear();
            dataGridView1.Rows.Clear();
            htMarkers.Clear();
            htRigidBodies.Clear();
            htRigidBodyMarkers.Clear();
            htSkels.Clear();
            htSkelRBs.Clear();

            OutputMessage("Retrieving Data Descriptions....");
            List<NatNetML.DataDescriptor> descs = new List<NatNetML.DataDescriptor>();
            bool bSuccess = m_NatNet.GetDataDescriptions(out descs);
            if (bSuccess)
            {
                OutputMessage(String.Format("Retrieved {0} Data Descriptions....", descs.Count));
                int iObject = 0;
                foreach (NatNetML.DataDescriptor d in descs)
                {
                    iObject++;

                    // MarkerSets
                    if (d.type == (int)NatNetML.DataDescriptorType.eMarkerSetData)
                    {
                        NatNetML.MarkerSet ms = (NatNetML.MarkerSet)d;
                        OutputMessage("Data Def " + iObject.ToString() + " [MarkerSet]");

                        OutputMessage(" Name : " + ms.Name);
                        OutputMessage(String.Format(" Markers ({0}) ", ms.nMarkers));
                        dataGridView1.Rows.Add("MarkerSet: " + ms.Name);
                        for (int i = 0; i < ms.nMarkers; i++)
                        {
                            OutputMessage(("  " + ms.MarkerNames[i]));
                            int rowIndex = dataGridView1.Rows.Add("  " + ms.MarkerNames[i]);
                            // MarkerNameIndexToRow map
                            String strUniqueName = ms.Name + i.ToString();
                            int key = strUniqueName.GetHashCode();
                            htMarkers.Add(key, rowIndex);
                        }
                    }

                    // RigidBodies
                    else if (d.type == (int)NatNetML.DataDescriptorType.eRigidbodyData)
                    {
                        NatNetML.RigidBody rb = (NatNetML.RigidBody)d;

                        OutputMessage("Data Def " + iObject.ToString() + " [RigidBody]");
                        OutputMessage(" Name : " + rb.Name);
                        OutputMessage(" ID : " + rb.ID);
                        OutputMessage(" ParentID : " + rb.parentID);
                        OutputMessage(" OffsetX : " + rb.offsetx);
                        OutputMessage(" OffsetY : " + rb.offsety);
                        OutputMessage(" OffsetZ : " + rb.offsetz);

                        mRigidBodies.Add(rb);

                        int rowIndex = dataGridView1.Rows.Add("RigidBody: " + rb.Name + " (ID:"+rb.ID+")");
                        // RigidBodyIDToRow map
                        int key = rb.ID.GetHashCode();
                        try
                        {
                            htRigidBodies.Add(key, rowIndex);
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Duplicate RigidBody ID Detected : " + ex.Message);
                        }

                        // RigidBody Markers
                        for (int i = 0; i < rb.nMarkers; i++)
                        {
                            // Uses Active Markers?
                            if (rb.MarkerRequiredLabels[i] > 0)
                            {
                                key = rb.MarkerRequiredLabels[i].GetHashCode();
                                int markerRowIndex = dataGridView1.Rows.Add("Marker " + rb.MarkerRequiredLabels[i]);
                                htRigidBodyMarkers.Add(key, markerRowIndex);
                            }
                        }
                    }

                    // Skeletons
                    else if (d.type == (int)NatNetML.DataDescriptorType.eSkeletonData)
                    {
                        NatNetML.Skeleton sk = (NatNetML.Skeleton)d;
                        int key = sk.ID.GetHashCode();
                        try
                        {
                            htSkels.Add(key, sk);
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("Duplicate Skeleton ID Detected : " + ex.Message);
                        }


                        OutputMessage("Data Def " + iObject.ToString() + " [Skeleton]");
                        OutputMessage(" Name : " + sk.Name);
                        OutputMessage(" ID : " + sk.ID);
                        dataGridView1.Rows.Add("Skeleton: " + sk.Name);
                        for (int i = 0; i < sk.nRigidBodies; i++)
                        {
                            RigidBody rb = sk.RigidBodies[i];
                            OutputMessage(" RB Name  : " + rb.Name);
                            OutputMessage(" RB ID    : " + rb.ID);
                            OutputMessage(" ParentID : " + rb.parentID);
                            OutputMessage(" OffsetX  : " + rb.offsetx);
                            OutputMessage(" OffsetY  : " + rb.offsety);
                            OutputMessage(" OffsetZ  : " + rb.offsetz);

                            //mRigidBodies.Add(rb);
                            key = GetUniqueRBKey(sk.ID, rb.ID);
                            htSkelRBs.Add(key, rb);
#if true
                            int rowIndex = dataGridView1.Rows.Add("Bone: " + rb.Name);
                            // RigidBodyIDToRow map
                            int uniqueID = GetUniqueRBKey(sk.ID, rb.ID);
                            key = uniqueID.GetHashCode();
                            if (htRigidBodies.ContainsKey(key))
                                MessageBox.Show("Duplicate RigidBody ID");
                            else
                                htRigidBodies.Add(key, rowIndex);
#endif

                        }
                    }

                    // ForcePlates
                    else if (d.type == (int)NatNetML.DataDescriptorType.eForcePlateData)
                    {
                        NatNetML.ForcePlate fp = (NatNetML.ForcePlate)d;


                        OutputMessage("Data Def " + iObject.ToString() + " [ForcePlate]");
                        OutputMessage(" Name : " + fp.Serial);
                        OutputMessage(" ID : " + fp.ID);
                        OutputMessage(" Width : " + fp.Width);
                        OutputMessage(" Length : " + fp.Length);

                        mForcePlates.Add(fp);

                        //int rowIndex = dataGridView1.Rows.Add("ForcePlate: " + fp.Serial);
                        int rowIndex = dataGridView1.Rows.Add("ForcePlate " + fp.ID.ToString());
                        // ForcePlateIDToRow map
                        int key = fp.ID.GetHashCode();
                        htForcePlates.Add(key, rowIndex);
                    }

                    // Devices
                    else if (d.type == (int)NatNetML.DataDescriptorType.eDeviceData)
                    {
                        NatNetML.Device device = (NatNetML.Device)d;

                        OutputMessage("Data Def " + iObject.ToString() + " [Device]");
                        OutputMessage(" Name : " + device.Name);
                        OutputMessage(" Serial : " + device.Serial);
                        OutputMessage(" ID : " + device.ID);
                        OutputMessage(" Channels : " + device.ChannelCount);
                        for (int i = 0; i < device.ChannelCount; i++)
                        {
                            OutputMessage("  " + device.ChannelNames[i]);
                        }
                        mDevices.Add(device);


                        int rowIndex = dataGridView1.Rows.Add("Device: " + device.Name);
                        int key = device.ID.GetHashCode();
                        if (htDevices.ContainsKey(key))
                            MessageBox.Show("Duplicate Device ID");
                        else
                            htDevices.Add(key, rowIndex);

                    }
                    // Cameras
                    else if (d.type == (int)NatNetML.DataDescriptorType.eCameraData)
                    {
                        NatNetML.Camera camera = (NatNetML.Camera)d;
                        OutputMessage("Data Def " + iObject.ToString() + " [Camera]");
                        OutputMessage(" Name : " + camera.Name);

                        String strPos = String.Format(" Position {0},{1},{2}", camera.x, camera.y, camera.z);
                        OutputMessage(strPos);

                        String strOri = String.Format(" Orientation {0},{1},{2},{3}", camera.qx, camera.qy, camera.qz, camera.qw);
                        OutputMessage(strOri);
                    }
                    else
                    {
                        OutputMessage("Unknown DataType");
                    }
                }
            }
            else
            {
                OutputMessage("Unable to retrieve DataDescriptions");
            }
        }

        private void buttonGetDataDescriptions_Click(object sender, EventArgs e)
        {
            GetDataDescriptions();
        }

        void ProcessFrameOfData(ref NatNetML.FrameOfMocapData data)
        {

            TelemetryData telemetry = new TelemetryData();
            bool bMotiveHardwareLatenciesAvailable = data.CameraMidExposureTimestamp != 0;
            if (bMotiveHardwareLatenciesAvailable)
            {
                telemetry.TotalLatency = m_NatNet.SecondsSinceHostTimestamp(data.CameraMidExposureTimestamp) * 1000.0;
                telemetry.MotiveTotalLatency = (data.TransmitTimestamp - data.CameraMidExposureTimestamp) / (double)desc.HighResClockFrequency * 1000.0;
            }
            bool bMotiveLatenciesAvailable = data.CameraDataReceivedTimestamp != 0;
            if(bMotiveLatenciesAvailable)
            {
            }
            telemetry.TransmitLatency = m_NatNet.SecondsSinceHostTimestamp(data.TransmitTimestamp) * 1000.0;


            // detect and reported any 'reported' frame drop (as reported by server)
            if (m_fLastFrameTimestamp != 0.0f)
            {
                double framePeriod = 1.0f / m_ServerFramerate;
                double thisPeriod = data.fTimestamp - m_fLastFrameTimestamp;
                double delta = thisPeriod - framePeriod;
                double fudgeFactor = 0.002f; // 2 ms
                if (delta > fudgeFactor)
                {
                    //OutputMessage("Frame Drop: ( ThisTS: " + data.fTimestamp.ToString("F3") + "  LastTS: " + m_fLastFrameTimestamp.ToString("F3") + " )");
                    double missingPeriod = delta / framePeriod;
                    int nMissing = (int)(missingPeriod + 0.5);
                    mDroppedFrames += nMissing;
                    telemetry.DroppedFrames = nMissing;
                    droppedFrameIndicator = 10; // for graphing only
                }
                else
                {
                    droppedFrameIndicator = 0;
                }
            }

            // check and report frame drop (frame id based)
            if (mLastFrame != 0)
            {
                if ((data.iFrame - mLastFrame) != 1)
                {
                    //OutputMessage("Frame Drop: ( ThisFrame: " + data.iFrame.ToString() + "  LastFrame: " + mLastFrame.ToString() + " )");
                    //mDroppedFrames++;
                }
            }

            if (data.bTrackingModelsChanged)
                mNeedTrackingListUpdate = true;

            // NatNet manages the incoming frame of mocap data, so if we want to keep it, we must make a copy of it
            FrameOfMocapData deepCopy = new FrameOfMocapData(data);
            
            // Add frame to a background queue for access by other threads
            // Note: this lock should always succeed immediately, unless connecting/disconnecting, when the queue gets reset
            lock(BackQueueLock)
            {
                m_BackQueue.Enqueue(deepCopy);

                // limit background queue size to 10 frames
                while(m_BackQueue.Count > 10)
                {
                    m_BackQueue.Dequeue();
                }
            }

            // Update the shared UI queue, only if the UI thread is not updating (we don't want to wait here as we're in the data update thread)
            bool lockAcquired = false;
            try
            {
                Monitor.TryEnter(FrontQueueLock, ref lockAcquired);
                if (lockAcquired)
                {
                    // [optional] clear the frame queue before adding a new frame (UI only wants most recent frame)
                    m_FrontQueue.Clear();
                    m_FrontQueue.Enqueue(deepCopy);
                
                    m_FrameTransitLatencies.Clear();
                    m_FrameTransitLatencies.Enqueue(telemetry.TransmitLatency);

                    m_TotalLatencies.Clear();
                    m_TotalLatencies.Enqueue(telemetry.TotalLatency);
                }
                else 
                {
                    mUIBusyCount++;
                }
            }
            finally
            {
                if(lockAcquired)
                    Monitor.Exit(FrontQueueLock);
            }

            // recording : write packet to data file
            if (mRecording)
            {
                WriteFrame(deepCopy, telemetry);
            }

            mLastFrame = data.iFrame;
            m_fLastFrameTimestamp = data.fTimestamp;

        }

        /// <summary>
        /// [NatNet] m_NatNet_OnFrameReady will be called when a frame of Mocap
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
        void m_NatNet_OnFrameReady(NatNetML.FrameOfMocapData data, NatNetML.NatNetClientML client)
        {
            // measure time between frame arrival (inter frame)
            m_FramePeriodTimer.Stop();
            interframeDuration = m_FramePeriodTimer.Duration();

            // measure processing time (intra frame)
            m_ProcessingTimer.Start();

            // process data
            // NOTE!  do as little as possible here as we're on the data servicing thread
            ProcessFrameOfData(ref data);

            // report if we are taking longer than a mocap frame time
            // which eventually will back up the network receive buffer and result in frame drop
            m_ProcessingTimer.Stop();
            double appProcessingTimeMSecs = m_ProcessingTimer.Duration();
            double mocapFramePeriodMSecs = (1.0f / m_ServerFramerate) * 1000.0f;
            if (appProcessingTimeMSecs > mocapFramePeriodMSecs)
            {
                OutputMessage("Warning : Frame handler taking longer than frame period: " + appProcessingTimeMSecs.ToString("F2"));
            }

            m_FramePeriodTimer.Start();
        }

        // [NatNet] [optional] alternate function signatured frame ready callback handler for .NET applications/hosts
        // that don't support the m_NatNet_OnFrameReady defined above (e.g. MATLAB)
        void m_NatNet_OnFrameReady2(object sender, NatNetEventArgs e)
        {
            m_NatNet_OnFrameReady(e.data, e.client);
        }

        private void PollData()
        {
            FrameOfMocapData data = m_NatNet.GetLastFrameOfData();
            ProcessFrameOfData(ref data);
        }

        private void SetDataPolling(bool poll)
        {
            if (poll)
            {
                // disable event based data handling
                m_NatNet.OnFrameReady -= m_NatNet_OnFrameReady;

                // enable polling 
                mPolling = true;
                pollThread.Start();
            }
            else
            {
                // disable polling
                mPolling = false;

                // enable event based data handling
                m_NatNet.OnFrameReady += new NatNetML.FrameReadyEventHandler(m_NatNet_OnFrameReady);
            }
        }

        private void GetLastFrameOfData()
        {
            FrameOfMocapData data = m_NatNet.GetLastFrameOfData();
            ProcessFrameOfData(ref data);
        }


        private void GetLastFrameOfDataButton_Click(object sender, EventArgs e)
        {
            // [NatNet] GetLastFrameOfData can be used to poll for the most recent avail frame of mocap data.
            // This mechanism is slower than the event handler mechanism, and in general is not recommended,
            // since it must wait for a frame to become available and apply a lock to that frame while it copies
            // the data to the returned value.

            // get a copy of the most recent frame of data
            // returns null if not available or cannot obtain a lock on it within a specified timeout
            FrameOfMocapData data = m_NatNet.GetLastFrameOfData();
            if (data != null)
            {
                // do something with the data
                String frameInfo = String.Format("FrameID : {0}", data.iFrame);
                OutputMessage(frameInfo);
            }
        }


        private void WriteFrame(FrameOfMocapData data, TelemetryData telemetry)
        {
            String str = "";
            bool recordMarkerData = false;
            bool recordForcerData = false;
            bool recordRBData = false;

            str += data.fTimestamp.ToString("F3") + "\t";
            str += telemetry.TransmitLatency.ToString("F3") + "\t";
            str += telemetry.TotalLatency.ToString("F3") + "\t";
            str += telemetry.DroppedFrames.ToString() + "\t";

            // 'all' markerset data
            if(recordMarkerData)
            {
                for (int i = 0; i < m_FrameOfData.nMarkerSets; i++)
                {
                    NatNetML.MarkerSetData ms = m_FrameOfData.MarkerSets[i];
                    if(ms.MarkerSetName == "all")
                    {
                       for (int j = 0; j < ms.nMarkers; j++)
                        {
                           str += ms.Markers[j].x.ToString("F3") + "\t";
                           str += ms.Markers[j].y.ToString("F3") + "\t";
                           str += ms.Markers[j].z.ToString("F3") + "\t";
                        }
                    }
                }
            }

            // force plates
            if(recordForcerData)
            {
                // just write first subframe from each channel (fx[0], fy[0], fz[0], mx[0], my[0], mz[0])
                for (int i = 0; i < m_FrameOfData.nForcePlates; i++)
                {
                    NatNetML.ForcePlateData fp = m_FrameOfData.ForcePlates[i];
                    for(int iChannel=0; iChannel < fp.nChannels; iChannel++)
                    {
                        if(fp.ChannelData[iChannel].nFrames == 0)
                        {
                            str += 0.0f;    // empty frame
                        }
                        else
                        {
                            str += fp.ChannelData[iChannel].Values[0] + "\t";
                        }
                    }
                }
            }

            mWriter.WriteLine(str);
        }

        private void RecordDataButton_CheckedChanged(object sender, EventArgs e)
        {
            if (RecordDataButton.Checked)
            {
                try
                {
                    mWriter = File.CreateText("WinFormsData.txt");
                    mRecording = true;
                }
                catch (System.Exception ex)
                {
                    OutputMessage("Record Error : " + ex.Message);
                }
            }
            else
            {
                mWriter.Close();
                mRecording = false;
            }

        }

        private void UpdateUI()
        {
            if(mWantAutoconnect && mServerDetected && !mServerEstablished)
            {
                OutputMessage("Auto-Connecting to Motive...");

                int index = comboBoxLocal.FindString(mDetectedLocalIP);
                if (index >= 0)
                    comboBoxLocal.SelectedIndex = index;
                textBoxServer.Text = mDetectedServerIP;

                checkBoxConnect.Checked = true; // calls connect
                GetDataDescriptions();

            }

            // The frame queue is a shared resource with the FrameOfMocap delivery thread, so lock it while reading
            bool lockAcquired = false;
            try
            {
                if(mNeedTrackingListUpdate)
                {
                    GetDataDescriptions();
                    mNeedTrackingListUpdate = false;
                }

                Monitor.TryEnter(FrontQueueLock, ref lockAcquired);
                if (lockAcquired)
                {
                    if (m_FrontQueue.Count > 0)
                    {
                        // policy: only draw the most recent frame in queue, discard the rest
                        while(m_FrontQueue.Count > 0)
                            m_FrameOfData = m_FrontQueue.Dequeue();
                        while (m_FrameTransitLatencies.Count > 0)
                            m_FrameOfDataTransitLatency = m_FrameTransitLatencies.Dequeue();
                        while (m_TotalLatencies.Count > 0)
                            m_TotalLatency = m_TotalLatencies.Dequeue();


                        Monitor.Exit(FrontQueueLock);
                        lockAcquired = false;

                        // update the data grid
                        UpdateDataGrid();

                        // update the chart
                        UpdateChart(m_FrameOfData.iFrame);

                        // redraw the chart
                        chart1.ChartAreas[0].RecalculateAxesScale();
                        chart1.ChartAreas[0].AxisX.Minimum = 0;
                        chart1.ChartAreas[0].AxisX.Maximum = GraphFrames;
                        chart1.Invalidate();

                        // Mocap server timestamp (in seconds)
                        TimestampValue.Text = m_FrameOfData.fTimestamp.ToString("F3");
                        DroppedFrameCountLabel.Text = mDroppedFrames.ToString();

                        // SMPTE timecode (if timecode generator present)
                        int hour, minute, second, frame, subframe;
                        bool bSuccess = NatNetClientML.DecodeTimecode(m_FrameOfData.Timecode, m_FrameOfData.TimecodeSubframe, out hour, out minute, out second, out frame, out subframe);
                        if (bSuccess)
                            TimecodeValue.Text = string.Format("{0:D2}:{1:D2}:{2:D2}:{3:D2}.{4:D2}", hour, minute, second, frame, subframe);

                        if (m_FrameOfData.bRecording)
                            chart1.BackColor = Color.Red;
                        else
                            chart1.BackColor = Color.White;
                    }
                }
            }
            finally 
            {
                if (lockAcquired)
                {
                    Monitor.Exit(FrontQueueLock);
                }
            }

        }

        public int LowWord(int number)
        {
            return number & 0xFFFF;
        }

        public int HighWord(int number)
        {
            return ((number >> 16) & 0xFFFF);
        }

        double RadiansToDegrees(double dRads)
        {
            return dRads * (180.0f / Math.PI);
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            mApplicationRunning = false;

            m_Discovery.EndDiscovery();

            if(UIUpdateThread.IsAlive)
                UIUpdateThread.Abort();

            m_NatNet.Disconnect();
        }

        private void RadioMulticast_CheckedChanged(object sender, EventArgs e)
        {
            bool bNeedReconnect = checkBoxConnect.Checked;
            if (bNeedReconnect)
            {
                Disconnect();
                Connect();
            }
        }

        private void RadioUnicast_CheckedChanged(object sender, EventArgs e)
        {
            bool bNeedReconnect = checkBoxConnect.Checked;
            if (bNeedReconnect)
            {
                Disconnect();
                Connect();
            }
        }

        private void RecordButton_Click(object sender, EventArgs e)
        {
            string command = "StartRecording";

            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc = m_NatNet.SendMessageAndWait(command, 3, 100, out response, out nBytes);
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }
        }

        private void StopRecordButton_Click(object sender, EventArgs e)
        {
            string command = "StopRecording";

            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
             
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }
        }

        private void LiveModeButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc = m_NatNet.SendMessageAndWait("LiveMode", out response, out nBytes);
        }

        private void EditModeButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc = m_NatNet.SendMessageAndWait("EditMode", out response, out nBytes);
        }

        private void TimelinePlayButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc = m_NatNet.SendMessageAndWait("TimelinePlay", out response, out nBytes);
        }

        private void TimelineStopButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc = m_NatNet.SendMessageAndWait("TimelineStop", out response, out nBytes);
        }

        private void SetRecordingTakeButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            String strCommand = "SetRecordTakeName," + RecordingTakeNameText.Text;
            int rc = m_NatNet.SendMessageAndWait(strCommand, out response, out nBytes);
        }

        private void SetPlaybackTakeButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            String strCommand = "SetPlaybackTakeName," + PlaybackTakeNameText.Text;
            int rc = m_NatNet.SendMessageAndWait(strCommand, out response, out nBytes);
        }

        private void TestButton_Click(object sender, EventArgs e)
        {
#if true
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc;
            rc = m_NatNet.SendMessageAndWait("FrameRate", out response, out nBytes);
            if (rc == 0)
            {
                try
                {
                    m_ServerFramerate = BitConverter.ToSingle(response, 0);
                    OutputMessage(String.Format("   Camera Framerate: {0}", m_ServerFramerate));
                }
                catch (System.Exception ex)
                {
                    OutputMessage(ex.Message);
                }
            }

#else
            int nBytes = 0;
            byte[] response = new byte[10000];
            int testVal;
            String command;
            int returnCode;

            command = "SetPlaybackTakeName," + PlaybackTakeNameText.Text;
            OutputMessage("Sending " + command);
            returnCode = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            // process return codes
            if (returnCode != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }

            testVal = 25;
            command = "SetPlaybackStartFrame," + testVal.ToString();
            OutputMessage("Sending " + command);
            returnCode = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            // process return codes
            if (returnCode != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if(opResult==0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command +  " handled but failed.");
            }
               
            testVal = 50;
            command = "SetPlaybackStopFrame," + testVal.ToString();
            OutputMessage("Sending " + command);
            returnCode = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (returnCode != 0)
            {
                OutputMessage("SetPlaybackStartFrame not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }

            testVal = 0;
            command = "SetPlaybackLooping," + testVal.ToString();
            OutputMessage("Sending " + command);
            returnCode = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (returnCode != 0)
            {
                OutputMessage("SetPlaybackStartFrame not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }

            testVal = 35;
            OutputMessage("Sending " + command);
            command = "SetPlaybackCurrentFrame," + testVal.ToString();
            returnCode = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (returnCode != 0)
            {
                OutputMessage("SetPlaybackStartFrame not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }

#endif

        }

        private void contextMenuStrip1_Opening(object sender, CancelEventArgs e)
        {

        }

        private void menuClear_Click(object sender, EventArgs e)
        {
            listView1.Items.Clear();
        }

        private void menuPause_Click(object sender, EventArgs e)
        {
            mPaused = menuPause.Checked;
        }

        private void GetTakeRangeButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc;
            rc = m_NatNet.SendMessageAndWait("CurrentTakeLength", out response, out nBytes);
            if (rc == 0)
            {
                try
                {
                    int takeLength = BitConverter.ToInt32(response, 0);
                    OutputMessage(String.Format("Current Take Length: {0}", takeLength));
                }
                catch (System.Exception ex)
                {
                    OutputMessage(ex.Message);
                }
            }
        }

        private void GetModeButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            int rc;
            rc = m_NatNet.SendMessageAndWait("CurrentMode", out response, out nBytes);
            if (rc == 0)
            {
                try
                {
                    String strMode = "";
                    int mode = BitConverter.ToInt32(response, 0);
                    if (mode == 0)
                        strMode = String.Format("Mode : Live");
                    else if (mode == 1)
                        strMode = String.Format("Mode : Recording");
                    else if (mode == 2)
                        strMode = String.Format("Mode : Edit");
                    OutputMessage(strMode);
                }
                catch (System.Exception ex)
                {
                    OutputMessage(ex.Message);
                }
            }
        }

        private void PollCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            SetDataPolling(PollCheckBox.Checked);
        }

        private void SubscribeButton_CheckChanged(object sender, EventArgs e)
        {
            if(checkBoxConnect.Checked)
            {
                Disconnect();
                Connect();
            }
        }

        private void SetPropertyButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            string command = "SetProperty," + NodeNameText.Text + "," + PropertyNameText.Text + "," + PropertyValueText.Text;
            int rc = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }
        }

        private void GetPropertyButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            string command = "GetProperty," + NodeNameText.Text + "," + PropertyNameText.Text;
            int rc = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                System.Text.Encoding encoding = System.Text.Encoding.UTF8;
                string result = new string(encoding.GetChars(response));
                result = result.Trim('\0'); // .NET string are not null terminated
                if ((result.Length == 0) || (result=="error"))
                    OutputMessage(command + " handled but failed.");
                else
                {
                    PropertyValueText.Text = result;
                    OutputMessage(command + " Value = " + result);
                }
            }
        }

        private void DisableAssetButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            string command = "SetProperty," + NodeNameText.Text + "," + "Enable,False";
            int rc = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }
        }

        private void EnableAssetButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            string command = "SetProperty," + NodeNameText.Text + "," + "Enable,True";
            int rc = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                int opResult = System.BitConverter.ToInt32(response, 0);
                if (opResult == 0)
                    OutputMessage(command + " handled and succeeded.");
                else
                    OutputMessage(command + " handled but failed.");
            }
        }

        void UpdatePing()
        {
            if (mServerIP.Length == 0)
                return;

            Ping pingSender = new Ping();
            PingOptions options = new PingOptions();

            // Use the default Ttl value which is 128,
            // but change the fragmentation behavior.
            options.DontFragment = true;

            // Create a buffer of 32 bytes of data to be transmitted.
            string data = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
            byte[] buffer = Encoding.ASCII.GetBytes(data);
            int timeout = 120;

            try
            {
                PingReply reply = pingSender.Send(mServerIP, timeout, buffer, options);
                if (reply.Status == IPStatus.Success)
                {
                    mLastPingTimeMs = reply.RoundtripTime;
                }
                else
                {
                    reply = pingSender.Send("192.168.1.1", timeout, buffer, options);
                    if (reply.Status == IPStatus.Success)
                    {
                        mLastPingTimeMs = reply.RoundtripTime;
                    }
                    else
                    {
                        mLastPingTimeMs = -1.0;
                    }
                }
            }
            catch (Exception ex)
            {

                OutputMessage("Ping Failed : " + ex.Message);
            }

        }

        private void CommandButton_Click(object sender, EventArgs e)
        {
            int nBytes = 0;
            byte[] response = new byte[10000];
            string command = CommandText.Text;
            //string command = "Bitstream," + SubscribeTextCommand.Text;
            int rc = m_NatNet.SendMessageAndWait(command, out response, out nBytes);
            if (rc != 0)
            {
                OutputMessage(command + " not handled by server");
            }
            else
            {
                // Command Response Buffer contents will vary by command:
                //
                //  - Most commands : response buffer is a 4 byte success code (success=0, failure=1)
                //  - Value query commands (e.g. GetProperty) : response buffer is either the value (if retrieved), else "error"

                if( command.Contains("GetProperty") || (command.Contains("GetTakeProperty")))
                {
                    System.Text.Encoding encoding = System.Text.Encoding.UTF8;
                    string result = new string(encoding.GetChars(response));
                    result = result.Trim('\0'); // .NET strings are not null terminated
                    if ((result.Length == 0) || (result == "error"))
                        OutputMessage(command + " handled but failed.");
                    else
                    {
                        PropertyValueText.Text = result;
                        OutputMessage(command + " Value = " + result);
                    }
                }
                else
                {
                    int opResult = System.BitConverter.ToInt32(response, 0);
                    if (opResult == 0)
                        OutputMessage(command + " handled and succeeded.");
                    else
                        OutputMessage(command + " handled but failed.");
                }

            }
        }

    }

    // Wrapper class for the windows high performance timer QueryPerfCounter
    // ( adapted from MSDN https://msdn.microsoft.com/en-us/library/ff650674.aspx )
    public class QueryPerfCounter
    {
        [DllImport("KERNEL32")]
        private static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

        [DllImport("Kernel32.dll")]
        private static extern bool QueryPerformanceFrequency(out long lpFrequency);

        private long start;
        private long stop;
        private long frequency;
        Decimal multiplier = new Decimal(1.0e9);

        public QueryPerfCounter()
        {
            if (QueryPerformanceFrequency(out frequency) == false)
            {
                // Frequency not supported
                throw new Win32Exception();
            }
        }

        public void Start()
        {
            QueryPerformanceCounter(out start);
        }

        public void Stop()
        {
            QueryPerformanceCounter(out stop);
        }

        // return elapsed time between start and stop, in milliseconds.
        public double Duration()
        {
            double val = ((double)(stop - start) * (double)multiplier) / (double)frequency;
            val = val / 1000000.0f;   // convert to ms
            return val;
        }
    }

    public class TelemetryData
    {
        public double MotiveTotalLatency = -1.0;
        public double TransmitLatency = -1.0;
        public double TotalLatency = -1.0;
        public int DroppedFrames = 0;
    }

}