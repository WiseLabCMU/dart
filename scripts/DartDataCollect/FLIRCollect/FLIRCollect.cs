//=============================================================================
// Copyright (c) 2001-2023 FLIR Systems, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

/**
 *	@example FLIRCollect.cs
 *
 *  @brief FLIRCollect.cs shows how to get chunk data on an image, either
 *  from the nodemap or from the image itself. It relies on information provided
 *  in the Enumeration_CSharp, Acquisition_CSharp, and NodeMapInfo_CSharp
 *  examples.
 *
 *	It can also be helpful to familiarize yourself with the
 *  ImageFormatControl_CSharp and Exposure_CSharp examples. As they are somewhat
 *  shorter and simpler, either provides a strong introduction to camera
 *  customization.
 *
 *	Chunk data provides information on various traits of an image. This includes
 *	identifiers such as frame ID, properties such as black level, and more. This
 *	information can be acquired from either the nodemap or the image itself.
 *
 *	It may be preferrable to grab chunk data from each individual image, as it
 *	can be hard to verify whether data is coming from the correct image when
 *	using the nodemap. This is because chunk data retrieved from the nodemap is
 *	only valid for the current image; when GetNextImage() is called, chunk data
 *	will be updated to that of the new current image.
 *	
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

using System;
using System.IO;
using SpinnakerNET.GenApi;
using SpinnakerNET;
using Newtonsoft.Json;

namespace FLIRCollect
{
    class FLIRMetaData
    {
        public double timestamp;
        public string filename;
        public ManagedChunkData chunkData;
    }

    class Program
    {
        // This function configures the camera to add chunk data to each image.
        // It does this by enabling each type of chunk data after enabling
        // chunk data mode. When chunk data mode is turned on, the data is made
        // available in both the nodemap and each image.
        static int ConfigureChunkData(INodeMap nodeMap)
        {
            int result = 0;

            Console.WriteLine("\n*** CONFIGURING CHUNK DATA ***\n");

            try
            {
                //
                // Activate chunk mode
                //
                // *** NOTES ***
                // Once enabled, chunk data will be available at the end of the
                // payload of every image captured until it is disabled. Chunk
                // data can also be retrieved from the nodemap.
                //
                IBool iChunkModeActive = nodeMap.GetNode<IBool>("ChunkModeActive");
                if (iChunkModeActive == null || !iChunkModeActive.IsWritable)
                {
                    Console.WriteLine("Cannot active chunk mode. Aborting...");
                    return -1;
                }

                iChunkModeActive.Value = true;

                Console.WriteLine("Chunk mode activated...");

                //
                // Enable all types of chunk data
                //
                // *** NOTES ***
                // Enabling chunk data requires working with nodes:
                // "ChunkSelector" is an enumeration selector node and
                // "ChunkEnable" is a boolean. It requires retrieving the
                // selector node (which is of enumeration node type), selecting
                // the entry of the chunk data to be enabled, retrieving the
                // corresponding boolean, and setting it to true.
                //
                // In this example, all chunk data is enabled, so these steps
                // are performed in a loop. Once this is complete, chunk mode
                // still needs to be activated.
                //
                // Retrieve selector node
                IEnum iChunkSelector = nodeMap.GetNode<IEnum>("ChunkSelector");
                if (iChunkSelector == null || !iChunkSelector.IsReadable || !iChunkSelector.IsWritable)
                {
                    Console.WriteLine("Chunk selector not available. Aborting...");
                    return -1;
                }

                // Retrieve entries
                EnumEntry[] entries = iChunkSelector.Entries;

                Console.WriteLine("Enabling entries...");

                for (int i = 0; i < entries.Length; i++)
                {
                    // Select entry to be enabled
                    IEnumEntry iChunkSelectorEntry = entries[i];

                    // Go to next node if problem occurs
                    if (!iChunkSelectorEntry.IsReadable)
                    {
                        continue;
                    }

                    iChunkSelector.Value = iChunkSelectorEntry.Value;

                    Console.Write("\t{0}: ", iChunkSelectorEntry.Symbolic);

                    // Retrieve corresponding boolean
                    IBool iChunkEnable = nodeMap.GetNode<IBool>("ChunkEnable");

                    // Enable the boolean, thus enabling the corresponding chunk
                    // data
                    if (iChunkEnable == null)
                    {
                        Console.WriteLine("not available");
                        result = -1;
                    }
                    else if (iChunkEnable.Value)
                    {
                        Console.WriteLine("enabled");
                    }
                    else if (iChunkEnable.IsWritable)
                    {
                        iChunkEnable.Value = true;
                        Console.WriteLine("enabled");
                    }
                    else
                    {
                        Console.WriteLine("not writable");
                        result = -1;
                    }
                }
                Console.WriteLine();
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function displays a select amount of chunk data from the image.
        // Unlike accessing chunk data via the nodemap, there is no way to loop
        // through all available data.
        static ManagedChunkData GetChunkData(IManagedImage managedImage)
        {
            Console.WriteLine("Printing chunk data from image...");
            try
            {
                //
                // Retrieve chunk data from image
                //
                // *** NOTES ***
                // When retrieving chunk data from an image, the data is stored
                // in a a ChunkData object and accessed with getter functions.
                //
                ManagedChunkData managedChunkData = managedImage.ChunkData;

                //
                // Retrieve exposure time; exposure time recorded in microseconds
                //
                // *** NOTES ***
                // In C#, Floating point numbers are returned from chunk data
                // objects as a double.
                //
                double exposureTime = managedChunkData.ExposureTime;
                Console.WriteLine("\tExposure time: {0}", exposureTime);

                //
                // Retrieve frame ID
                //
                // *** NOTES ***
                // In C#, Integers are returned as a long.
                //
                long frameID = managedChunkData.FrameID;
                Console.WriteLine("\tFrame ID: {0}", frameID);

                // Retrieve gain; gain recorded in decibels
                double gain = managedChunkData.Gain;
                Console.WriteLine("\tGain: {0}", gain);

                // Retrieve height; height recorded in pixels
                long height = managedChunkData.Height;
                Console.WriteLine("\tHeight: {0}", height);

                // Retrieve offset X; offset X recorded in pixels
                long offsetX = managedChunkData.OffsetX;
                Console.WriteLine("\tOffset X: {0}", offsetX);

                // Retrieve offset Y; offset Y recorded in pixels
                long offsetY = managedChunkData.OffsetY;
                Console.WriteLine("\tOffset Y: {0}", offsetY);

                // Retrieve sequencer set active
                long sequencerSetActive = managedChunkData.SequencerSetActive;
                Console.WriteLine("\tSequencer set active: {0}", sequencerSetActive);

                // Retrieve timestamp
                long timestamp = managedChunkData.Timestamp;
                Console.WriteLine("\tTimestamp: {0}", timestamp);

                // Retrieve width; width recorded in pixels
                long width = managedChunkData.Width;
                Console.WriteLine("\tWidth: {0}", width);

                return managedChunkData;
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
            }

            return null;
        }

        // This function prints the device information of the camera from the
        // transport layer; please see NodeMapInfo_CSharp example for more
        // in-depth comments on printing device information from the nodemap.
        static int PrintDeviceInfo(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                Console.WriteLine("\n*** DEVICE INFORMATION ***\n");

                ICategory category = nodeMap.GetNode<ICategory>("DeviceInformation");
                if (category != null && category.IsReadable)
                {
                    for (int i = 0; i < category.Children.Length; i++)
                    {
                        Console.WriteLine(
                            "{0}: {1}",
                            category.Children[i].Name,
                            (category.Children[i].IsReadable ? category.Children[i].ToString()
                             : "Node not available"));
                    }
                    Console.WriteLine();
                }
                else
                {
                    Console.WriteLine("Device control information not available.");
                }
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function acquires and saves 10 images from a device; please see
        // Acquisition_CSharp example for more in-depth comments on the
        // acquisition of images.
        static int AcquireImages(IManagedCamera cam, INodeMap nodeMap, INodeMap nodeMapTLDevice, string outputPath)
        {
            int result = 0;
            Console.WriteLine("\n*** IMAGE ACQUISITION ***\n");
            try
            {
                // Set acquisition mode to continuous
                IEnum iAcquisitionMode = nodeMap.GetNode<IEnum>("AcquisitionMode");
                if (iAcquisitionMode == null || !iAcquisitionMode.IsWritable || !iAcquisitionMode.IsReadable)
                {
                    Console.WriteLine("Unable to set acquisition mode to continuous (node retrieval). Aborting...\n");
                    return -1;
                }

                IEnumEntry iAcquisitionModeContinuous = iAcquisitionMode.GetEntryByName("Continuous");
                if (iAcquisitionModeContinuous == null || !iAcquisitionModeContinuous.IsReadable)
                {
                    Console.WriteLine(
                        "Unable to set acquisition mode to continuous (enum entry retrieval). Aborting...\n");
                    return -1;
                }

                iAcquisitionMode.Value = iAcquisitionModeContinuous.Symbolic;
                Console.WriteLine("Acquisition mode set to continuous...");

                // Set timestamp latch
                ICommand iTimestampLatch = nodeMap.GetNode<ICommand>("TimestampLatch");
                DateTime startTime = DateTime.UtcNow;
                iTimestampLatch.Execute();
                DateTime endTime = DateTime.UtcNow;
                DateTime pcLatchTime = startTime.AddTicks((endTime - startTime).Ticks / 2);

                IInteger iTimestampLatchValue = nodeMap.GetNode<IInteger>("TimestampLatchValue");
                double flirLatchTime = iTimestampLatchValue.Value * 1e-9;

                double timeOffset = new DateTimeOffset(pcLatchTime).ToUnixTimeMilliseconds() / 1000.0 - flirLatchTime;

                // Begin acquiring images
                cam.BeginAcquisition();
                Console.WriteLine("Acquiring images: Press ESC to stop...");

                // Retrieve device serial number for filename
                string deviceSerialNumber = "";

                IString iDeviceSerialNumber = nodeMapTLDevice.GetNode<IString>("DeviceSerialNumber");
                if (iDeviceSerialNumber != null && iDeviceSerialNumber.IsReadable)
                {
                    deviceSerialNumber = iDeviceSerialNumber.Value;
                    Console.WriteLine("Device serial number retrieved as {0}...", deviceSerialNumber);
                }
                Console.WriteLine();

                // Retrieve, convert, and save images

                //
                // Create ImageProcessor instance for post processing images
                //
                IManagedImageProcessor processor = new ManagedImageProcessor();

                //
                // Set default image processor color processing method
                //
                // *** NOTES ***
                // By default, if no specific color processing algorithm is set, the image
                // processor will default to NEAREST_NEIGHBOR method.
                //
                processor.SetColorProcessing(ColorProcessingAlgorithm.HQ_LINEAR);

                int imageCnt = 0;
                using (StreamWriter metaDataWriter = File.CreateText(Path.Combine(outputPath, "..", "flir.txt")))
                {
                    while (!(Console.KeyAvailable && Console.ReadKey().Key == ConsoleKey.Escape))
                    {
                        try
                        {
                            // Retrieve next received image and ensure image completion
                            using (IManagedImage rawImage = cam.GetNextImage(1000))
                            {
                                if (rawImage.IsIncomplete)
                                {
                                    Console.WriteLine("Image incomplete with image status {0}...", rawImage.ImageStatus);
                                }
                                else
                                {
                                    // Print image information
                                    Console.WriteLine(
                                        "Grabbed image {0}, width = {1}, height = {1}",
                                        imageCnt,
                                        rawImage.Width,
                                        rawImage.Height);

                                    // Convert image to mono 8
                                    using (
                                        IManagedImage convertedImage = processor.Convert(rawImage, PixelFormatEnums.RGB8))
                                    {
                                        // Create unique file name
                                        string filename_nopath = imageCnt + ".jpg";
                                        string filename = Path.Combine(outputPath, filename_nopath);

                                        // Save image
                                        convertedImage.Save(filename);
                                        Console.WriteLine("Image saved at {0}", filename);

                                        // Display chunk data
                                        ManagedChunkData managedChunkData = GetChunkData(rawImage);
                                        FLIRMetaData metaData = new FLIRMetaData
                                        {
                                            timestamp = timeOffset + managedChunkData.Timestamp * 1e-9,
                                            filename = filename_nopath,
                                            chunkData = managedChunkData
                                        };
                                        if (managedChunkData != null)
                                            metaDataWriter.WriteLine(JsonConvert.SerializeObject(metaData));
                                        else
                                            result = -1;
                                    }
                                }
                            }
                            Console.WriteLine();
                        }
                        catch (SpinnakerException ex)
                        {
                            Console.WriteLine("Error: {0}", ex.Message);
                            result = -1;
                        }
                        ++imageCnt;
                    }
                }

                // End acquisition
                cam.EndAcquisition();
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function disables each type of chunk data before disabling chunk data mode.
        static int DisableChunkData(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                // Retrieve selector node
                IEnum iChunkSelector = nodeMap.GetNode<IEnum>("ChunkSelector");
                if (iChunkSelector == null || !iChunkSelector.IsReadable || !iChunkSelector.IsWritable)
                {
                    Console.WriteLine("Chunk selector not available. Aborting...");
                    return -1;
                }

                // Retrieve entries
                EnumEntry[] entries = iChunkSelector.Entries;

                Console.WriteLine("Disabling entries...");

                for (int i = 0; i < entries.Length; i++)
                {
                    // Select entry to be disabled
                    IEnumEntry iChunkSelectorEntry = entries[i];

                    // Go to next node if problem occurs
                    if (!iChunkSelectorEntry.IsReadable)
                    {
                        continue;
                    }

                    iChunkSelector.Value = iChunkSelectorEntry.Value;

                    Console.Write("\t{0}: ", iChunkSelectorEntry.Symbolic);

                    // Retrieve corresponding boolean
                    IBool iChunkEnable = nodeMap.GetNode<IBool>("ChunkEnable");

                    // Disable the boolean, thus disabling the corresponding chunk
                    // data
                    if (iChunkEnable == null)
                    {
                        Console.WriteLine("not available");
                        result = -1;
                    }
                    else if (!iChunkEnable.Value)
                    {
                        Console.WriteLine("disabled");
                    }
                    else if (iChunkEnable.IsWritable)
                    {
                        iChunkEnable.Value = false;
                        Console.WriteLine("disabled");
                    }
                    else
                    {
                        Console.WriteLine("not writable");
                    }
                }
                Console.WriteLine();

                // Deactivate ChunkMode
                IBool iChunkModeActive = nodeMap.GetNode<IBool>("ChunkModeActive");
                if (iChunkModeActive == null || !iChunkModeActive.IsWritable)
                {
                    Console.WriteLine("Cannot deactive chunk mode. Aborting...");
                    result = -1;
                }

                iChunkModeActive.Value = false;

                Console.WriteLine("Chunk mode deactivated...");
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function acts as the body of the example; please see
        // NodeMapInfo_CSharp example for more in-depth comments on setting up
        // cameras.
        int RunSingleCamera(IManagedCamera cam, string outputPath)
        {
            int result;
            try
            {
                // Retrieve TL device nodemap and print device information
                INodeMap nodeMapTLDevice = cam.GetTLDeviceNodeMap();

                result = PrintDeviceInfo(nodeMapTLDevice);

                // Initialize camera
                cam.Init();

                // Retrieve GenICam nodemap
                INodeMap nodeMap = cam.GetNodeMap();

                // Configure chunk data
                int err = ConfigureChunkData(nodeMap);
                if (err < 0)
                {
                    return err;
                }

                // Acquire images and display chunk data
                result |= AcquireImages(cam, nodeMap, nodeMapTLDevice, outputPath);

                // Disable chunk data
                err = DisableChunkData(nodeMap);
                if (err < 0)
                {
                    return err;
                }

                // Deinitialize camera
                cam.DeInit();
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // Example entry point; please see Enumeration_CSharp example for more
        // in-depth comments on preparing and cleaning up the system.
        static int Main(string[] args)
        {
            int result = 0;

            string strOutputDir = @".\";

            Console.WriteLine("FLIRCollect managed application starting...\n");
            if (args.Length == 0)
            {
                Console.WriteLine("  command line options: \n");
                Console.WriteLine("  FLIRCollect [output_dir] \n");
                Console.WriteLine("  Examples: \n");
                Console.WriteLine(@"    FLIRCollect C:\Users\Administrator\Desktop\dartdata\dataset0" + "\n");
                Console.WriteLine("\n");
            }
            else
            {
                strOutputDir = args[0];
            }
            strOutputDir = Path.Combine(strOutputDir, "flir");
            Directory.CreateDirectory(strOutputDir);

            Program program = new Program();

            // Retrieve singleton reference to system object
            ManagedSystem system = new ManagedSystem();

            // Print out current library version
            LibraryVersion spinVersion = system.GetLibraryVersion();
            Console.WriteLine(
                "Spinnaker library version: {0}.{1}.{2}.{3}\n\n",
                spinVersion.major,
                spinVersion.minor,
                spinVersion.type,
                spinVersion.build);

            // Retrieve list of cameras from the system
            ManagedCameraList camList = system.GetCameras();

            Console.WriteLine("Number of cameras detected: {0}\n\n", camList.Count);

            // Finish if there are no cameras
            if (camList.Count == 0)
            {
                // Clear camera list before releasing system
                camList.Clear();

                // Release system
                system.Dispose();

                Console.WriteLine("Not enough cameras!");
                Console.WriteLine("Done! Press Enter to exit...");
                Console.ReadLine();

                return -1;
            }

            // Run example on each camera
            int i = 0;

            foreach(IManagedCamera managedCamera in camList) using(managedCamera)
            {
                Console.WriteLine("Running example for camera {0}...", i);

                try
                {
                    // Run example
                    result |= program.RunSingleCamera(managedCamera, strOutputDir);
                }
                catch (SpinnakerException ex)
                {
                    Console.WriteLine("Error: {0}", ex.Message);
                    result = -1;
                }

                Console.WriteLine("Camera {0} example complete...\n", i++);
            }

            // Clear camera list before releasing system
            camList.Clear();

            // Release system
            system.Dispose();

            Console.WriteLine("\nDone! Press Enter to exit...");
            Console.ReadLine();

            return result;
        }
    }
}
