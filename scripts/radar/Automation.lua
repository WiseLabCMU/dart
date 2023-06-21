---------------------------------- STARTUP -------------------------------------
------------------------ DO NOT MODIFY THIS SECTION ----------------------------

-- mmwavestudio installation path
RSTD_PATH = RSTD.GetRstdPath()

-- Declare the loading function
dofile(RSTD_PATH .. "\\Scripts\\Startup.lua")

----------------------------------- CONFIG -------------------------------------

ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(5,115200,1000)
ar1.Calling_IsConnected()
ar1.SelectChipVersion("AR1642")
ar1.SelectChipVersion("AR1642")
ar1.deviceVariantSelection("XWR1843")
ar1.frequencyBandSelection("77G")
ar1.SelectChipVersion("XWR1843")
ar1.DownloadBSSFw("C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr18xx_radarss.bin")
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()
ar1.DownloadMSSFw("C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr18xx_masterss.bin")
ar1.GetMSSFwVersion()
ar1.PowerOn(0, 1000, 0, 0)
ar1.SelectChipVersion("AR1642")
ar1.SelectChipVersion("XWR1843")
ar1.RfEnable()
ar1.GetMSSFwVersion()
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 2, 0)
ar1.LPModConfig(0, 1)
ar1.RfInit()
ar1.SetCalMonFreqLimitConfig(77,81)
ar1.DataPathConfig(513, 1216644097, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)
ar1.ProfileConfig(0, 77, 10, 6, 63.14, 0, 0, 0, 0, 0, 0, 63.343, 1, 512, 9121, 0, 0, 30)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
-- ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
ar1.DisableTestSource(0)
ar1.FrameConfig(0, 1, 0, 1, 0.5, 0, 0, 1)
ar1.GetCaptureCardDllVersion()
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(25)
ar1.GetCaptureCardFPGAVersion()

------------------------- Start the capture ------------------------------------

RSTD.Sleep(15000)
ar1.CaptureCardConfig_StartRecord("C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data.bin", 1)
RSTD.Sleep(1000)
ar1.StartFrame()
RSTD.Sleep(1000)

------------------------- End the capture --------------------------------------

-- ar1.StopFrame()
-- ar1.PowerOff()
-- ar1.Disconnect()

os.exit()

-- end
