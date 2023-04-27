namespace WinFormTestApp
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle1 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle2 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle3 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle4 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle5 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle6 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle7 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle8 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle9 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle10 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle11 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataGridViewCellStyle dataGridViewCellStyle12 = new System.Windows.Forms.DataGridViewCellStyle();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.Title title1 = new System.Windows.Forms.DataVisualization.Charting.Title();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.ID = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.X = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Y = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Z = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Yaw = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Pitch = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Roll = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.InterframeTime = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.FrameDrop = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SystemLatency = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.SoftwareLatency = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TransitLatency = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.TotalLatency = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.Ping = new System.Windows.Forms.DataGridViewTextBoxColumn();
            this.listView1 = new System.Windows.Forms.ListView();
            this.columnHeader1 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.columnHeader2 = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.contextMenuStrip1 = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.menuClear = new System.Windows.Forms.ToolStripMenuItem();
            this.menuPause = new System.Windows.Forms.ToolStripMenuItem();
            this.checkBoxConnect = new System.Windows.Forms.CheckBox();
            this.buttonGetDataDescriptions = new System.Windows.Forms.Button();
            this.chart1 = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.label1 = new System.Windows.Forms.Label();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.SubscribeOnlyCheckBox = new System.Windows.Forms.CheckBox();
            this.RadioBroadcast = new System.Windows.Forms.RadioButton();
            this.LabeledMarkersCheckBox = new System.Windows.Forms.CheckBox();
            this.PollCheckBox = new System.Windows.Forms.CheckBox();
            this.RecordDataButton = new System.Windows.Forms.CheckBox();
            this.DroppedFrameCountLabel = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.TimecodeValue = new System.Windows.Forms.Label();
            this.TimestampValue = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.TimestampLabel = new System.Windows.Forms.Label();
            this.comboBoxLocal = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.Local = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.textBoxServer = new System.Windows.Forms.TextBox();
            this.RadioUnicast = new System.Windows.Forms.RadioButton();
            this.RadioMulticast = new System.Windows.Forms.RadioButton();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.CommandButton = new System.Windows.Forms.Button();
            this.CommandText = new System.Windows.Forms.TextBox();
            this.GetModeButton = new System.Windows.Forms.Button();
            this.GetTakeRangeButton = new System.Windows.Forms.Button();
            this.TestButton = new System.Windows.Forms.Button();
            this.GetLastFrameOfDataButton = new System.Windows.Forms.Button();
            this.SetPlaybackTakeButton = new System.Windows.Forms.Button();
            this.PlaybackTakeNameText = new System.Windows.Forms.TextBox();
            this.StopRecordButton = new System.Windows.Forms.Button();
            this.SetRecordingTakeButton = new System.Windows.Forms.Button();
            this.RecordingTakeNameText = new System.Windows.Forms.TextBox();
            this.TimelineStopButton = new System.Windows.Forms.Button();
            this.LiveModeButton = new System.Windows.Forms.Button();
            this.RecordButton = new System.Windows.Forms.Button();
            this.EditModeButton = new System.Windows.Forms.Button();
            this.TimelinePlayButton = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.DisableAssetButton = new System.Windows.Forms.Button();
            this.EnableAssetButton = new System.Windows.Forms.Button();
            this.GetPropertyButton = new System.Windows.Forms.Button();
            this.label8 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.NodeNameText = new System.Windows.Forms.TextBox();
            this.PropertyNameText = new System.Windows.Forms.TextBox();
            this.PropertyValueText = new System.Windows.Forms.TextBox();
            this.SetPropertyButton = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            this.contextMenuStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chart1)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.panel1.SuspendLayout();
            this.SuspendLayout();
            // 
            // dataGridView1
            // 
            this.dataGridView1.AllowUserToAddRows = false;
            this.dataGridView1.AllowUserToDeleteRows = false;
            this.dataGridView1.AllowUserToResizeRows = false;
            this.dataGridView1.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.Fill;
            dataGridViewCellStyle1.Alignment = System.Windows.Forms.DataGridViewContentAlignment.MiddleLeft;
            dataGridViewCellStyle1.BackColor = System.Drawing.Color.Silver;
            dataGridViewCellStyle1.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            dataGridViewCellStyle1.ForeColor = System.Drawing.SystemColors.WindowText;
            dataGridViewCellStyle1.SelectionBackColor = System.Drawing.SystemColors.Highlight;
            dataGridViewCellStyle1.SelectionForeColor = System.Drawing.SystemColors.HighlightText;
            dataGridViewCellStyle1.WrapMode = System.Windows.Forms.DataGridViewTriState.True;
            this.dataGridView1.ColumnHeadersDefaultCellStyle = dataGridViewCellStyle1;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Columns.AddRange(new System.Windows.Forms.DataGridViewColumn[] {
            this.ID,
            this.X,
            this.Y,
            this.Z,
            this.Yaw,
            this.Pitch,
            this.Roll,
            this.InterframeTime,
            this.FrameDrop,
            this.SystemLatency,
            this.SoftwareLatency,
            this.TransitLatency,
            this.TotalLatency,
            this.Ping});
            this.dataGridView1.EnableHeadersVisualStyles = false;
            this.dataGridView1.Location = new System.Drawing.Point(3, 0);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.ReadOnly = true;
            this.dataGridView1.RowHeadersVisible = false;
            this.dataGridView1.RowTemplate.ReadOnly = true;
            this.dataGridView1.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.dataGridView1.SelectionMode = System.Windows.Forms.DataGridViewSelectionMode.CellSelect;
            this.dataGridView1.Size = new System.Drawing.Size(991, 474);
            this.dataGridView1.TabIndex = 1;
            // 
            // ID
            // 
            this.ID.AutoSizeMode = System.Windows.Forms.DataGridViewAutoSizeColumnMode.None;
            this.ID.FillWeight = 1000F;
            this.ID.HeaderText = "ID";
            this.ID.MinimumWidth = 100;
            this.ID.Name = "ID";
            this.ID.ReadOnly = true;
            this.ID.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.ID.Width = 150;
            // 
            // X
            // 
            dataGridViewCellStyle2.Format = "N2";
            dataGridViewCellStyle2.NullValue = null;
            this.X.DefaultCellStyle = dataGridViewCellStyle2;
            this.X.HeaderText = "X";
            this.X.Name = "X";
            this.X.ReadOnly = true;
            this.X.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // Y
            // 
            dataGridViewCellStyle3.Format = "N2";
            dataGridViewCellStyle3.NullValue = null;
            this.Y.DefaultCellStyle = dataGridViewCellStyle3;
            this.Y.HeaderText = "Y";
            this.Y.Name = "Y";
            this.Y.ReadOnly = true;
            this.Y.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // Z
            // 
            dataGridViewCellStyle4.Format = "N2";
            dataGridViewCellStyle4.NullValue = null;
            this.Z.DefaultCellStyle = dataGridViewCellStyle4;
            this.Z.HeaderText = "Z";
            this.Z.Name = "Z";
            this.Z.ReadOnly = true;
            this.Z.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // Yaw
            // 
            dataGridViewCellStyle5.Format = "N2";
            dataGridViewCellStyle5.NullValue = null;
            this.Yaw.DefaultCellStyle = dataGridViewCellStyle5;
            this.Yaw.HeaderText = "Pitch (X)";
            this.Yaw.Name = "Yaw";
            this.Yaw.ReadOnly = true;
            this.Yaw.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // Pitch
            // 
            dataGridViewCellStyle6.Format = "N2";
            dataGridViewCellStyle6.NullValue = null;
            this.Pitch.DefaultCellStyle = dataGridViewCellStyle6;
            this.Pitch.HeaderText = "Yaw (Y)";
            this.Pitch.Name = "Pitch";
            this.Pitch.ReadOnly = true;
            this.Pitch.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // Roll
            // 
            dataGridViewCellStyle7.Format = "N2";
            dataGridViewCellStyle7.NullValue = null;
            this.Roll.DefaultCellStyle = dataGridViewCellStyle7;
            this.Roll.HeaderText = "Roll (Z)";
            this.Roll.Name = "Roll";
            this.Roll.ReadOnly = true;
            this.Roll.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // InterframeTime
            // 
            dataGridViewCellStyle8.Format = "N2";
            dataGridViewCellStyle8.NullValue = null;
            this.InterframeTime.DefaultCellStyle = dataGridViewCellStyle8;
            this.InterframeTime.HeaderText = "Interframe Time";
            this.InterframeTime.Name = "InterframeTime";
            this.InterframeTime.ReadOnly = true;
            this.InterframeTime.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // FrameDrop
            // 
            dataGridViewCellStyle9.Format = "N2";
            dataGridViewCellStyle9.NullValue = null;
            this.FrameDrop.DefaultCellStyle = dataGridViewCellStyle9;
            this.FrameDrop.HeaderText = "Frame Drops";
            this.FrameDrop.Name = "FrameDrop";
            this.FrameDrop.ReadOnly = true;
            this.FrameDrop.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            // 
            // SystemLatency
            // 
            dataGridViewCellStyle10.Format = "N2";
            this.SystemLatency.DefaultCellStyle = dataGridViewCellStyle10;
            this.SystemLatency.HeaderText = "Motive System Latency";
            this.SystemLatency.Name = "SystemLatency";
            this.SystemLatency.ReadOnly = true;
            this.SystemLatency.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.SystemLatency.ToolTipText = "Camnera Photons -> Motive Transmit";
            // 
            // SoftwareLatency
            // 
            dataGridViewCellStyle11.Format = "N2";
            this.SoftwareLatency.DefaultCellStyle = dataGridViewCellStyle11;
            this.SoftwareLatency.HeaderText = "Motive Software Latency";
            this.SoftwareLatency.Name = "SoftwareLatency";
            this.SoftwareLatency.ReadOnly = true;
            this.SoftwareLatency.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.SoftwareLatency.ToolTipText = "Camera Group -> Motive Transmit";
            // 
            // TransitLatency
            // 
            dataGridViewCellStyle12.Format = "N2";
            this.TransitLatency.DefaultCellStyle = dataGridViewCellStyle12;
            this.TransitLatency.HeaderText = "Transit Latency";
            this.TransitLatency.Name = "TransitLatency";
            this.TransitLatency.ReadOnly = true;
            this.TransitLatency.SortMode = System.Windows.Forms.DataGridViewColumnSortMode.NotSortable;
            this.TransitLatency.ToolTipText = "Motive Transmit -> Client Receive";
            // 
            // TotalLatency
            // 
            this.TotalLatency.HeaderText = "Total Latency";
            this.TotalLatency.Name = "TotalLatency";
            this.TotalLatency.ReadOnly = true;
            this.TotalLatency.ToolTipText = "Camera Photons -> Client Receive";
            // 
            // Ping
            // 
            this.Ping.HeaderText = "Ping";
            this.Ping.Name = "Ping";
            this.Ping.ReadOnly = true;
            // 
            // listView1
            // 
            this.listView1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listView1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.listView1.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2});
            this.listView1.ContextMenuStrip = this.contextMenuStrip1;
            this.listView1.GridLines = true;
            this.listView1.HeaderStyle = System.Windows.Forms.ColumnHeaderStyle.None;
            this.listView1.Location = new System.Drawing.Point(1012, 266);
            this.listView1.Name = "listView1";
            this.listView1.Size = new System.Drawing.Size(383, 220);
            this.listView1.TabIndex = 3;
            this.listView1.UseCompatibleStateImageBehavior = false;
            this.listView1.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Time";
            this.columnHeader1.Width = 80;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Message";
            this.columnHeader2.Width = 400;
            // 
            // contextMenuStrip1
            // 
            this.contextMenuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.menuClear,
            this.menuPause});
            this.contextMenuStrip1.Name = "contextMenuStrip1";
            this.contextMenuStrip1.Size = new System.Drawing.Size(106, 48);
            this.contextMenuStrip1.Opening += new System.ComponentModel.CancelEventHandler(this.contextMenuStrip1_Opening);
            // 
            // menuClear
            // 
            this.menuClear.Name = "menuClear";
            this.menuClear.Size = new System.Drawing.Size(105, 22);
            this.menuClear.Text = "Clear";
            this.menuClear.Click += new System.EventHandler(this.menuClear_Click);
            // 
            // menuPause
            // 
            this.menuPause.CheckOnClick = true;
            this.menuPause.Name = "menuPause";
            this.menuPause.Size = new System.Drawing.Size(105, 22);
            this.menuPause.Text = "Pause";
            this.menuPause.Click += new System.EventHandler(this.menuPause_Click);
            // 
            // checkBoxConnect
            // 
            this.checkBoxConnect.Appearance = System.Windows.Forms.Appearance.Button;
            this.checkBoxConnect.FlatAppearance.CheckedBackColor = System.Drawing.Color.Red;
            this.checkBoxConnect.FlatAppearance.MouseDownBackColor = System.Drawing.Color.White;
            this.checkBoxConnect.Location = new System.Drawing.Point(12, 102);
            this.checkBoxConnect.Name = "checkBoxConnect";
            this.checkBoxConnect.Size = new System.Drawing.Size(80, 23);
            this.checkBoxConnect.TabIndex = 5;
            this.checkBoxConnect.Text = "Connect";
            this.checkBoxConnect.UseVisualStyleBackColor = true;
            this.checkBoxConnect.CheckedChanged += new System.EventHandler(this.checkBoxConnect_CheckedChanged);
            // 
            // buttonGetDataDescriptions
            // 
            this.buttonGetDataDescriptions.Location = new System.Drawing.Point(110, 103);
            this.buttonGetDataDescriptions.Name = "buttonGetDataDescriptions";
            this.buttonGetDataDescriptions.Size = new System.Drawing.Size(122, 23);
            this.buttonGetDataDescriptions.TabIndex = 6;
            this.buttonGetDataDescriptions.Text = "Get Data Descriptions";
            this.buttonGetDataDescriptions.UseVisualStyleBackColor = true;
            this.buttonGetDataDescriptions.Click += new System.EventHandler(this.buttonGetDataDescriptions_Click);
            // 
            // chart1
            // 
            this.chart1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.chart1.BackColor = System.Drawing.SystemColors.AppWorkspace;
            this.chart1.BackGradientStyle = System.Windows.Forms.DataVisualization.Charting.GradientStyle.TopBottom;
            this.chart1.BackSecondaryColor = System.Drawing.SystemColors.ButtonHighlight;
            this.chart1.BorderlineColor = System.Drawing.Color.Black;
            this.chart1.BorderlineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.Solid;
            this.chart1.BorderSkin.BackGradientStyle = System.Windows.Forms.DataVisualization.Charting.GradientStyle.TopBottom;
            this.chart1.BorderSkin.BorderDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.Solid;
            this.chart1.BorderSkin.BorderWidth = 5;
            this.chart1.BorderSkin.PageColor = System.Drawing.SystemColors.ButtonFace;
            chartArea1.AxisX.MinorTickMark.Enabled = true;
            chartArea1.AxisX.Title = "Frame";
            chartArea1.AxisY.IsLabelAutoFit = false;
            chartArea1.AxisY.MajorGrid.LineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.Dash;
            chartArea1.BackColor = System.Drawing.Color.LightGray;
            chartArea1.BackGradientStyle = System.Windows.Forms.DataVisualization.Charting.GradientStyle.TopBottom;
            chartArea1.Name = "ChartArea1";
            this.chart1.ChartAreas.Add(chartArea1);
            legend1.BackColor = System.Drawing.Color.Transparent;
            legend1.BorderColor = System.Drawing.Color.Transparent;
            legend1.DockedToChartArea = "ChartArea1";
            legend1.Name = "Legend1";
            legend1.Position.Auto = false;
            legend1.Position.Height = 10F;
            legend1.Position.Width = 11.48825F;
            legend1.Position.X = 84.04308F;
            legend1.Position.Y = 4F;
            this.chart1.Legends.Add(legend1);
            this.chart1.Location = new System.Drawing.Point(12, 492);
            this.chart1.Name = "chart1";
            series1.ChartArea = "ChartArea1";
            series1.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.FastLine;
            series1.IsValueShownAsLabel = true;
            series1.Label = "value";
            series1.LabelToolTip = "value";
            series1.Legend = "Legend1";
            series1.MarkerBorderColor = System.Drawing.Color.Transparent;
            series1.MarkerColor = System.Drawing.Color.Black;
            series1.MarkerStyle = System.Windows.Forms.DataVisualization.Charting.MarkerStyle.Circle;
            series1.Name = "Series1";
            this.chart1.Series.Add(series1);
            this.chart1.Size = new System.Drawing.Size(1383, 415);
            this.chart1.TabIndex = 12;
            this.chart1.Text = "chart1";
            title1.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            title1.Name = "NatNetData";
            title1.Text = "NatNet Demo";
            title1.TextStyle = System.Windows.Forms.DataVisualization.Charting.TextStyle.Shadow;
            this.chart1.Titles.Add(title1);
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.label1.BackColor = System.Drawing.Color.DarkGray;
            this.label1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 10F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.SystemColors.ControlText;
            this.label1.Location = new System.Drawing.Point(1012, 246);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(383, 21);
            this.label1.TabIndex = 13;
            this.label1.Text = "Messages";
            // 
            // tabControl1
            // 
            this.tabControl1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Location = new System.Drawing.Point(1012, 12);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(383, 227);
            this.tabControl1.TabIndex = 21;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.SubscribeOnlyCheckBox);
            this.tabPage1.Controls.Add(this.RadioBroadcast);
            this.tabPage1.Controls.Add(this.LabeledMarkersCheckBox);
            this.tabPage1.Controls.Add(this.PollCheckBox);
            this.tabPage1.Controls.Add(this.RecordDataButton);
            this.tabPage1.Controls.Add(this.DroppedFrameCountLabel);
            this.tabPage1.Controls.Add(this.label6);
            this.tabPage1.Controls.Add(this.TimecodeValue);
            this.tabPage1.Controls.Add(this.TimestampValue);
            this.tabPage1.Controls.Add(this.label4);
            this.tabPage1.Controls.Add(this.TimestampLabel);
            this.tabPage1.Controls.Add(this.comboBoxLocal);
            this.tabPage1.Controls.Add(this.label3);
            this.tabPage1.Controls.Add(this.checkBoxConnect);
            this.tabPage1.Controls.Add(this.Local);
            this.tabPage1.Controls.Add(this.label2);
            this.tabPage1.Controls.Add(this.textBoxServer);
            this.tabPage1.Controls.Add(this.RadioUnicast);
            this.tabPage1.Controls.Add(this.RadioMulticast);
            this.tabPage1.Controls.Add(this.buttonGetDataDescriptions);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(375, 201);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Connect";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // SubscribeOnlyCheckBox
            // 
            this.SubscribeOnlyCheckBox.AutoSize = true;
            this.SubscribeOnlyCheckBox.Location = new System.Drawing.Point(217, 50);
            this.SubscribeOnlyCheckBox.Name = "SubscribeOnlyCheckBox";
            this.SubscribeOnlyCheckBox.Size = new System.Drawing.Size(97, 17);
            this.SubscribeOnlyCheckBox.TabIndex = 32;
            this.SubscribeOnlyCheckBox.Text = "Subscribe Only";
            this.SubscribeOnlyCheckBox.UseVisualStyleBackColor = true;
            this.SubscribeOnlyCheckBox.CheckedChanged += new System.EventHandler(this.SubscribeButton_CheckChanged);
            // 
            // RadioBroadcast
            // 
            this.RadioBroadcast.AutoSize = true;
            this.RadioBroadcast.Location = new System.Drawing.Point(197, 73);
            this.RadioBroadcast.Name = "RadioBroadcast";
            this.RadioBroadcast.Size = new System.Drawing.Size(73, 17);
            this.RadioBroadcast.TabIndex = 31;
            this.RadioBroadcast.Text = "Broadcast";
            this.RadioBroadcast.UseVisualStyleBackColor = true;
            // 
            // LabeledMarkersCheckBox
            // 
            this.LabeledMarkersCheckBox.AutoSize = true;
            this.LabeledMarkersCheckBox.Location = new System.Drawing.Point(217, 30);
            this.LabeledMarkersCheckBox.Name = "LabeledMarkersCheckBox";
            this.LabeledMarkersCheckBox.Size = new System.Drawing.Size(64, 17);
            this.LabeledMarkersCheckBox.TabIndex = 30;
            this.LabeledMarkersCheckBox.Text = "Markers";
            this.LabeledMarkersCheckBox.UseVisualStyleBackColor = true;
            // 
            // PollCheckBox
            // 
            this.PollCheckBox.AutoSize = true;
            this.PollCheckBox.Location = new System.Drawing.Point(217, 10);
            this.PollCheckBox.Name = "PollCheckBox";
            this.PollCheckBox.Size = new System.Drawing.Size(43, 17);
            this.PollCheckBox.TabIndex = 4;
            this.PollCheckBox.Text = "Poll";
            this.PollCheckBox.UseVisualStyleBackColor = true;
            this.PollCheckBox.CheckedChanged += new System.EventHandler(this.PollCheckBox_CheckedChanged);
            // 
            // RecordDataButton
            // 
            this.RecordDataButton.Appearance = System.Windows.Forms.Appearance.Button;
            this.RecordDataButton.FlatAppearance.CheckedBackColor = System.Drawing.Color.Red;
            this.RecordDataButton.FlatAppearance.MouseDownBackColor = System.Drawing.Color.White;
            this.RecordDataButton.Location = new System.Drawing.Point(247, 103);
            this.RecordDataButton.Name = "RecordDataButton";
            this.RecordDataButton.Size = new System.Drawing.Size(80, 23);
            this.RecordDataButton.TabIndex = 7;
            this.RecordDataButton.Text = "Record";
            this.RecordDataButton.UseVisualStyleBackColor = true;
            this.RecordDataButton.CheckedChanged += new System.EventHandler(this.RecordDataButton_CheckedChanged);
            // 
            // DroppedFrameCountLabel
            // 
            this.DroppedFrameCountLabel.AutoSize = true;
            this.DroppedFrameCountLabel.Location = new System.Drawing.Point(272, 140);
            this.DroppedFrameCountLabel.Name = "DroppedFrameCountLabel";
            this.DroppedFrameCountLabel.Size = new System.Drawing.Size(43, 13);
            this.DroppedFrameCountLabel.TabIndex = 23;
            this.DroppedFrameCountLabel.Text = "<none>";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(177, 140);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(94, 13);
            this.label6.TabIndex = 22;
            this.label6.Text = "Dropped Frames : ";
            // 
            // TimecodeValue
            // 
            this.TimecodeValue.AutoSize = true;
            this.TimecodeValue.Location = new System.Drawing.Point(89, 163);
            this.TimecodeValue.Name = "TimecodeValue";
            this.TimecodeValue.Size = new System.Drawing.Size(43, 13);
            this.TimecodeValue.TabIndex = 21;
            this.TimecodeValue.Text = "<none>";
            // 
            // TimestampValue
            // 
            this.TimestampValue.AutoSize = true;
            this.TimestampValue.Location = new System.Drawing.Point(89, 140);
            this.TimestampValue.Name = "TimestampValue";
            this.TimestampValue.Size = new System.Drawing.Size(43, 13);
            this.TimestampValue.TabIndex = 20;
            this.TimestampValue.Text = "<none>";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(11, 163);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(60, 13);
            this.label4.TabIndex = 19;
            this.label4.Text = "Timecode :";
            // 
            // TimestampLabel
            // 
            this.TimestampLabel.AutoSize = true;
            this.TimestampLabel.Location = new System.Drawing.Point(11, 140);
            this.TimestampLabel.Name = "TimestampLabel";
            this.TimestampLabel.Size = new System.Drawing.Size(67, 13);
            this.TimestampLabel.TabIndex = 18;
            this.TimestampLabel.Text = "Timestamp : ";
            // 
            // comboBoxLocal
            // 
            this.comboBoxLocal.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxLocal.FormattingEnabled = true;
            this.comboBoxLocal.Location = new System.Drawing.Point(58, 15);
            this.comboBoxLocal.Name = "comboBoxLocal";
            this.comboBoxLocal.Size = new System.Drawing.Size(121, 21);
            this.comboBoxLocal.TabIndex = 0;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(9, 75);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(31, 13);
            this.label3.TabIndex = 16;
            this.label3.Text = "Type";
            // 
            // Local
            // 
            this.Local.AutoSize = true;
            this.Local.Location = new System.Drawing.Point(9, 18);
            this.Local.Name = "Local";
            this.Local.Size = new System.Drawing.Size(33, 13);
            this.Local.TabIndex = 9;
            this.Local.Text = "Local";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(9, 48);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(38, 13);
            this.label2.TabIndex = 10;
            this.label2.Text = "Server";
            // 
            // textBoxServer
            // 
            this.textBoxServer.Location = new System.Drawing.Point(58, 45);
            this.textBoxServer.Name = "textBoxServer";
            this.textBoxServer.Size = new System.Drawing.Size(121, 20);
            this.textBoxServer.TabIndex = 1;
            this.textBoxServer.Text = "127.0.0.1";
            // 
            // RadioUnicast
            // 
            this.RadioUnicast.AutoSize = true;
            this.RadioUnicast.Location = new System.Drawing.Point(130, 73);
            this.RadioUnicast.Name = "RadioUnicast";
            this.RadioUnicast.Size = new System.Drawing.Size(61, 17);
            this.RadioUnicast.TabIndex = 3;
            this.RadioUnicast.Text = "Unicast";
            this.RadioUnicast.UseVisualStyleBackColor = true;
            this.RadioUnicast.CheckedChanged += new System.EventHandler(this.RadioUnicast_CheckedChanged);
            // 
            // RadioMulticast
            // 
            this.RadioMulticast.AutoSize = true;
            this.RadioMulticast.Checked = true;
            this.RadioMulticast.Location = new System.Drawing.Point(58, 73);
            this.RadioMulticast.Name = "RadioMulticast";
            this.RadioMulticast.Size = new System.Drawing.Size(67, 17);
            this.RadioMulticast.TabIndex = 2;
            this.RadioMulticast.TabStop = true;
            this.RadioMulticast.Text = "Multicast";
            this.RadioMulticast.UseVisualStyleBackColor = true;
            this.RadioMulticast.CheckedChanged += new System.EventHandler(this.RadioMulticast_CheckedChanged);
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.CommandButton);
            this.tabPage2.Controls.Add(this.CommandText);
            this.tabPage2.Controls.Add(this.GetModeButton);
            this.tabPage2.Controls.Add(this.GetTakeRangeButton);
            this.tabPage2.Controls.Add(this.TestButton);
            this.tabPage2.Controls.Add(this.GetLastFrameOfDataButton);
            this.tabPage2.Controls.Add(this.SetPlaybackTakeButton);
            this.tabPage2.Controls.Add(this.PlaybackTakeNameText);
            this.tabPage2.Controls.Add(this.StopRecordButton);
            this.tabPage2.Controls.Add(this.SetRecordingTakeButton);
            this.tabPage2.Controls.Add(this.RecordingTakeNameText);
            this.tabPage2.Controls.Add(this.TimelineStopButton);
            this.tabPage2.Controls.Add(this.LiveModeButton);
            this.tabPage2.Controls.Add(this.RecordButton);
            this.tabPage2.Controls.Add(this.EditModeButton);
            this.tabPage2.Controls.Add(this.TimelinePlayButton);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(375, 201);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Commands";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // CommandButton
            // 
            this.CommandButton.Location = new System.Drawing.Point(11, 70);
            this.CommandButton.Name = "CommandButton";
            this.CommandButton.Size = new System.Drawing.Size(149, 23);
            this.CommandButton.TabIndex = 14;
            this.CommandButton.Text = "Command";
            this.CommandButton.UseVisualStyleBackColor = true;
            this.CommandButton.Click += new System.EventHandler(this.CommandButton_Click);
            // 
            // CommandText
            // 
            this.CommandText.Location = new System.Drawing.Point(166, 72);
            this.CommandText.Name = "CommandText";
            this.CommandText.Size = new System.Drawing.Size(183, 20);
            this.CommandText.TabIndex = 15;
            // 
            // GetModeButton
            // 
            this.GetModeButton.Location = new System.Drawing.Point(296, 137);
            this.GetModeButton.Name = "GetModeButton";
            this.GetModeButton.Size = new System.Drawing.Size(62, 23);
            this.GetModeButton.TabIndex = 13;
            this.GetModeButton.Text = "Mode?";
            this.GetModeButton.UseVisualStyleBackColor = true;
            this.GetModeButton.Click += new System.EventHandler(this.GetModeButton_Click);
            // 
            // GetTakeRangeButton
            // 
            this.GetTakeRangeButton.Location = new System.Drawing.Point(296, 166);
            this.GetTakeRangeButton.Name = "GetTakeRangeButton";
            this.GetTakeRangeButton.Size = new System.Drawing.Size(62, 23);
            this.GetTakeRangeButton.TabIndex = 12;
            this.GetTakeRangeButton.Text = "Range?";
            this.GetTakeRangeButton.UseVisualStyleBackColor = true;
            this.GetTakeRangeButton.Click += new System.EventHandler(this.GetTakeRangeButton_Click);
            // 
            // TestButton
            // 
            this.TestButton.Location = new System.Drawing.Point(227, 137);
            this.TestButton.Name = "TestButton";
            this.TestButton.Size = new System.Drawing.Size(63, 23);
            this.TestButton.TabIndex = 10;
            this.TestButton.Text = "Test";
            this.TestButton.UseVisualStyleBackColor = true;
            this.TestButton.Click += new System.EventHandler(this.TestButton_Click);
            // 
            // GetLastFrameOfDataButton
            // 
            this.GetLastFrameOfDataButton.Location = new System.Drawing.Point(227, 166);
            this.GetLastFrameOfDataButton.Name = "GetLastFrameOfDataButton";
            this.GetLastFrameOfDataButton.Size = new System.Drawing.Size(63, 23);
            this.GetLastFrameOfDataButton.TabIndex = 11;
            this.GetLastFrameOfDataButton.Text = "Frame?";
            this.GetLastFrameOfDataButton.UseVisualStyleBackColor = true;
            this.GetLastFrameOfDataButton.Click += new System.EventHandler(this.GetLastFrameOfDataButton_Click);
            // 
            // SetPlaybackTakeButton
            // 
            this.SetPlaybackTakeButton.Location = new System.Drawing.Point(11, 39);
            this.SetPlaybackTakeButton.Name = "SetPlaybackTakeButton";
            this.SetPlaybackTakeButton.Size = new System.Drawing.Size(149, 23);
            this.SetPlaybackTakeButton.TabIndex = 2;
            this.SetPlaybackTakeButton.Text = "Set Playback Take Name";
            this.SetPlaybackTakeButton.UseVisualStyleBackColor = true;
            this.SetPlaybackTakeButton.Click += new System.EventHandler(this.SetPlaybackTakeButton_Click);
            // 
            // PlaybackTakeNameText
            // 
            this.PlaybackTakeNameText.Location = new System.Drawing.Point(166, 41);
            this.PlaybackTakeNameText.Name = "PlaybackTakeNameText";
            this.PlaybackTakeNameText.Size = new System.Drawing.Size(183, 20);
            this.PlaybackTakeNameText.TabIndex = 3;
            // 
            // StopRecordButton
            // 
            this.StopRecordButton.Location = new System.Drawing.Point(122, 104);
            this.StopRecordButton.Name = "StopRecordButton";
            this.StopRecordButton.Size = new System.Drawing.Size(99, 23);
            this.StopRecordButton.TabIndex = 5;
            this.StopRecordButton.Text = "Stop Record";
            this.StopRecordButton.UseVisualStyleBackColor = true;
            this.StopRecordButton.Click += new System.EventHandler(this.StopRecordButton_Click);
            // 
            // SetRecordingTakeButton
            // 
            this.SetRecordingTakeButton.Location = new System.Drawing.Point(11, 10);
            this.SetRecordingTakeButton.Name = "SetRecordingTakeButton";
            this.SetRecordingTakeButton.Size = new System.Drawing.Size(149, 23);
            this.SetRecordingTakeButton.TabIndex = 0;
            this.SetRecordingTakeButton.Text = "Set Recording Take Name";
            this.SetRecordingTakeButton.UseVisualStyleBackColor = true;
            this.SetRecordingTakeButton.Click += new System.EventHandler(this.SetRecordingTakeButton_Click);
            // 
            // RecordingTakeNameText
            // 
            this.RecordingTakeNameText.Location = new System.Drawing.Point(166, 12);
            this.RecordingTakeNameText.Name = "RecordingTakeNameText";
            this.RecordingTakeNameText.Size = new System.Drawing.Size(183, 20);
            this.RecordingTakeNameText.TabIndex = 1;
            // 
            // TimelineStopButton
            // 
            this.TimelineStopButton.Location = new System.Drawing.Point(122, 166);
            this.TimelineStopButton.Name = "TimelineStopButton";
            this.TimelineStopButton.Size = new System.Drawing.Size(99, 23);
            this.TimelineStopButton.TabIndex = 9;
            this.TimelineStopButton.Text = "Timeline Stop";
            this.TimelineStopButton.UseVisualStyleBackColor = true;
            this.TimelineStopButton.Click += new System.EventHandler(this.TimelineStopButton_Click);
            // 
            // LiveModeButton
            // 
            this.LiveModeButton.Location = new System.Drawing.Point(11, 135);
            this.LiveModeButton.Name = "LiveModeButton";
            this.LiveModeButton.Size = new System.Drawing.Size(99, 23);
            this.LiveModeButton.TabIndex = 6;
            this.LiveModeButton.Text = "Live Mode";
            this.LiveModeButton.UseVisualStyleBackColor = true;
            this.LiveModeButton.Click += new System.EventHandler(this.LiveModeButton_Click);
            // 
            // RecordButton
            // 
            this.RecordButton.Location = new System.Drawing.Point(11, 104);
            this.RecordButton.Name = "RecordButton";
            this.RecordButton.Size = new System.Drawing.Size(99, 23);
            this.RecordButton.TabIndex = 4;
            this.RecordButton.Text = "Record";
            this.RecordButton.UseVisualStyleBackColor = true;
            this.RecordButton.Click += new System.EventHandler(this.RecordButton_Click);
            // 
            // EditModeButton
            // 
            this.EditModeButton.Location = new System.Drawing.Point(122, 135);
            this.EditModeButton.Name = "EditModeButton";
            this.EditModeButton.Size = new System.Drawing.Size(99, 23);
            this.EditModeButton.TabIndex = 7;
            this.EditModeButton.Text = "Edit Mode";
            this.EditModeButton.UseVisualStyleBackColor = true;
            this.EditModeButton.Click += new System.EventHandler(this.EditModeButton_Click);
            // 
            // TimelinePlayButton
            // 
            this.TimelinePlayButton.Location = new System.Drawing.Point(11, 166);
            this.TimelinePlayButton.Name = "TimelinePlayButton";
            this.TimelinePlayButton.Size = new System.Drawing.Size(99, 23);
            this.TimelinePlayButton.TabIndex = 8;
            this.TimelinePlayButton.Text = "Timeline Play";
            this.TimelinePlayButton.UseVisualStyleBackColor = true;
            this.TimelinePlayButton.Click += new System.EventHandler(this.TimelinePlayButton_Click);
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.DisableAssetButton);
            this.tabPage3.Controls.Add(this.EnableAssetButton);
            this.tabPage3.Controls.Add(this.GetPropertyButton);
            this.tabPage3.Controls.Add(this.label8);
            this.tabPage3.Controls.Add(this.label7);
            this.tabPage3.Controls.Add(this.label5);
            this.tabPage3.Controls.Add(this.NodeNameText);
            this.tabPage3.Controls.Add(this.PropertyNameText);
            this.tabPage3.Controls.Add(this.PropertyValueText);
            this.tabPage3.Controls.Add(this.SetPropertyButton);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(375, 201);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Properties";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // DisableAssetButton
            // 
            this.DisableAssetButton.Location = new System.Drawing.Point(112, 129);
            this.DisableAssetButton.Name = "DisableAssetButton";
            this.DisableAssetButton.Size = new System.Drawing.Size(98, 23);
            this.DisableAssetButton.TabIndex = 6;
            this.DisableAssetButton.Text = "Disable Asset";
            this.DisableAssetButton.UseVisualStyleBackColor = true;
            this.DisableAssetButton.Click += new System.EventHandler(this.DisableAssetButton_Click);
            // 
            // EnableAssetButton
            // 
            this.EnableAssetButton.Location = new System.Drawing.Point(9, 129);
            this.EnableAssetButton.Name = "EnableAssetButton";
            this.EnableAssetButton.Size = new System.Drawing.Size(98, 23);
            this.EnableAssetButton.TabIndex = 5;
            this.EnableAssetButton.Text = "Enable Asset";
            this.EnableAssetButton.UseVisualStyleBackColor = true;
            this.EnableAssetButton.Click += new System.EventHandler(this.EnableAssetButton_Click);
            // 
            // GetPropertyButton
            // 
            this.GetPropertyButton.Location = new System.Drawing.Point(112, 100);
            this.GetPropertyButton.Name = "GetPropertyButton";
            this.GetPropertyButton.Size = new System.Drawing.Size(98, 23);
            this.GetPropertyButton.TabIndex = 4;
            this.GetPropertyButton.Text = "Get Property";
            this.GetPropertyButton.UseVisualStyleBackColor = true;
            this.GetPropertyButton.Click += new System.EventHandler(this.GetPropertyButton_Click);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(6, 38);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(46, 13);
            this.label8.TabIndex = 40;
            this.label8.Text = "Property";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(6, 64);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(34, 13);
            this.label7.TabIndex = 39;
            this.label7.Text = "Value";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(6, 13);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(33, 13);
            this.label5.TabIndex = 38;
            this.label5.Text = "Asset";
            // 
            // NodeNameText
            // 
            this.NodeNameText.Location = new System.Drawing.Point(74, 9);
            this.NodeNameText.Name = "NodeNameText";
            this.NodeNameText.Size = new System.Drawing.Size(136, 20);
            this.NodeNameText.TabIndex = 0;
            // 
            // PropertyNameText
            // 
            this.PropertyNameText.Location = new System.Drawing.Point(74, 35);
            this.PropertyNameText.Name = "PropertyNameText";
            this.PropertyNameText.Size = new System.Drawing.Size(136, 20);
            this.PropertyNameText.TabIndex = 1;
            // 
            // PropertyValueText
            // 
            this.PropertyValueText.Location = new System.Drawing.Point(74, 61);
            this.PropertyValueText.Name = "PropertyValueText";
            this.PropertyValueText.Size = new System.Drawing.Size(136, 20);
            this.PropertyValueText.TabIndex = 2;
            // 
            // SetPropertyButton
            // 
            this.SetPropertyButton.Location = new System.Drawing.Point(9, 100);
            this.SetPropertyButton.Name = "SetPropertyButton";
            this.SetPropertyButton.Size = new System.Drawing.Size(98, 23);
            this.SetPropertyButton.TabIndex = 3;
            this.SetPropertyButton.Text = "Set Property";
            this.SetPropertyButton.UseVisualStyleBackColor = true;
            this.SetPropertyButton.Click += new System.EventHandler(this.SetPropertyButton_Click);
            // 
            // panel1
            // 
            this.panel1.AutoScroll = true;
            this.panel1.Controls.Add(this.dataGridView1);
            this.panel1.Location = new System.Drawing.Point(12, 12);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(998, 474);
            this.panel1.TabIndex = 22;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.Gainsboro;
            this.ClientSize = new System.Drawing.Size(1407, 919);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.chart1);
            this.Controls.Add(this.listView1);
            this.Controls.Add(this.panel1);
            this.Name = "Form1";
            this.Text = "NatNet Managed Client Sample";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            this.contextMenuStrip1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chart1)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.DataGridView dataGridView1;
        private System.Windows.Forms.ListView listView1;
        private System.Windows.Forms.ColumnHeader columnHeader1;
        private System.Windows.Forms.CheckBox checkBoxConnect;
        private System.Windows.Forms.ColumnHeader columnHeader2;
        private System.Windows.Forms.Button buttonGetDataDescriptions;
        private System.Windows.Forms.DataVisualization.Charting.Chart chart1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.Label Local;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textBoxServer;
        private System.Windows.Forms.RadioButton RadioUnicast;
        private System.Windows.Forms.RadioButton RadioMulticast;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.ComboBox comboBoxLocal;
        private System.Windows.Forms.Label TimestampLabel;
        private System.Windows.Forms.Label TimecodeValue;
        private System.Windows.Forms.Label TimestampValue;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip1;
        private System.Windows.Forms.ToolStripMenuItem menuClear;
        private System.Windows.Forms.ToolStripMenuItem menuPause;
        private System.Windows.Forms.Label DroppedFrameCountLabel;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.CheckBox RecordDataButton;
        private System.Windows.Forms.CheckBox PollCheckBox;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.Button GetModeButton;
        private System.Windows.Forms.Button GetTakeRangeButton;
        private System.Windows.Forms.Button TestButton;
        private System.Windows.Forms.Button GetLastFrameOfDataButton;
        private System.Windows.Forms.Button SetPlaybackTakeButton;
        private System.Windows.Forms.TextBox PlaybackTakeNameText;
        private System.Windows.Forms.Button StopRecordButton;
        private System.Windows.Forms.Button SetRecordingTakeButton;
        private System.Windows.Forms.TextBox RecordingTakeNameText;
        private System.Windows.Forms.Button TimelineStopButton;
        private System.Windows.Forms.Button LiveModeButton;
        private System.Windows.Forms.Button RecordButton;
        private System.Windows.Forms.Button EditModeButton;
        private System.Windows.Forms.Button TimelinePlayButton;
        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.Button DisableAssetButton;
        private System.Windows.Forms.Button EnableAssetButton;
        private System.Windows.Forms.Button GetPropertyButton;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox NodeNameText;
        private System.Windows.Forms.TextBox PropertyNameText;
        private System.Windows.Forms.TextBox PropertyValueText;
        private System.Windows.Forms.Button SetPropertyButton;
        private System.Windows.Forms.CheckBox LabeledMarkersCheckBox;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.DataGridViewTextBoxColumn ID;
        private System.Windows.Forms.DataGridViewTextBoxColumn X;
        private System.Windows.Forms.DataGridViewTextBoxColumn Y;
        private System.Windows.Forms.DataGridViewTextBoxColumn Z;
        private System.Windows.Forms.DataGridViewTextBoxColumn Yaw;
        private System.Windows.Forms.DataGridViewTextBoxColumn Pitch;
        private System.Windows.Forms.DataGridViewTextBoxColumn Roll;
        private System.Windows.Forms.DataGridViewTextBoxColumn InterframeTime;
        private System.Windows.Forms.DataGridViewTextBoxColumn FrameDrop;
        private System.Windows.Forms.DataGridViewTextBoxColumn SystemLatency;
        private System.Windows.Forms.DataGridViewTextBoxColumn SoftwareLatency;
        private System.Windows.Forms.DataGridViewTextBoxColumn TransitLatency;
        private System.Windows.Forms.DataGridViewTextBoxColumn TotalLatency;
        private System.Windows.Forms.DataGridViewTextBoxColumn Ping;
        private System.Windows.Forms.RadioButton RadioBroadcast;
        private System.Windows.Forms.Button CommandButton;
        private System.Windows.Forms.TextBox CommandText;
        private System.Windows.Forms.CheckBox SubscribeOnlyCheckBox;
    }
}

