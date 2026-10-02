function frequency_modulation_piecewise_consolidated_gui
% FREQUENCY_MODULATION_PIECEWISE_CONSOLIDATED_GUI
% Consolidated open-loop piecewise frequency command editor and 2-DOF simulator.
% Uses classic MATLAB graphics (uicontrol/uipanel) for cross-version compatibility.
%
% Features:
%   - Open-loop piecewise frequency trajectory generation (Hold, Polynomial, Exponential)
%   - 2-DOF coupled dynamics: Spin synchronization (delta, omega) + Vertical motion (z, z_dot)
%   - Pure feedforward control (No active control loops or state feedback)

%% Fixed Robot Physical Parameters
I_robot = 3.89E-9 + 2.0 * ...
    (1.0/12.0 * 1.17832E-5 * ...
    (3.0 * 0.79375^2.0 * 1E-6 + 0.79375^2.0 * 1E-6) ...
    + 1.17832E-5 * (0.496875^2.0 * 1E-6));

fre_points = (10:10:230)';
drag_torque_points = [
    -7.20311E-08; -8.54759E-07; -1.18389E-06; -1.55412E-06; -2.06800E-06;
    -2.56102E-06; -2.88399E-06; -3.59653E-06; -3.83331E-06; -4.99189E-06;
    -5.72497E-06; -6.29224E-06; -7.17918E-06; -8.19104E-06; -9.07986E-06;
    -1.00964E-05; -1.08262E-05; -1.21433E-05; -1.34310E-05; -1.48856E-05;
    -1.68245E-05; -1.88440E-05; -2.10429E-05
    ];

f2 = fre_points.^2;
k_drag = sum(f2 .* (-drag_torque_points)) / sum(f2.^2);
drag_fit = -k_drag .* fre_points.^2;
SS_res = sum((drag_torque_points - drag_fit).^2);
SS_tot = sum((drag_torque_points - mean(drag_torque_points)).^2);
R_squared = 1 - SS_res / SS_tot;

exampleData = {
    'Polynomial',  160, 140, 0.20, 1;
    'Hold',        140, 140, 1.00, 0;
    'Exponential', 140, 180, 0.50, 4
    };

selectedRow = 1;

%% GUI Layout Setup
fig = figure( ...
    'Name', 'Flying Robot Open-Loop Piecewise Frequency GUI', ...
    'NumberTitle', 'off', ...
    'MenuBar', 'none', ...
    'ToolBar', 'none', ...
    'Color', [0.94 0.94 0.94], ...
    'Units', 'normalized', ...
    'Position', [0.025 0.055 0.95 0.87], ...
    'Resize', 'on');

leftPanel = uipanel( ...
    'Parent', fig, ...
    'Title', 'Piecewise Frequency Command Editor', ...
    'FontWeight', 'bold', ...
    'Units', 'normalized', ...
    'Position', [0.010 0.020 0.405 0.965]);

rightPanel = uipanel( ...
    'Parent', fig, ...
    'Title', 'Open-Loop Response & Dynamics', ...
    'FontWeight', 'bold', ...
    'Units', 'normalized', ...
    'Position', [0.425 0.020 0.565 0.965]);

%% Left Panel: Title & Global Settings
uicontrol( ...
    'Parent', leftPanel, ...
    'Style', 'text', ...
    'String', 'Build the external-field frequency one segment at a time', ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', get(leftPanel, 'BackgroundColor'), ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'center', ...
    'Position', [0.025 0.935 0.950 0.040]);

settingsPanel = uipanel( ...
    'Parent', leftPanel, ...
    'Title', 'Global Model Settings', ...
    'Units', 'normalized', ...
    'Position', [0.025 0.725 0.950 0.195]);

marginTip = 'Torque headroom M = max magnetic torque / drag at the first Start freq. M >= 1.';
uicontrol('Parent', settingsPanel, 'Style', 'text', 'String', 'Torque margin', ...
    'HorizontalAlignment', 'left', 'TooltipString', marginTip, ...
    'Units', 'normalized', 'Position', [0.020 0.680 0.160 0.180]);
marginEdit = uicontrol('Parent', settingsPanel, 'Style', 'edit', 'String', '5', ...
    'BackgroundColor', 'white', 'TooltipString', marginTip, ...
    'Units', 'normalized', 'Position', [0.175 0.705 0.105 0.180]);

tolTip = 'Hold pass/fail band: max allowed error in last 20% of Hold segment.';
uicontrol('Parent', settingsPanel, 'Style', 'text', 'String', 'Tolerance (Hz)', ...
    'HorizontalAlignment', 'left', 'TooltipString', tolTip, ...
    'Units', 'normalized', 'Position', [0.310 0.680 0.160 0.180]);
toleranceEdit = uicontrol('Parent', settingsPanel, 'Style', 'edit', 'String', '1', ...
    'BackgroundColor', 'white', 'TooltipString', tolTip, ...
    'Units', 'normalized', 'Position', [0.465 0.705 0.105 0.180]);

autoChainCheck = uicontrol('Parent', settingsPanel, 'Style', 'checkbox', ...
    'String', 'Auto-chain segments', 'Value', 1, ...
    'TooltipString', 'Automatically match segment Start to previous End frequency.', ...
    'Units', 'normalized', 'Position', [0.600 0.690 0.370 0.200], ...
    'Callback', @autoChainChanged);

velTip = 'Initial vertical velocity at t=0 (m/s), upward positive.';
uicontrol('Parent', settingsPanel, 'Style', 'text', 'String', 'Initial vert vel (m/s)', ...
    'HorizontalAlignment', 'left', 'TooltipString', velTip, ...
    'Units', 'normalized', 'Position', [0.020 0.395 0.255 0.180]);
initialVelocityEdit = uicontrol('Parent', settingsPanel, 'Style', 'edit', 'String', '0', ...
    'BackgroundColor', 'white', 'TooltipString', velTip, ...
    'Units', 'normalized', 'Position', [0.270 0.420 0.105 0.180]);

hoverTip = 'Frequency where lift equals weight (Hz).';
uicontrol('Parent', settingsPanel, 'Style', 'text', 'String', 'Hover frequency (Hz)', ...
    'HorizontalAlignment', 'left', 'TooltipString', hoverTip, ...
    'Units', 'normalized', 'Position', [0.405 0.395 0.245 0.180]);
hoverFrequencyEdit = uicontrol('Parent', settingsPanel, 'Style', 'edit', 'String', '140', ...
    'BackgroundColor', 'white', 'TooltipString', hoverTip, ...
    'Units', 'normalized', 'Position', [0.650 0.420 0.105 0.180]);

totalTimeLabel = uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', 'Total time: 1.700 s', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', 'Units', 'normalized', ...
    'Position', [0.765 0.410 0.205 0.180]);

uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', 'Open-loop model: Lift/mg = (f_robot / f_hover)^2. Upward positive.', ...
    'FontSize', 8, 'ForegroundColor', [0.35 0.35 0.35], 'HorizontalAlignment', 'left', ...
    'Units', 'normalized', 'Position', [0.020 0.035 0.950 0.245]);

%% Buttons: Row Operations
buttonY1 = 0.675; buttonH = 0.042; buttonGap = 0.008;
buttonW = (0.950 - 5*buttonGap)/6; buttonX0 = 0.025;
buttonLabels = {'+ Hold', '+ Polynomial', '+ Exponential', 'Copy', 'Delete', 'Clear All'};
buttonCallbacks = {@(~,~)addSegment('Hold'), @(~,~)addSegment('Polynomial'), ...
    @(~,~)addSegment('Exponential'), @copySegment, @deleteSegment, @clearSegments};
for k = 1:6
    uicontrol('Parent', leftPanel, 'Style', 'pushbutton', 'String', buttonLabels{k}, ...
        'Units', 'normalized', ...
        'Position', [buttonX0+(k-1)*(buttonW+buttonGap), buttonY1, buttonW, buttonH], ...
        'Callback', buttonCallbacks{k});
end

%% Editable Table
signalTable = uitable( ...
    'Parent', leftPanel, ...
    'Data', exampleData, ...
    'ColumnName', {'Type', 'Start (Hz)', 'End (Hz)', 'Duration (s)', 'Order / Exp k'}, ...
    'ColumnFormat', {{'Hold', 'Polynomial', 'Exponential'}, 'numeric', 'numeric', 'numeric', 'numeric'}, ...
    'ColumnEditable', [true true true true true], ...
    'ColumnWidth', {100 80 80 85 90}, ...
    'RowName', [], ...
    'Units', 'normalized', ...
    'Position', [0.025 0.320 0.950 0.345], ...
    'CellEditCallback', @tableEdited, ...
    'CellSelectionCallback', @tableSelected);

%% Buttons: Execution & Order
buttonY2 = 0.268; buttonW2 = (0.950 - 3*buttonGap)/4;
moveLabels = {'Move Up', 'Move Down', 'Preview Command', 'Run Simulation'};
moveCallbacks = {@(~,~)moveSegment(-1), @(~,~)moveSegment(1), @previewCommand, @runSimulation};
for k = 1:4
    h = uicontrol('Parent', leftPanel, 'Style', 'pushbutton', 'String', moveLabels{k}, ...
        'Units', 'normalized', ...
        'Position', [buttonX0+(k-1)*(buttonW2+buttonGap), buttonY2, buttonW2, buttonH], ...
        'Callback', moveCallbacks{k});
    if k == 3, previewButton = h; elseif k == 4, runButton = h; set(runButton, 'FontWeight', 'bold'); end
end

%% Fixed Model Parameters Panel
infoPanel = uipanel('Parent', leftPanel, 'Title', 'Fixed Robot Model', ...
    'Units', 'normalized', 'Position', [0.025 0.120 0.950 0.135]);

uicontrol('Parent', infoPanel, 'Style', 'text', ...
    'String', sprintf('I_robot = %.5e kg m^2', I_robot), ...
    'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0.025 0.560 0.420 0.280]);
uicontrol('Parent', infoPanel, 'Style', 'text', ...
    'String', sprintf('k_drag = %.5e N m/Hz^2', k_drag), ...
    'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0.025 0.150 0.420 0.280]);
uicontrol('Parent', infoPanel, 'Style', 'text', ...
    'String', sprintf('Drag R^2 = %.5f', R_squared), ...
    'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0.465 0.560 0.290 0.280]);
uicontrol('Parent', infoPanel, 'Style', 'pushbutton', 'String', 'Reset Example', ...
    'Units', 'normalized', 'Position', [0.760 0.230 0.215 0.520], 'Callback', @resetExample);

statusText = uicontrol('Parent', leftPanel, 'Style', 'text', 'String', 'Ready', ...
    'HorizontalAlignment', 'left', 'FontWeight', 'bold', ...
    'BackgroundColor', [0.85 0.85 0.85], 'Units', 'normalized', 'Position', [0.025 0.045 0.950 0.050]);

%% Right Panel: Axes Setup
frequencyAxes = axes('Parent', rightPanel, 'Units', 'normalized', 'Position', [0.085 0.685 0.875 0.245], 'Box', 'on');
grid(frequencyAxes, 'on'); xlabel(frequencyAxes, 'Time (s)'); ylabel(frequencyAxes, 'Frequency (Hz)');
title(frequencyAxes, 'External-field Command Preview');

phaseAxes = axes('Parent', rightPanel, 'Units', 'normalized', 'Position', [0.085 0.465 0.875 0.155], 'Box', 'on');
grid(phaseAxes, 'on'); xlabel(phaseAxes, 'Time (s)'); ylabel(phaseAxes, '\delta (degree)');
title(phaseAxes, 'Wrapped Phase Difference'); set(phaseAxes, 'YLim', [-180 180]);

verticalAxes = axes('Parent', rightPanel, 'Units', 'normalized', 'Position', [0.085 0.245 0.875 0.155], 'Box', 'on');
grid(verticalAxes, 'on'); xlabel(verticalAxes, 'Time (s)'); ylabel(verticalAxes, 'Vertical Disp. (mm)');
title(verticalAxes, 'Vertical Displacement (mm)');

uicontrol('Parent', rightPanel, 'Style', 'text', 'String', 'Simulation Results', ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'Units', 'normalized', 'Position', [0.035 0.195 0.300 0.030]);

resultBox = uicontrol('Parent', rightPanel, 'Style', 'edit', 'Max', 20, 'Min', 0, ...
    'Enable', 'inactive', 'BackgroundColor', 'white', 'HorizontalAlignment', 'left', ...
    'FontName', 'Consolas', 'FontSize', 9, ...
    'String', {'Edit table, then click Run Simulation.'}, ...
    'Units', 'normalized', 'Position', [0.035 0.025 0.925 0.165]);

previewCommand();

%% ---- GUI Callbacks -----------------------------------------------------
    function addSegment(typeName)
        data = get(signalTable, 'Data');
        startFrequency = 160;
        if ~isempty(data)
            val = numericValue(data{end,3});
            if isfinite(val), startFrequency = val; end
        end
        switch typeName
            case 'Hold', newRow = {'Hold', startFrequency, startFrequency, 1.0, 0};
            case 'Polynomial', newRow = {'Polynomial', startFrequency, startFrequency, 0.2, 1};
            otherwise, newRow = {'Exponential', startFrequency, startFrequency, 0.5, 4};
        end
        data(end+1,:) = newRow;
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = size(data,1);
        afterTableChange('Segment added');
    end

    function copySegment(~,~)
        data = get(signalTable, 'Data');
        if isempty(data), addSegment('Hold'); return; end
        row = getSelectedRow(data);
        data = [data(1:row,:); data(row,:); data(row+1:end,:)];
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = row + 1;
        afterTableChange('Segment copied');
    end

    function deleteSegment(~,~)
        data = get(signalTable, 'Data');
        if isempty(data), return; end
        row = getSelectedRow(data);
        data(row,:) = [];
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = min(row, max(1,size(data,1)));
        afterTableChange('Segment deleted');
    end

    function clearSegments(~,~)
        set(signalTable, 'Data', cell(0,5));
        selectedRow = 1;
        set(totalTimeLabel, 'String', 'Total time: 0 s');
        cla(frequencyAxes); cla(phaseAxes); cla(verticalAxes);
        set(resultBox, 'String', {'Add at least one segment before simulation.'});
        setStatus('Table cleared', [0.85 0.85 0.85]);
    end

    function moveSegment(direction)
        data = get(signalTable, 'Data');
        if size(data,1) < 2, return; end
        row = getSelectedRow(data);
        dest = row + direction;
        if dest < 1 || dest > size(data,1), return; end
        tmp = data(row,:); data(row,:) = data(dest,:); data(dest,:) = tmp;
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = dest;
        afterTableChange('Segment moved');
    end

    function tableSelected(~,event)
        if isfield(event, 'Indices') && ~isempty(event.Indices)
            selectedRow = event.Indices(1,1);
        end
    end

    function tableEdited(~,event)
        data = get(signalTable, 'Data');
        row = event.Indices(1); col = event.Indices(2);
        selectedRow = row;
        try
            if col >= 2
                val = numericValue(data{row,col});
                if ~isfinite(val), error('Edited value must be finite.'); end
                if (col == 2 || col == 3) && val < 0, error('Frequency cannot be negative.'); end
                if col == 4 && val <= 0, error('Segment duration must be positive.'); end
                data{row,col} = val;
            end
            data = normalizeSegments(data);
            validateSegments(data);
            set(signalTable, 'Data', data);
            afterTableChange('Table updated');
        catch ME
            data{row,col} = event.PreviousData;
            set(signalTable, 'Data', normalizeSegments(data));
            errordlg(ME.message, 'Invalid Entry', 'modal');
        end
    end

    function autoChainChanged(~,~)
        data = normalizeSegments(get(signalTable, 'Data'));
        set(signalTable, 'Data', data);
        afterTableChange('Auto-chain changed');
    end

    function resetExample(~,~)
        set(signalTable, 'Data', exampleData);
        selectedRow = 1;
        set(marginEdit, 'String', '5'); set(toleranceEdit, 'String', '1');
        set(initialVelocityEdit, 'String', '0'); set(hoverFrequencyEdit, 'String', '140');
        set(autoChainCheck, 'Value', 1);
        afterTableChange('Example restored');
    end

    function afterTableChange(msg)
        updateTotalTime(); previewCommand(); setStatus(msg, [0.80 0.90 1.00]);
    end

    function previewCommand(varargin)
        data = get(signalTable, 'Data');
        if isempty(data), cla(frequencyAxes); return; end
        try
            data = normalizeSegments(data); validateSegments(data);
            set(signalTable, 'Data', data);
            [types, starts, ends, durations, shapes, edges] = parseSegments(data);
            [tPrev, cmdPrev] = sampleCommand(types, starts, ends, durations, shapes, edges, 5000);

            cla(frequencyAxes);
            plot(frequencyAxes, tPrev, cmdPrev, '--', 'LineWidth', 1.7);
            grid(frequencyAxes, 'on'); box(frequencyAxes, 'on');
            xlabel(frequencyAxes, 'Time (s)'); ylabel(frequencyAxes, 'Frequency (Hz)');
            title(frequencyAxes, 'External-field Command Preview');
            legend(frequencyAxes, {'External Command'}, 'Location', 'best');

            cla(phaseAxes); grid(phaseAxes, 'on'); title(phaseAxes, 'Run Simulation for phase response');
            set(phaseAxes, 'YLim', [-180 180]);
            cla(verticalAxes); grid(verticalAxes, 'on'); title(verticalAxes, 'Run Simulation for vertical motion');
            updateTotalTime();
        catch ME
            set(resultBox, 'String', {['Preview error: ' ME.message]});
        end
    end

    function runSimulation(varargin)
        data = get(signalTable, 'Data');
        if isempty(data), showError('Add at least one signal segment.'); return; end

        torqueMargin = str2double(get(marginEdit, 'String'));
        frequencyTolerance = str2double(get(toleranceEdit, 'String'));
        initialVertVel = str2double(get(initialVelocityEdit, 'String'));
        hoverFreq = str2double(get(hoverFrequencyEdit, 'String'));

        if ~isfinite(torqueMargin) || torqueMargin < 1, showError('Torque margin must be >= 1.'); return; end
        if ~isfinite(frequencyTolerance) || frequencyTolerance <= 0, showError('Tolerance must be positive.'); return; end
        if ~isfinite(initialVertVel), showError('Initial vertical velocity must be finite.'); return; end
        if ~isfinite(hoverFreq) || hoverFreq <= 0, showError('Hover frequency must be positive.'); return; end

        try
            data = normalizeSegments(data); validateSegments(data);
            set(signalTable, 'Data', data);
            [types, starts, ends, durations, shapes, edges] = parseSegments(data);
        catch ME
            showError(ME.message); return;
        end

        set(runButton, 'Enable', 'off'); set(previewButton, 'Enable', 'off');
        setStatus('Solving open-loop dynamics...', [1.00 0.90 0.65]); drawnow;

        try
            fInitial = starts(1);
            if fInitial <= 0, error('First Start frequency must be positive.'); end

            tauReqInitial = k_drag * fInitial^2;
            tauMagMax = torqueMargin * tauReqInitial;
            deltaInitial = asin(min(max(1/torqueMargin, -1), 1));
            stateAtStart = [deltaInitial; 2*pi*fInitial];

            totalTime = edges(end);
            outputStep = max(1E-4, totalTime/15000);
            solverMaxStep = max(min(2E-4, min(durations)/20), 1E-7);
            options = odeset('RelTol',1E-8, 'AbsTol',[1E-9 1E-7], 'MaxStep',solverMaxStep);

            allTime = []; allState = []; allCommand = [];

            for segIdx = 1:numel(durations)
                t0 = edges(segIdx); t1 = edges(segIdx+1);
                cmdFn = @(t)evaluateSegmentFrequency(t, t0, durations(segIdx), types{segIdx}, starts(segIdx), ends(segIdx), shapes(segIdx));
                rotationODE = @(t,y)[2*pi*cmdFn(t) - y(2); (tauMagMax*sin(y(1)) - k_drag*(y(2)/(2*pi))*abs(y(2)/(2*pi))) / I_robot];

                ptCount = max(2, ceil((t1-t0)/outputStep)+1);
                evalTime = linspace(t0, t1, ptCount)';
                [tSeg, ySeg] = ode45(rotationODE, evalTime, stateAtStart, options);
                cmdSeg = arrayfun(cmdFn, tSeg);

                if segIdx > 1
                    tSeg(1) = []; ySeg(1,:) = []; cmdSeg(1) = [];
                end
                allTime = [allTime; tSeg]; allState = [allState; ySeg]; allCommand = [allCommand; cmdSeg];
                stateAtStart = ySeg(end,:)';
            end

            delta = allState(:,1); fRobot = allState(:,2)/(2*pi);
            deltaWrapped = atan2(sin(delta), cos(delta));
            freqErr = fRobot - allCommand;
            tauMag = tauMagMax * sin(delta);
            tauDrag = -k_drag * fRobot .* abs(fRobot);
            angAccel = (tauMag + tauDrag) / I_robot;

            % Open-Loop Vertical Motion Model
            g = 9.80665;
            liftRatio = (fRobot ./ hoverFreq).^2;
            vertAccel = g * (liftRatio - 1);
            vertVel = initialVertVel + cumtrapz(allTime, vertAccel);
            vertDisp = cumtrapz(allTime, vertVel);

            rmsErr = sqrt(mean(freqErr.^2)); maxAbsErr = max(abs(freqErr));
            netPhaseTurns = (delta(end)-delta(1))/(2*pi);
            holdSummary = buildHoldSummary(types, edges, allTime, freqErr, frequencyTolerance);

            cla(frequencyAxes);
            plot(frequencyAxes, allTime, allCommand, '--', 'LineWidth', 1.6); hold(frequencyAxes, 'on');
            plot(frequencyAxes, allTime, fRobot, '-', 'LineWidth', 1.6);
            drawBoundaries(frequencyAxes, edges, [0.55 0.55 0.55]); hold(frequencyAxes, 'off');
            grid(frequencyAxes, 'on'); box(frequencyAxes, 'on');
            xlabel(frequencyAxes, 'Time (s)'); ylabel(frequencyAxes, 'Frequency (Hz)');
            title(frequencyAxes, sprintf('Open-Loop Response (%d segments)', numel(durations)));
            legend(frequencyAxes, {'External Command', 'Robot Frequency'}, 'Location', 'best');

            cla(phaseAxes);
            plot(phaseAxes, allTime, deltaWrapped*180/pi, 'LineWidth', 1.3); hold(phaseAxes, 'on');
            drawBoundaries(phaseAxes, edges, [0.55 0.55 0.55]); hold(phaseAxes, 'off');
            grid(phaseAxes, 'on'); box(phaseAxes, 'on'); set(phaseAxes, 'YLim', [-180 180]);
            xlabel(phaseAxes, 'Time (s)'); ylabel(phaseAxes, '\delta (deg)');

            cla(verticalAxes);
            plot(verticalAxes, allTime, 1000*vertDisp, 'LineWidth', 1.5); hold(verticalAxes, 'on');
            drawBoundaries(verticalAxes, edges, [0.55 0.55 0.55]); hold(verticalAxes, 'off');
            grid(verticalAxes, 'on'); box(verticalAxes, 'on');
            xlabel(verticalAxes, 'Time (s)'); ylabel(verticalAxes, 'Vertical Disp (mm)');

            resLines = {
                sprintf('Segments                  : %d', numel(durations));
                sprintf('Total time                : %.6f s', totalTime);
                sprintf('Torque margin             : %.6f', torqueMargin);
                sprintf('Hover frequency           : %.6f Hz', hoverFreq);
                sprintf('Final vertical displacement: %.6f mm', 1000*vertDisp(end));
                sprintf('Max vertical displacement : %.6f mm', 1000*max(vertDisp));
                sprintf('RMS tracking error        : %.6f Hz', rmsErr);
                sprintf('Max tracking error        : %.6f Hz', maxAbsErr);
                sprintf('Net relative phase turns  : %.4f', netPhaseTurns)
                };
            set(resultBox, 'String', [resLines; {''}; holdSummary]);
            setStatus('Simulation complete', [0.75 0.95 0.78]);
        catch ME
            showError(ME.message);
        end
        set(runButton, 'Enable', 'on'); set(previewButton, 'Enable', 'on'); drawnow;
    end

%% ---- Helper Functions --------------------------------------------------
    function data = normalizeSegments(data)
        if isempty(data), return; end
        validTypes = {'Hold','Polynomial','Exponential'};
        for row = 1:size(data,1)
            typeName = data{row,1};
            if strcmpi(typeName,'Linear')
                typeName = 'Polynomial';
                if ~isfinite(numericValue(data{row,5})) || numericValue(data{row,5}) == 0
                    data{row,5} = 1;
                end
            else
                m = find(strcmpi(typeName,validTypes),1);
                if isempty(m), typeName = 'Hold'; else, typeName = validTypes{m}; end
            end
            data{row,1} = typeName;
            for col = 2:5, data{row,col} = numericValue(data{row,col}); end
            if get(autoChainCheck,'Value') && row > 1, data{row,2} = data{row-1,3}; end
            if strcmp(typeName,'Hold'), data{row,3} = data{row,2}; data{row,5} = 0;
            elseif strcmp(typeName,'Polynomial') && isfinite(data{row,5})
                data{row,5} = max(1,round(data{row,5}));
            end
        end
    end

    function validateSegments(data)
        if isempty(data), error('At least one signal segment is required.'); end
        for row = 1:size(data,1)
            vals = [numericValue(data{row,2}), numericValue(data{row,3}), numericValue(data{row,4}), numericValue(data{row,5})];
            if any(~isfinite(vals)), error('Row %d contains non-finite numbers.', row); end
            if vals(1) < 0 || vals(2) < 0, error('Row %d contains negative frequency.', row); end
            if vals(3) <= 0, error('Row %d duration must be positive.', row); end
        end
    end

    function [types,starts,ends,durations,shapes,edges] = parseSegments(data)
        n = size(data,1); types = cell(n,1); starts = zeros(n,1);
        ends = zeros(n,1); durations = zeros(n,1); shapes = zeros(n,1);
        for row = 1:n
            types{row} = data{row,1}; starts(row) = numericValue(data{row,2});
            ends(row) = numericValue(data{row,3}); durations(row) = numericValue(data{row,4});
            shapes(row) = numericValue(data{row,5});
        end
        edges = [0; cumsum(durations)];
    end

    function freq = evaluateSegmentFrequency(time, segStart, duration, typeName, startF, endF, shape)
        s = min(max((time - segStart)/duration, 0), 1);
        switch lower(typeName)
            case 'hold', blend = zeros(size(s));
            case {'polynomial','linear'}, order = max(1,round(shape)); blend = s.^order;
            case 'exponential'
                if abs(shape) < 1E-9, blend = s; else, blend = (1-exp(-shape.*s))./(1-exp(-shape)); end
            otherwise, error('Unsupported segment type: %s', typeName);
        end
        freq = startF + (endF - startF) .* blend;
    end

    function [tVec, cmdVec] = sampleCommand(types, starts, ends, durations, shapes, edges, maxPts)
        totalT = edges(end);
        sampleCount = max(300, min(maxPts, ceil(totalT/2E-4)+1));
        tVec = linspace(0, totalT, sampleCount)';
        cmdVec = zeros(size(tVec));
        for i = 1:numel(durations)
            if i < numel(durations), mask = tVec >= edges(i) & tVec < edges(i+1);
            else, mask = tVec >= edges(i) & tVec <= edges(i+1); end
            cmdVec(mask) = evaluateSegmentFrequency(tVec(mask), edges(i), durations(i), types{i}, starts(i), ends(i), shapes(i));
        end
    end

    function lines = buildHoldSummary(types, edges, time, errSig, tol)
        lines = {'Hold tracking (last 20% tail):'}; holdCount = 0;
        for i = 1:numel(types)
            if strcmpi(types{i}, 'Hold')
                holdCount = holdCount + 1;
                tStart = edges(i) + 0.8*(edges(i+1)-edges(i));
                mask = time >= tStart & time <= edges(i+1);
                if any(mask)
                    maxErr = max(abs(errSig(mask)));
                    if maxErr <= tol, st = 'within tol'; else, st = 'outside tol'; end
                    lines{end+1,1} = sprintf('  Hold %d (seg %d): tail err %.5f Hz, %s', holdCount, i, maxErr, st);
                end
            end
        end
        if holdCount == 0, lines{end+1,1} = '  No Hold segments.'; end
    end

    function drawBoundaries(ax, edges, lineCol)
        yLim = get(ax, 'YLim');
        for i = 2:numel(edges)-1
            line(ax, [edges(i) edges(i)], yLim, 'LineStyle', ':', 'Color', lineCol, 'HandleVisibility', 'off');
        end
        set(ax, 'YLim', yLim);
    end

    function row = getSelectedRow(data)
        row = selectedRow;
        if isempty(row) || ~isfinite(row) || row < 1 || row > size(data,1), row = size(data,1); end
        row = round(row);
    end

    function updateTotalTime()
        data = get(signalTable, 'Data');
        if isempty(data), set(totalTimeLabel, 'String', 'Total time: 0 s'); return; end
        durations = zeros(size(data,1),1);
        for row = 1:size(data,1), durations(row) = numericValue(data{row,4}); end
        if all(isfinite(durations))
            set(totalTimeLabel, 'String', sprintf('Total time: %.4f s', sum(durations)));
        else
            set(totalTimeLabel, 'String', 'Total time: invalid');
        end
    end

    function val = numericValue(inVal)
        if isnumeric(inVal) && isscalar(inVal), val = double(inVal);
        elseif ischar(inVal), val = str2double(inVal);
        else, try val = str2double(char(inVal)); catch, val = NaN; end
        end
    end

    function setStatus(msg, bg)
        set(statusText, 'String', msg, 'BackgroundColor', bg);
    end

    function showError(msg)
        setStatus('Input/Model Error', [1.00 0.72 0.72]);
        set(resultBox, 'String', {['Error: ' msg]});
        set(runButton, 'Enable', 'on'); set(previewButton, 'Enable', 'on');
        errordlg(msg, 'Simulation Error', 'modal');
    end
end