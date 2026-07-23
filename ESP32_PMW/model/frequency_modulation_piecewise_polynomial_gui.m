
function frequency_modulation_piecewise_polynomial_gui
% FREQUENCY_MODULATION_PIECEWISE_POLYNOMIAL_GUI
% Robust piecewise frequency-command editor using classic MATLAB graphics.
% It avoids uifigure/uigridlayout so the layout remains visible across
% MATLAB releases, Windows display scaling settings, and screen sizes.
%
% Each table row defines one external-field frequency segment:
%   Hold        : constant frequency
%   Polynomial  : normalized power ramp; order 1 = linear, 2 = quadratic,
%                 3 = cubic, etc.
%   Exponential : normalized exponential ramp that reaches End exactly
%
% Dynamics:
%   delta_dot = 2*pi*f_field(t) - omega_robot
%   I*omega_dot = tau_max*sin(delta) - k_drag*f_robot*abs(f_robot)
%
% Margin definition:
%   margin = tau_max / |tau_drag at the first Start frequency|

%% Fixed robot model
% Moment of inertia about the spin axis: central body plus n_blades
% identical blades. Each blade is a cylinder about a transverse axis
% through its own center, shifted to the spin axis by the parallel-axis
% theorem:
%   I_blade = 1/12*m_blade*(3*r_blade^2 + h_blade^2) + m_blade*d_blade^2
% Lengths are entered in mm; the 1E-6 factor converts mm^2 -> m^2.
% Derivation and the two-rod approximation (exact only when blades root
% at the axis, d_blade = h_blade/2): see model/lecture_notes.md, Sec. 3.
I_body   = 3.89E-9;      % central body/magnet inertia (kg m^2)
n_blades = 4;            % blade count (two opposite pairs)
m_blade  = 1.17832E-5;   % single-blade mass (kg)
r_blade  = 0.79375;      % blade cylinder radius (mm)
h_blade  = 0.79375;      % blade cylinder length (mm)
d_blade  = 0.496875;     % blade-center offset from spin axis (mm)

I_blade = (1.0/12.0 * m_blade * (3.0*r_blade^2 + h_blade^2) ...
    + m_blade * d_blade^2) * 1E-6;
I_robot = I_body + n_blades * I_blade;

fre_points = (10:10:230)';
drag_torque_points = [
    -7.20311E-08;
    -8.54759E-07;
    -1.18389E-06;
    -1.55412E-06;
    -2.06800E-06;
    -2.56102E-06;
    -2.88399E-06;
    -3.59653E-06;
    -3.83331E-06;
    -4.99189E-06;
    -5.72497E-06;
    -6.29224E-06;
    -7.17918E-06;
    -8.19104E-06;
    -9.07986E-06;
    -1.00964E-05;
    -1.08262E-05;
    -1.21433E-05;
    -1.34310E-05;
    -1.48856E-05;
    -1.68245E-05;
    -1.88440E-05;
    -2.10429E-05
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

%% Main window: all positions use normalized coordinates
fig = figure( ...
    'Name', 'Flying Robot Piecewise Frequency Modulation - Classic GUI', ...
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
    'Title', 'Command and Robot Response', ...
    'FontWeight', 'bold', ...
    'Units', 'normalized', ...
    'Position', [0.425 0.020 0.565 0.965]);

%% Left panel: header and settings
uicontrol( ...
    'Parent', leftPanel, ...
    'Style', 'text', ...
    'String', 'Build the external-field frequency one segment at a time', ...
    'FontSize', 14, ...
    'FontWeight', 'bold', ...
    'BackgroundColor', get(leftPanel, 'BackgroundColor'), ...
    'Units', 'normalized', ...
    'HorizontalAlignment', 'center', ...
    'Position', [0.025 0.925 0.950 0.050]);

settingsPanel = uipanel( ...
    'Parent', leftPanel, ...
    'Title', 'Global model settings', ...
    'Units', 'normalized', ...
    'Position', [0.025 0.805 0.950 0.115]);

uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', 'Torque margin', 'HorizontalAlignment', 'left', ...
    'Units', 'normalized', 'Position', [0.020 0.555 0.170 0.300]);
marginEdit = uicontrol('Parent', settingsPanel, 'Style', 'edit', ...
    'String', '5', 'BackgroundColor', 'white', ...
    'Units', 'normalized', 'Position', [0.190 0.590 0.105 0.300]);

uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', 'Tolerance (Hz)', 'HorizontalAlignment', 'left', ...
    'Units', 'normalized', 'Position', [0.325 0.555 0.170 0.300]);
toleranceEdit = uicontrol('Parent', settingsPanel, 'Style', 'edit', ...
    'String', '1', 'BackgroundColor', 'white', ...
    'Units', 'normalized', 'Position', [0.495 0.590 0.105 0.300]);

autoChainCheck = uicontrol('Parent', settingsPanel, 'Style', 'checkbox', ...
    'String', 'Auto-chain adjacent segments', ...
    'Value', 1, ...
    'Units', 'normalized', ...
    'Position', [0.625 0.570 0.350 0.330], ...
    'Callback', @autoChainChanged);

uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', ['Margin = maximum magnetic torque / drag at the first Start frequency.  ' ...
        'Polynomial: order 1 = linear, 2 = quadratic, 3 = cubic.'], ...
    'FontSize', 8.5, ...
    'ForegroundColor', [0.35 0.35 0.35], ...
    'HorizontalAlignment', 'left', ...
    'Units', 'normalized', ...
    'Position', [0.020 0.030 0.720 0.390]);

totalTimeLabel = uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', 'Total time: 1.700 s', ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', ...
    'Units', 'normalized', ...
    'Position', [0.745 0.070 0.230 0.300]);

%% Add buttons
buttonY1 = 0.755;
buttonH = 0.042;
buttonGap = 0.008;
buttonW = (0.950 - 5*buttonGap)/6;
buttonX0 = 0.025;
buttonLabels = {'+ Hold', '+ Polynomial', '+ Exponential', 'Copy', 'Delete', 'Clear All'};
buttonCallbacks = {@(~,~)addSegment('Hold'), @(~,~)addSegment('Polynomial'), ...
    @(~,~)addSegment('Exponential'), @copySegment, @deleteSegment, @clearSegments};
for k = 1:6
    uicontrol('Parent', leftPanel, 'Style', 'pushbutton', ...
        'String', buttonLabels{k}, ...
        'Units', 'normalized', ...
        'Position', [buttonX0+(k-1)*(buttonW+buttonGap), buttonY1, buttonW, buttonH], ...
        'Callback', buttonCallbacks{k});
end

%% Editable segment table
signalTable = uitable( ...
    'Parent', leftPanel, ...
    'Data', exampleData, ...
    'ColumnName', {'Type', 'Start (Hz)', 'End (Hz)', 'Duration (s)', 'Order / exp k'}, ...
    'ColumnFormat', {{'Hold', 'Polynomial', 'Exponential'}, ...
                     'numeric', 'numeric', 'numeric', 'numeric'}, ...
    'ColumnEditable', [true true true true true], ...
    'ColumnWidth', {105 82 82 88 94}, ...
    'RowName', [], ...
    'Units', 'normalized', ...
    'Position', [0.025 0.320 0.950 0.425], ...
    'CellEditCallback', @tableEdited, ...
    'CellSelectionCallback', @tableSelected);

%% Move and run buttons
buttonY2 = 0.268;
buttonW2 = (0.950 - 3*buttonGap)/4;
moveLabels = {'Move Up', 'Move Down', 'Preview Command', 'Run Simulation'};
moveCallbacks = {@(~,~)moveSegment(-1), @(~,~)moveSegment(1), @previewCommand, @runSimulation};
for k = 1:4
    h = uicontrol('Parent', leftPanel, 'Style', 'pushbutton', ...
        'String', moveLabels{k}, ...
        'Units', 'normalized', ...
        'Position', [buttonX0+(k-1)*(buttonW2+buttonGap), buttonY2, buttonW2, buttonH], ...
        'Callback', moveCallbacks{k});
    if k == 3
        previewButton = h;
    elseif k == 4
        runButton = h;
        set(runButton, 'FontWeight', 'bold');
    end
end

%% Fixed model information
infoPanel = uipanel( ...
    'Parent', leftPanel, ...
    'Title', 'Fixed model', ...
    'Units', 'normalized', ...
    'Position', [0.025 0.120 0.950 0.135]);

uicontrol('Parent', infoPanel, 'Style', 'text', ...
    'String', sprintf('I_robot = %.5e kg m^2', I_robot), ...
    'HorizontalAlignment', 'left', ...
    'Units', 'normalized', 'Position', [0.025 0.560 0.420 0.280]);
uicontrol('Parent', infoPanel, 'Style', 'text', ...
    'String', sprintf('k_drag = %.5e N m/Hz^2', k_drag), ...
    'HorizontalAlignment', 'left', ...
    'Units', 'normalized', 'Position', [0.025 0.150 0.420 0.280]);
uicontrol('Parent', infoPanel, 'Style', 'text', ...
    'String', sprintf('Drag-fit R^2 = %.5f', R_squared), ...
    'HorizontalAlignment', 'left', ...
    'Units', 'normalized', 'Position', [0.465 0.560 0.290 0.280]);
uicontrol('Parent', infoPanel, 'Style', 'pushbutton', ...
    'String', 'Reset Example', ...
    'Units', 'normalized', ...
    'Position', [0.760 0.230 0.215 0.520], ...
    'Callback', @resetExample);

statusText = uicontrol( ...
    'Parent', leftPanel, ...
    'Style', 'text', ...
    'String', 'Ready', ...
    'HorizontalAlignment', 'left', ...
    'FontWeight', 'bold', ...
    'BackgroundColor', [0.85 0.85 0.85], ...
    'Units', 'normalized', ...
    'Position', [0.025 0.045 0.950 0.050]);

%% Right panel: frequency and phase axes plus result box
frequencyAxes = axes( ...
    'Parent', rightPanel, ...
    'Units', 'normalized', ...
    'Position', [0.085 0.585 0.875 0.350], ...
    'Box', 'on');
grid(frequencyAxes, 'on');
xlabel(frequencyAxes, 'Time (s)');
ylabel(frequencyAxes, 'Frequency (Hz)');
title(frequencyAxes, 'External-field command preview');

phaseAxes = axes( ...
    'Parent', rightPanel, ...
    'Units', 'normalized', ...
    'Position', [0.085 0.310 0.875 0.215], ...
    'Box', 'on');
grid(phaseAxes, 'on');
xlabel(phaseAxes, 'Time (s)');
ylabel(phaseAxes, '\delta (degree)');
title(phaseAxes, 'Wrapped phase difference');
set(phaseAxes, 'YLim', [-180 180]);

uicontrol('Parent', rightPanel, 'Style', 'text', ...
    'String', 'Simulation results', ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', ...
    'Units', 'normalized', ...
    'Position', [0.035 0.255 0.300 0.035]);

resultBox = uicontrol( ...
    'Parent', rightPanel, ...
    'Style', 'edit', ...
    'Max', 20, ...
    'Min', 0, ...
    'Enable', 'inactive', ...
    'BackgroundColor', 'white', ...
    'HorizontalAlignment', 'left', ...
    'FontName', 'Consolas', ...
    'FontSize', 10, ...
    'String', {'Edit the segment table, then click Run Simulation.'}, ...
    'Units', 'normalized', ...
    'Position', [0.035 0.035 0.925 0.220]);

previewCommand();

%% ---- GUI callbacks -----------------------------------------------------
    function addSegment(typeName)
        data = get(signalTable, 'Data');
        if isempty(data)
            startFrequency = 160;
        else
            startFrequency = numericValue(data{end,3});
            if ~isfinite(startFrequency)
                startFrequency = 160;
            end
        end

        switch typeName
            case 'Hold'
                newRow = {'Hold', startFrequency, startFrequency, 1.0, 0};
            case 'Polynomial'
                newRow = {'Polynomial', startFrequency, startFrequency, 0.2, 1};
            otherwise
                newRow = {'Exponential', startFrequency, startFrequency, 0.5, 4};
        end

        data(end+1,:) = newRow;
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = size(data,1);
        afterTableChange('Segment added');
    end

    function copySegment(~,~)
        data = get(signalTable, 'Data');
        if isempty(data)
            addSegment('Hold');
            return;
        end
        row = getSelectedRow(data);
        data = [data(1:row,:); data(row,:); data(row+1:end,:)]; %#ok<AGROW>
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = row + 1;
        afterTableChange('Segment copied');
    end

    function deleteSegment(~,~)
        data = get(signalTable, 'Data');
        if isempty(data)
            return;
        end
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
        cla(frequencyAxes);
        cla(phaseAxes);
        set(resultBox, 'String', {'Add at least one segment before simulation.'});
        setStatus('Table cleared', [0.85 0.85 0.85]);
    end

    function moveSegment(direction)
        data = get(signalTable, 'Data');
        if size(data,1) < 2
            return;
        end
        row = getSelectedRow(data);
        destination = row + direction;
        if destination < 1 || destination > size(data,1)
            return;
        end
        temporary = data(row,:);
        data(row,:) = data(destination,:);
        data(destination,:) = temporary;
        data = normalizeSegments(data);
        set(signalTable, 'Data', data);
        selectedRow = destination;
        afterTableChange('Segment moved');
    end

    function tableSelected(~,event)
        try
            indices = event.Indices;
        catch
            indices = [];
        end
        if ~isempty(indices)
            selectedRow = indices(1,1);
        end
    end

    function tableEdited(~,event)
        data = get(signalTable, 'Data');
        row = event.Indices(1);
        column = event.Indices(2);
        selectedRow = row;

        try
            if column >= 2
                value = numericValue(data{row,column});
                if ~isfinite(value)
                    error('The edited value must be finite.');
                end
                if (column == 2 || column == 3) && value < 0
                    error('Frequency cannot be negative.');
                end
                if column == 4 && value <= 0
                    error('Segment duration must be greater than zero.');
                end
                data{row,column} = value;
            end

            data = normalizeSegments(data);
            validateSegments(data);
            set(signalTable, 'Data', data);
            afterTableChange('Table updated');
        catch ME
            data{row,column} = event.PreviousData;
            set(signalTable, 'Data', normalizeSegments(data));
            errordlg(ME.message, 'Invalid Table Entry', 'modal');
        end
    end

    function autoChainChanged(~,~)
        data = normalizeSegments(get(signalTable, 'Data'));
        set(signalTable, 'Data', data);
        afterTableChange('Auto-chain setting changed');
    end

    function resetExample(~,~)
        set(signalTable, 'Data', exampleData);
        selectedRow = 1;
        set(marginEdit, 'String', '5');
        set(toleranceEdit, 'String', '1');
        set(autoChainCheck, 'Value', 1);
        afterTableChange('Example restored');
    end

    function afterTableChange(message)
        updateTotalTime();
        previewCommand();
        setStatus(message, [0.80 0.90 1.00]);
    end

    function previewCommand(varargin)
        data = get(signalTable, 'Data');
        if isempty(data)
            cla(frequencyAxes);
            return;
        end

        try
            data = normalizeSegments(data);
            validateSegments(data);
            set(signalTable, 'Data', data);
            [types, starts, ends, durations, shapes, edges] = parseSegments(data);
            [timePreview, commandPreview] = sampleCommand( ...
                types, starts, ends, durations, shapes, edges, 5000);

            cla(frequencyAxes);
            plot(frequencyAxes, timePreview, commandPreview, '--', 'LineWidth', 1.7);
            grid(frequencyAxes, 'on');
            box(frequencyAxes, 'on');
            xlabel(frequencyAxes, 'Time (s)');
            ylabel(frequencyAxes, 'Frequency (Hz)');
            title(frequencyAxes, 'External-field command preview');
            legend(frequencyAxes, {'External-field command'}, 'Location', 'best');

            cla(phaseAxes);
            grid(phaseAxes, 'on');
            box(phaseAxes, 'on');
            xlabel(phaseAxes, 'Time (s)');
            ylabel(phaseAxes, '\delta (degree)');
            title(phaseAxes, 'Run Simulation to calculate robot phase response');
            set(phaseAxes, 'YLim', [-180 180]);
            updateTotalTime();
        catch ME
            set(resultBox, 'String', {['Preview error: ' ME.message]});
        end
    end

    function runSimulation(varargin)
        data = get(signalTable, 'Data');
        if isempty(data)
            showError('Add at least one signal segment.');
            return;
        end

        torqueMargin = str2double(get(marginEdit, 'String'));
        frequencyTolerance = str2double(get(toleranceEdit, 'String'));

        if ~isfinite(torqueMargin) || torqueMargin < 1
            showError('Torque margin must be finite and greater than or equal to 1.');
            return;
        end
        if ~isfinite(frequencyTolerance) || frequencyTolerance <= 0
            showError('Tracking tolerance must be a positive finite number.');
            return;
        end

        try
            data = normalizeSegments(data);
            validateSegments(data);
            set(signalTable, 'Data', data);
            [types, starts, ends, durations, shapes, edges] = parseSegments(data);
        catch ME
            showError(ME.message);
            return;
        end

        set(runButton, 'Enable', 'off');
        set(previewButton, 'Enable', 'off');
        setStatus('Solving piecewise dynamics...', [1.00 0.90 0.65]);
        drawnow;

        try
            fInitial = starts(1);
            if fInitial <= 0
                error('The first segment Start frequency must be greater than zero.');
            end

            tauRequiredInitial = k_drag * fInitial^2;
            tauMagMax = torqueMargin * tauRequiredInitial;
            deltaInitial = asin(min(max(1/torqueMargin, -1), 1));
            stateAtStart = [deltaInitial; 2*pi*fInitial];

            totalTime = edges(end);
            outputStep = max(1E-4, totalTime/15000);
            solverMaxStep = min(2E-4, min(durations)/20);
            solverMaxStep = max(solverMaxStep, 1E-7);
            options = odeset('RelTol',1E-8, 'AbsTol',[1E-9 1E-7], ...
                'MaxStep',solverMaxStep);

            allTime = [];
            allState = [];
            allCommand = [];

            for segmentIndex = 1:numel(durations)
                t0 = edges(segmentIndex);
                t1 = edges(segmentIndex+1);

                commandFunction = @(t)evaluateSegmentFrequency( ...
                    t, t0, durations(segmentIndex), types{segmentIndex}, ...
                    starts(segmentIndex), ends(segmentIndex), shapes(segmentIndex));

                rotationODE = @(t,y)[
                    2*pi*commandFunction(t) - y(2);
                    (tauMagMax*sin(y(1)) ...
                    - k_drag*(y(2)/(2*pi))*abs(y(2)/(2*pi))) / I_robot
                    ];

                pointCount = max(2, ceil((t1-t0)/outputStep)+1);
                evaluationTime = linspace(t0,t1,pointCount)';
                [timeSegment, stateSegment] = ode45( ...
                    rotationODE, evaluationTime, stateAtStart, options);
                commandSegment = arrayfun(commandFunction, timeSegment);

                if segmentIndex > 1
                    timeSegment(1) = [];
                    stateSegment(1,:) = [];
                    commandSegment(1) = [];
                end

                allTime = [allTime; timeSegment]; %#ok<AGROW>
                allState = [allState; stateSegment]; %#ok<AGROW>
                allCommand = [allCommand; commandSegment]; %#ok<AGROW>
                stateAtStart = stateSegment(end,:)';
            end

            delta = allState(:,1);
            fRobot = allState(:,2)/(2*pi);
            deltaWrapped = atan2(sin(delta),cos(delta));
            frequencyError = fRobot - allCommand;
            tauMagnetic = tauMagMax*sin(delta);
            tauDrag = -k_drag*fRobot.*abs(fRobot);
            angularAcceleration = (tauMagnetic + tauDrag)/I_robot;

            rmsError = sqrt(mean(frequencyError.^2));
            maximumAbsoluteError = max(abs(frequencyError));
            netPhaseTurns = (delta(end)-delta(1))/(2*pi);
            holdSummary = buildHoldSummary(types,edges,allTime, ...
                frequencyError,frequencyTolerance);

            cla(frequencyAxes);
            plot(frequencyAxes, allTime, allCommand, '--', 'LineWidth',1.6);
            hold(frequencyAxes, 'on');
            plot(frequencyAxes, allTime, fRobot, '-', 'LineWidth',1.6);
            drawBoundaries(frequencyAxes, edges, [0.55 0.55 0.55]);
            hold(frequencyAxes, 'off');
            grid(frequencyAxes, 'on');
            box(frequencyAxes, 'on');
            xlabel(frequencyAxes, 'Time (s)');
            ylabel(frequencyAxes, 'Frequency (Hz)');
            title(frequencyAxes, sprintf('Piecewise command and robot response (%d segments)',numel(durations)));
            legend(frequencyAxes, {'External-field command','Robot frequency'}, 'Location','best');

            cla(phaseAxes);
            plot(phaseAxes, allTime, deltaWrapped*180/pi, 'LineWidth',1.3);
            hold(phaseAxes, 'on');
            drawBoundaries(phaseAxes, edges, [0.55 0.55 0.55]);
            hold(phaseAxes, 'off');
            grid(phaseAxes, 'on');
            box(phaseAxes, 'on');
            set(phaseAxes, 'YLim',[-180 180]);
            xlabel(phaseAxes, 'Time (s)');
            ylabel(phaseAxes, '\delta (degree)');
            title(phaseAxes, 'Wrapped phase difference: \delta = \theta_{field} - \theta_{robot}');

            resultLines = {
                sprintf('Segments                  : %d', numel(durations));
                sprintf('Total command time        : %.6f s', totalTime);
                sprintf('Initial command frequency : %.6f Hz', fInitial);
                sprintf('Final command frequency   : %.6f Hz', allCommand(end));
                sprintf('Torque margin             : %.6f', torqueMargin);
                sprintf('Maximum magnetic torque   : %.6e N m', tauMagMax);
                sprintf('Initial phase lag         : %.6f deg', deltaInitial*180/pi);
                sprintf('Final robot frequency     : %.6f Hz', fRobot(end));
                sprintf('Final tracking error      : %.6f Hz', frequencyError(end));
                sprintf('RMS tracking error        : %.6f Hz', rmsError);
                sprintf('Maximum |tracking error|  : %.6f Hz', maximumAbsoluteError);
                sprintf('Minimum robot frequency   : %.6f Hz', min(fRobot));
                sprintf('Maximum robot frequency   : %.6f Hz', max(fRobot));
                sprintf('Maximum acceleration      : %.6e rad/s^2', max(angularAcceleration));
                sprintf('Maximum deceleration      : %.6e rad/s^2', min(angularAcceleration));
                sprintf('Net relative phase turns  : %.4f', netPhaseTurns)
                };
            set(resultBox, 'String', [resultLines; {''}; holdSummary]);
            setStatus('Simulation complete', [0.75 0.95 0.78]);
        catch ME
            showError(ME.message);
        end

        set(runButton, 'Enable', 'on');
        set(previewButton, 'Enable', 'on');
        drawnow;
    end

%% ---- Data and model helpers -------------------------------------------
    function data = normalizeSegments(data)
        if isempty(data)
            return;
        end
        if size(data,2) ~= 5
            error('The signal table must contain exactly five columns.');
        end

        validTypes = {'Hold','Polynomial','Exponential'};
        for row = 1:size(data,1)
            typeName = data{row,1};
            if ~ischar(typeName)
                try
                    typeName = char(typeName);
                catch
                    typeName = 'Hold';
                end
            end

            % Backward compatibility: an old Linear row becomes a
            % first-order Polynomial row.
            if strcmpi(typeName,'Linear')
                typeName = 'Polynomial';
                oldParameter = numericValue(data{row,5});
                if ~isfinite(oldParameter) || oldParameter == 0
                    data{row,5} = 1;
                end
            else
                matchingIndex = find(strcmpi(typeName,validTypes),1);
                if isempty(matchingIndex)
                    typeName = 'Hold';
                else
                    typeName = validTypes{matchingIndex};
                end
            end
            data{row,1} = typeName;

            for column = 2:5
                data{row,column} = numericValue(data{row,column});
            end

            if get(autoChainCheck,'Value') && row > 1
                data{row,2} = data{row-1,3};
            end

            if strcmp(typeName,'Hold')
                data{row,3} = data{row,2};
                data{row,5} = 0;
            elseif strcmp(typeName,'Polynomial') && isfinite(data{row,5})
                data{row,5} = max(1,round(data{row,5}));
            end
        end
    end

    function validateSegments(data)
        if isempty(data)
            error('At least one signal segment is required.');
        end
        for row = 1:size(data,1)
            values = zeros(1,4);
            for column = 2:5
                values(column-1) = numericValue(data{row,column});
            end
            if any(~isfinite(values))
                error('Row %d contains a non-finite number.',row);
            end
            if values(1) < 0 || values(2) < 0
                error('Row %d contains a negative frequency.',row);
            end
            if values(3) <= 0
                error('Row %d must have a positive duration.',row);
            end
            if strcmpi(data{row,1},'Polynomial')
                polynomialOrder = values(4);
                if polynomialOrder < 1 || ...
                        abs(polynomialOrder-round(polynomialOrder)) > 1E-10
                    error(['Row %d Polynomial order must be a positive ' ...
                        'integer: 1 = linear, 2 = quadratic, 3 = cubic.'],row);
                end
            end
        end
    end

    function [types,starts,ends,durations,shapes,edges] = parseSegments(data)
        numberOfSegments = size(data,1);
        types = cell(numberOfSegments,1);
        starts = zeros(numberOfSegments,1);
        ends = zeros(numberOfSegments,1);
        durations = zeros(numberOfSegments,1);
        shapes = zeros(numberOfSegments,1);
        for row = 1:numberOfSegments
            types{row} = data{row,1};
            starts(row) = numericValue(data{row,2});
            ends(row) = numericValue(data{row,3});
            durations(row) = numericValue(data{row,4});
            shapes(row) = numericValue(data{row,5});
        end
        edges = [0; cumsum(durations)];
    end

    function frequency = evaluateSegmentFrequency(time,segmentStartTime, ...
            duration,typeName,startFrequency,endFrequency,shape)
        s = (time-segmentStartTime)./duration;
        s = min(max(s,0),1);
        switch lower(typeName)
            case 'hold'
                blend = zeros(size(s));
            case {'polynomial','linear'}
                % Polynomial order 1 reproduces the original linear ramp.
                polynomialOrder = max(1,round(shape));
                blend = s.^polynomialOrder;
            case 'exponential'
                if abs(shape) < 1E-9
                    blend = s;
                else
                    blend = (1-exp(-shape.*s))./(1-exp(-shape));
                end
            otherwise
                error('Unsupported segment type: %s',typeName);
        end
        frequency = startFrequency + (endFrequency-startFrequency).*blend;
    end

    function [timeVector,commandVector] = sampleCommand(types,starts,ends, ...
            durations,shapes,edges,maximumPoints)
        totalTime = edges(end);
        sampleCount = max(300,min(maximumPoints,ceil(totalTime/2E-4)+1));
        timeVector = linspace(0,totalTime,sampleCount)';
        commandVector = zeros(size(timeVector));
        for segmentIndex = 1:numel(durations)
            if segmentIndex < numel(durations)
                mask = timeVector >= edges(segmentIndex) & ...
                    timeVector < edges(segmentIndex+1);
            else
                mask = timeVector >= edges(segmentIndex) & ...
                    timeVector <= edges(segmentIndex+1);
            end
            commandVector(mask) = evaluateSegmentFrequency( ...
                timeVector(mask),edges(segmentIndex),durations(segmentIndex), ...
                types{segmentIndex},starts(segmentIndex),ends(segmentIndex), ...
                shapes(segmentIndex));
        end
    end

    function lines = buildHoldSummary(types,edges,time,errorSignal,tolerance)
        lines = {'Hold-segment tracking (last 20% of each Hold):'};
        holdCounter = 0;
        for segmentIndex = 1:numel(types)
            if strcmpi(types{segmentIndex},'Hold')
                holdCounter = holdCounter + 1;
                segmentStart = edges(segmentIndex);
                segmentEnd = edges(segmentIndex+1);
                tailStart = segmentStart + 0.8*(segmentEnd-segmentStart);
                mask = time >= tailStart & time <= segmentEnd;
                if any(mask)
                    tailMaxError = max(abs(errorSignal(mask)));
                    if tailMaxError <= tolerance
                        stateText = 'within tolerance';
                    else
                        stateText = 'outside tolerance';
                    end
                    lines{end+1,1} = sprintf( ... %#ok<AGROW>
                        '  Hold %d (segment %d): max tail error %.5f Hz, %s', ...
                        holdCounter,segmentIndex,tailMaxError,stateText);
                end
            end
        end
        if holdCounter == 0
            lines{end+1,1} = '  No Hold segments.';
        end
    end

    function drawBoundaries(ax,edges,lineColor)
        yLimits = get(ax,'YLim');
        for boundaryIndex = 2:numel(edges)-1
            line(ax,[edges(boundaryIndex) edges(boundaryIndex)],yLimits, ...
                'LineStyle',':','Color',lineColor,'HandleVisibility','off');
        end
        set(ax,'YLim',yLimits);
    end

    function row = getSelectedRow(data)
        row = selectedRow;
        if isempty(row) || ~isfinite(row) || row < 1 || row > size(data,1)
            row = size(data,1);
        end
        row = round(row);
    end

    function updateTotalTime()
        data = get(signalTable,'Data');
        if isempty(data)
            set(totalTimeLabel,'String','Total time: 0 s');
            return;
        end
        durations = zeros(size(data,1),1);
        for row = 1:size(data,1)
            durations(row) = numericValue(data{row,4});
        end
        if all(isfinite(durations))
            set(totalTimeLabel,'String',sprintf('Total time: %.4f s',sum(durations)));
        else
            set(totalTimeLabel,'String','Total time: invalid');
        end
    end

    function value = numericValue(inputValue)
        if isnumeric(inputValue) && isscalar(inputValue)
            value = double(inputValue);
        elseif ischar(inputValue)
            value = str2double(inputValue);
        else
            try
                value = str2double(char(inputValue));
            catch
                value = NaN;
            end
        end
    end

    function setStatus(message,backgroundColor)
        set(statusText,'String',message,'BackgroundColor',backgroundColor);
    end

    function showError(message)
        setStatus('Input/model error',[1.00 0.72 0.72]);
        set(resultBox,'String',{['Error: ' message]});
        set(runButton,'Enable','on');
        set(previewButton,'Enable','on');
        errordlg(message,'Simulation Error','modal');
    end
end
