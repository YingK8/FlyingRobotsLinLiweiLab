function frequency_tracking_statespace_gui
%% FREQUENCY_TRACKING_STATESPACE_GUI
% Front end for frequency_tracking_statespace_sim.m: it builds the piecewise
% frequency command, collects the drive and vertical settings, hands them to
% the engine and plots what comes back. All physics -- the four-state model,
% the RLC drive, ground contact, the hover linearization and the run
% summary -- lives in the sim file; this file owns only widgets, callbacks
% and axes. See the header of frequency_tracking_statespace_sim.m for the
% equations, and call that file directly for scripted or batch runs.

sim = frequency_tracking_statespace_sim();
constants = sim.constants();
defaults = constants.defaults;
exampleData = constants.exampleSegments;

selectedRow = 1;

%% Main window: all positions use normalized coordinates
fig = figure( ...
    'Name', 'Flying Robot Piecewise Frequency State-Space GUI', ...
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

% Settings live on a 16-column x 5-row grid (see gridPos), read as two field
% columns: the left one is the coil/field drive chain, the right one the
% magnet, the vertical model and the pass/fail band. Every label spans 6
% columns and every edit box 2, so the edit boxes line up on two x positions
% and each label has more room than its string needs at 12 pt.
settingsPanel = uipanel( ...
    'Parent', leftPanel, ...
    'Title', 'Global model settings', ...
    'Units', 'normalized', ...
    'Position', [0.025 0.620 0.950 0.300]);

bMaxTip = ['Field amplitude at the LC resonance, mT. Away from resonance the ' ...
    'series RLC passes less current, so B(f) = B_max*R/|Z(f)| and ' ...
    'tau_max(f) = m*B(f). B_max is the ceiling, not the value at every frequency.'];
momentTip = ['Combined dipole moment of the two magnets, mA m^2. tau_max = m*B. ' ...
    'Default 3.6256 is derived from the NdFeB cylinders already in I_robot ' ...
    '(r = h = 0.79375 mm, 7500 kg/m^3) assuming N52, Br = 1.45 T -- an assumption, not a measurement.'];
tolTip = 'Hold pass/fail band: a Hold passes if its last-20% tail tracking error stays within this many Hz.';
inductanceTip = 'Series coil inductance, mH. With C it sets the resonance f_res = 1/(2*pi*sqrt(L*C)).';
capacitanceTip = 'Series tuning capacitance, uF. Blocks DC, so gain -> 0 as f -> 0.';
resistanceTip = 'Series loop resistance, ohm. Sets the resonance sharpness Q = sqrt(L/C)/R.';
hoverTip = ['Spin frequency where lift = weight (hover). L/mg = (f_robot/f_hover)^2. ' ...
    'Also the liftoff frequency: below it the pad carries the difference as a normal ' ...
    'force and z stays at 0; above it z_ddot = g*((f_robot/f_hover)^2 - 1).'];
velTip = ['Initial vertical velocity at t=0, m/s, upward positive. A positive value ' ...
    'launches the robot off the pad; zero or negative is absorbed by the pad if it ' ...
    'is still resting on it.'];

% Left column (starts at 1) = the drive chain: field amplitude, then the
% series RLC. Right column (starts at 9) = magnet, vertical model, tolerance.
addLabel(settingsPanel, gridPos(1,1,6), 'B_max (mT)', bMaxTip);
bMaxEdit = addEdit(settingsPanel, gridPos(7,1,2), ...
    num2str(defaults.bMaxMilliTesla), bMaxTip, []);
addLabel(settingsPanel, gridPos(9,1,6), 'Magnet m (mA m^2)', momentTip);
momentEdit = addEdit(settingsPanel, gridPos(15,1,2), ...
    num2str(defaults.momentMilli), momentTip, @driveParametersChanged);

addLabel(settingsPanel, gridPos(1,2,6), 'Coil L (mH)', inductanceTip);
inductanceEdit = addEdit(settingsPanel, gridPos(7,2,2), ...
    num2str(defaults.inductanceMilliH), inductanceTip, @driveParametersChanged);
addLabel(settingsPanel, gridPos(9,2,6), 'Lift = weight frequency (Hz)', hoverTip);
hoverFrequencyEdit = addEdit(settingsPanel, gridPos(15,2,2), ...
    num2str(defaults.hoverFrequency), hoverTip, []);

addLabel(settingsPanel, gridPos(1,3,6), 'Coil C (uF)', capacitanceTip);
capacitanceEdit = addEdit(settingsPanel, gridPos(7,3,2), ...
    num2str(defaults.capacitanceMicroF), capacitanceTip, @driveParametersChanged);
addLabel(settingsPanel, gridPos(9,3,6), 'Initial vertical vel (m/s)', velTip);
initialVelocityEdit = addEdit(settingsPanel, gridPos(15,3,2), ...
    num2str(defaults.initialVerticalVelocity), velTip, []);

addLabel(settingsPanel, gridPos(1,4,6), 'Coil R (ohm)', resistanceTip);
resistanceEdit = addEdit(settingsPanel, gridPos(7,4,2), ...
    num2str(defaults.resistanceOhm), resistanceTip, @driveParametersChanged);
addLabel(settingsPanel, gridPos(9,4,6), 'Tolerance (Hz)', tolTip);
toleranceEdit = addEdit(settingsPanel, gridPos(15,4,2), ...
    num2str(defaults.frequencyTolerance), tolTip, []);

% Row 5 is readouts, not inputs, so it ignores the two-column split: the
% f_res/Q summary sits directly under the L/C/R it is computed from.
autoChainCheck = uicontrol('Parent', settingsPanel, 'Style', 'checkbox', ...
    'String', 'Auto-chain adjacent segments', 'Value', 1, ...
    'BackgroundColor', get(settingsPanel, 'BackgroundColor'), ...
    'TooltipString', 'On: each segment''s Start is set to the previous segment''s End, for one continuous command.', ...
    'Units', 'normalized', 'Position', gridPos(1,5,7), ...
    'Callback', @autoChainChanged);
driveLabel = uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', '', 'FontWeight', 'bold', 'FontSize', 10, ...
    'HorizontalAlignment', 'right', ...
    'BackgroundColor', get(settingsPanel, 'BackgroundColor'), ...
    'TooltipString', 'Series-RLC resonance and quality factor from L, C, R.', ...
    'Units', 'normalized', 'Position', gridPos(8,5,4));
totalTimeLabel = uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', 'Total time: 1.700 s', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', ...
    'BackgroundColor', get(settingsPanel, 'BackgroundColor'), ...
    'TooltipString', 'Sum of all segment Durations.', ...
    'Units', 'normalized', 'Position', gridPos(12,5,5));

% Plain ASCII: uicontrol static text has no TeX interpreter, so markup would
% render as literal backslashes here.
uicontrol('Parent', settingsPanel, 'Style', 'text', ...
    'String', { ...
        'x = [delta; omega; z; z_dot], u = f_field (Hz). tau_mag = m*B_max*R/|Z(f)|*sin(delta), weakest at DC.'; ...
        'On a pad until N/W = 1 - (f_robot/f_hover)^2 hits 0; then z_ddot = g*(L/W - 1), undamped.'}, ...
    'FontSize', 9, ...
    'ForegroundColor', [0.35 0.35 0.35], ...
    'BackgroundColor', get(settingsPanel, 'BackgroundColor'), ...
    'HorizontalAlignment', 'left', ...
    'Units', 'normalized', ...
    'Position', [0.020 0.005 0.960 0.176]);

%% Add buttons
buttonY1 = 0.568;
buttonH = 0.042;
buttonGap = 0.008;
buttonW = (0.950 - 5*buttonGap)/6;
buttonX0 = 0.025;
buttonLabels = {'+ Hold', '+ Poly', '+ Exp', 'Copy', 'Delete', 'Clear All'};
buttonCallbacks = {@(~,~)addSegment('Hold'), @(~,~)addSegment('Polynomial'), ...
    @(~,~)addSegment('Exponential'), @copySegment, @deleteSegment, @clearSegments};
buttonTooltips = {'Append a Hold: constant-frequency segment.', ...
    'Append a Polynomial: power-ramp segment (order n).', ...
    'Append an Exponential: exponential-ramp segment (curvature k).', ...
    'Duplicate the selected row.', 'Delete the selected row.', ...
    'Remove all segments.'};
for k = 1:6
    uicontrol('Parent', leftPanel, 'Style', 'pushbutton', ...
        'String', buttonLabels{k}, ...
        'TooltipString', buttonTooltips{k}, ...
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
    'TooltipString', sprintf(['One row per segment.\n' ...
        'Type: Hold / Polynomial / Exponential.  Start, End in Hz; Duration in s.\n' ...
        'Order / exp k: polynomial order (integer >=1: 1 linear, 2 quadratic, 3 cubic)\n' ...
        'or exponential curvature k (0 = linear); ignored for Hold.']), ...
    'Units', 'normalized', ...
    'Position', [0.025 0.320 0.950 0.232], ...
    'CellEditCallback', @tableEdited, ...
    'CellSelectionCallback', @tableSelected);

%% Move and run buttons
buttonY2 = 0.268;
buttonW2 = (0.950 - 3*buttonGap)/4;
moveLabels = {'Move Up', 'Move Down', 'Preview Command', 'Run Simulation'};
moveCallbacks = {@(~,~)moveSegment(-1), @(~,~)moveSegment(1), @previewCommand, @runSimulation};
moveTooltips = {'Move the selected segment earlier.', ...
    'Move the selected segment later.', ...
    'Plot the command only (no dynamics solved).', ...
    'Integrate the four-state model and plot the response with stats.'};
for k = 1:4
    h = uicontrol('Parent', leftPanel, 'Style', 'pushbutton', ...
        'String', moveLabels{k}, ...
        'TooltipString', moveTooltips{k}, ...
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

addLabel(infoPanel, [0.025 0.560 0.420 0.280], ...
    sprintf('I_robot = %.5e kg m^2', constants.I_robot), ...
    'Spin-axis moment of inertia (fixed): body + propellers + two magnets. See lecture_notes.md Sec. 3.');
addLabel(infoPanel, [0.025 0.150 0.420 0.280], ...
    sprintf('k_drag = %.5e N m/Hz^2', constants.k_drag), ...
    'Quadratic drag-torque coefficient from the fit tau_drag = -k*f^2.');
addLabel(infoPanel, [0.465 0.560 0.290 0.280], ...
    sprintf('Drag-fit R^2 = %.5f', constants.R_squared), ...
    'Goodness of the quadratic drag fit (near 1 validates the f^2 model).');
uicontrol('Parent', infoPanel, 'Style', 'pushbutton', ...
    'String', 'Reset Example', ...
    'TooltipString', 'Restore the example table and default settings.', ...
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

%% Right panel: frequency, phase and altitude axes plus the result box
frequencyAxes = axes('Parent', rightPanel, 'Units', 'normalized', ...
    'Position', [0.085 0.685 0.875 0.245], 'Box', 'on');
styleAxes(frequencyAxes, 'Frequency (Hz)', 'External-field command preview');

phaseAxes = axes('Parent', rightPanel, 'Units', 'normalized', ...
    'Position', [0.085 0.465 0.875 0.155], 'Box', 'on');
resetPhaseAxes();

verticalAxes = axes('Parent', rightPanel, 'Units', 'normalized', ...
    'Position', [0.085 0.245 0.875 0.155], 'Box', 'on');
resetVerticalAxes();

resultBox = uicontrol( ...
    'Parent', rightPanel, ...
    'Style', 'edit', ...
    'Max', 20, ...
    'Min', 0, ...
    'Enable', 'inactive', ...
    'BackgroundColor', 'white', ...
    'HorizontalAlignment', 'left', ...
    'FontName', 'Monospaced', ...
    'FontSize', 10, ...
    'String', {'Edit the segment table, then click Run Simulation.'}, ...
    'TooltipString', 'Read-only run summary: tracking errors, torque, phase turns, altitude, state-space matrices, per-Hold pass/fail.', ...
    'Units', 'normalized', ...
    'Position', [0.035 0.025 0.925 0.175]);

updateDriveLabel();
previewCommand();

%% ---- GUI callbacks -----------------------------------------------------
    function addSegment(typeName)
        data = get(signalTable, 'Data');
        if isempty(data)
            startFrequency = 160;
        else
            startFrequency = sim.numericValue(data{end,3});
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
        data = [data(1:row,:); data(row,:); data(row+1:end,:)];
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
        resetPhaseAxes();
        resetVerticalAxes();
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
                value = sim.numericValue(data{row,column});
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
            sim.validateSegments(data);
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
        set(bMaxEdit, 'String', num2str(defaults.bMaxMilliTesla));
        set(inductanceEdit, 'String', num2str(defaults.inductanceMilliH));
        set(capacitanceEdit, 'String', num2str(defaults.capacitanceMicroF));
        set(resistanceEdit, 'String', num2str(defaults.resistanceOhm));
        set(momentEdit, 'String', num2str(defaults.momentMilli));
        updateDriveLabel();
        set(toleranceEdit, 'String', num2str(defaults.frequencyTolerance));
        set(initialVelocityEdit, 'String', num2str(defaults.initialVerticalVelocity));
        set(hoverFrequencyEdit, 'String', num2str(defaults.hoverFrequency));
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
            sim.validateSegments(data);
            set(signalTable, 'Data', data);
            [types, starts, ends, durations, shapes, edges] = sim.parseSegments(data);
            [timePreview, commandPreview] = sim.sampleCommand( ...
                types, starts, ends, durations, shapes, edges, 5000);

            cla(frequencyAxes);
            plot(frequencyAxes, timePreview, commandPreview, '--', 'LineWidth', 1.7);
            styleAxes(frequencyAxes, 'Frequency (Hz)', 'External-field command preview');
            legend(frequencyAxes, {'External-field command'}, 'Location', 'best');

            resetPhaseAxes();
            title(phaseAxes, 'Run Simulation to calculate robot phase response');

            resetVerticalAxes();
            title(verticalAxes, ...
                'Run Simulation to calculate liftoff and vertical displacement');
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

        [drive, driveError] = readDriveParameters();
        [frequencyTolerance, tolError] = readScalar( ...
            toleranceEdit, 'Tracking tolerance', true);
        [initialVerticalVelocity, velError] = readScalar( ...
            initialVelocityEdit, 'Initial vertical velocity', false);
        [hoverFrequency, hoverError] = readScalar( ...
            hoverFrequencyEdit, 'Lift = weight frequency', true);

        problems = {driveError, tolError, velError, hoverError};
        firstProblem = find(~cellfun(@isempty, problems), 1);
        if ~isempty(firstProblem)
            showError(problems{firstProblem});
            return;
        end

        try
            data = normalizeSegments(data);
            sim.validateSegments(data);
            set(signalTable, 'Data', data);
        catch ME
            showError(ME.message);
            return;
        end

        set(runButton, 'Enable', 'off');
        set(previewButton, 'Enable', 'off');
        setStatus('Solving four-state model...', [1.00 0.90 0.65]);
        drawnow;

        try
            result = sim.run(data, struct( ...
                'drive',                   drive, ...
                'hoverFrequency',          hoverFrequency, ...
                'initialVerticalVelocity', initialVerticalVelocity, ...
                'frequencyTolerance',      frequencyTolerance, ...
                'autoChain',               logical(get(autoChainCheck,'Value'))));

            plotResult(result);
            set(resultBox, 'String', sim.summaryLines(result));
            setStatus('Simulation complete', [0.75 0.95 0.78]);
        catch ME
            showError(ME.message);
        end

        set(runButton, 'Enable', 'on');
        set(previewButton, 'Enable', 'on');
        drawnow;
    end

%% ---- Plotting ----------------------------------------------------------
    function plotResult(result)
        edges = result.edges;

        cla(frequencyAxes);
        plot(frequencyAxes, result.time, result.command, '--', 'LineWidth',1.6);
        hold(frequencyAxes, 'on');
        plot(frequencyAxes, result.time, result.fRobot, '-', 'LineWidth',1.6);
        drawBoundaries(frequencyAxes, edges, [0.55 0.55 0.55]);
        hold(frequencyAxes, 'off');
        styleAxes(frequencyAxes, 'Frequency (Hz)', ...
            sprintf('Piecewise command and robot response (%d segments)', ...
                numel(result.durations)));
        legend(frequencyAxes, {'External-field command','Robot frequency'}, 'Location','best');

        % Phase axis carries delta on the left and the torque it buys
        % on the right, because tau_mag = tau_max*sin(delta).
        resetPhaseAxes();
        yyaxis(phaseAxes, 'left');
        plot(phaseAxes, result.time, result.deltaWrapped*180/pi, 'LineWidth',1.3);
        set(phaseAxes, 'YLim',[-180 180]);
        ylabel(phaseAxes, '\delta (degree)');
        yyaxis(phaseAxes, 'right');
        plot(phaseAxes, result.time, 1E6*result.tauMagnetic, '-', 'LineWidth',1.1);
        hold(phaseAxes, 'on');
        % Envelope: the torque the coils could deliver at each command
        % frequency, i.e. what the RLC gain allows before sin(delta).
        plot(phaseAxes, result.time, 1E6*result.tauMaxTrace, ':', 'LineWidth',1.0);
        plot(phaseAxes, result.time, -1E6*result.tauMaxTrace, ':', 'LineWidth',1.0);
        hold(phaseAxes, 'off');
        ylabel(phaseAxes, '\tau_{mag} (\muN m)');
        set(phaseAxes, 'YLim', 1E6*max(result.tauMagMaxAtResonance,eps)*[-1 1]);
        yyaxis(phaseAxes, 'left');
        hold(phaseAxes, 'on');
        drawBoundaries(phaseAxes, edges, [0.55 0.55 0.55]);
        hold(phaseAxes, 'off');
        grid(phaseAxes, 'on');
        box(phaseAxes, 'on');
        xlabel(phaseAxes, 'Time (s)');
        title(phaseAxes, ['State 1: \delta = \theta_{field} - \theta_{robot}, ' ...
            'and the torque it delivers \tau_{max}sin\delta']);

        % Altitude on the left, the pad reaction N/W on the right as a
        % fraction of weight: it falls to zero at liftoff and releases z.
        resetVerticalAxes();
        yyaxis(verticalAxes, 'right');
        normalLine = plot(verticalAxes, result.time, result.normalForceRatio, ...
            '-', 'LineWidth',1.2);
        ylabel(verticalAxes, 'force / weight');
        set(verticalAxes, 'YLim', forceLimits(result.normalForceRatio));
        yyaxis(verticalAxes, 'left');
        heightLine = plot(verticalAxes, result.time, ...
            1000*result.verticalDisplacement, 'LineWidth',1.5);
        ylabel(verticalAxes, 'z (mm)');
        hold(verticalAxes, 'on');
        drawBoundaries(verticalAxes, edges, [0.55 0.55 0.55]);
        drawMarkers(verticalAxes, result.liftOffTimes, [0.85 0.33 0.10]);
        drawMarkers(verticalAxes, result.touchDownTimes, [0.30 0.30 0.75]);
        hold(verticalAxes, 'off');
        % Built last, from explicit handles, so the left-axis z line is
        % named too and AutoUpdate cannot append the boundary markers.
        legend(verticalAxes, [heightLine normalLine], ...
            {'z (mm)','N / weight (pad)'}, ...
            'Location','best', 'AutoUpdate','off');
        grid(verticalAxes, 'on');
        box(verticalAxes, 'on');
        xlabel(verticalAxes, 'Time (s)');
        if isnan(result.liftOffTime)
            title(verticalAxes, ['State 3: z stays on the pad -- ' ...
                'f_{robot} never reaches f_{hover}']);
        else
            title(verticalAxes, sprintf( ...
                'State 3: z, normal force releases at liftoff t = %.4f s', ...
                result.liftOffTime));
        end
    end

%% ---- Reading the settings ----------------------------------------------
    function [drive, errorMessage] = readDriveParameters()
        % Read and validate the coil/magnet fields, returning the SI drive
        % struct the sim expects.
        % Columns: handle, SI scale, display name, unit.
        spec = {
            bMaxEdit,        1E-3, 'B_max',             'mT'
            inductanceEdit,  1E-3, 'Coil inductance L', 'mH'
            capacitanceEdit, 1E-6, 'Coil capacitance C','uF'
            resistanceEdit,  1.0,  'Coil resistance R', 'ohm'
            momentEdit,      1E-3, 'Magnet moment m',   'mA m^2'
            };

        drive = [];
        errorMessage = '';
        siValues = zeros(size(spec,1),1);
        for index = 1:size(spec,1)
            value = str2double(get(spec{index,1}, 'String'));
            if ~isfinite(value) || value <= 0
                errorMessage = sprintf( ...
                    '%s must be a positive finite number of %s.', ...
                    spec{index,3}, spec{index,4});
                return;
            end
            siValues(index) = spec{index,2}*value;
        end

        drive = sim.makeDrive(siValues(1), siValues(2), siValues(3), ...
            siValues(4), siValues(5));
    end

    function [value, errorMessage] = readScalar(handle, name, mustBePositive)
        % Read a numeric scalar from an edit box. Returns errorMessage rather
        % than throwing, so the caller can collect several and report the first.
        errorMessage = '';
        value = str2double(get(handle, 'String'));
        if ~isfinite(value)
            errorMessage = sprintf('%s must be finite.', name);
        elseif mustBePositive && value <= 0
            errorMessage = sprintf('%s must be a positive finite number.', name);
        end
    end

    function data = normalizeSegments(data)
        % Thin wrapper: the auto-chain flag is a widget, so only the GUI
        % knows it.
        data = sim.normalizeSegments(data, logical(get(autoChainCheck,'Value')));
    end

    function driveParametersChanged(~,~)
        updateDriveLabel();
    end

    function updateDriveLabel()
        [drive, errorMessage] = readDriveParameters();
        if isempty(errorMessage)
            % %.0f keeps this inside its 4 columns; the exact value is
            % printed to 4 decimals in the result box.
            set(driveLabel, 'String', sprintf('f_res %.0f Hz, Q %.2f', ...
                drive.resonanceHz, drive.qualityFactor), ...
                'ForegroundColor', [0 0 0]);
        else
            set(driveLabel, 'String', 'f_res: invalid', ...
                'ForegroundColor', [0.7 0 0]);
        end
    end

%% ---- Layout and formatting helpers -------------------------------------
    function position = gridPos(column, row, span)
        % Cell rectangle on the settings panel's 16-column x 5-row grid.
        % Columns 1-8 are the left field column, 9-16 the right one, so a
        % label at 1 or 9 and its edit box at 7 or 15 puts every edit box on
        % one of exactly two x positions. Five rows span 0.780 down to 0.190
        % at a pitch of 0.1475, with cells 0.100 tall; the model-description
        % text below starts at 0.005 and must stay clear of the last row at
        % 0.190. That text gets 0.176 (~43 px in the panel's ~244 px) --
        % measure the uicontrol Extent against Position before shrinking it.
        columnWidth = 0.960/16;
        rowBottom = [0.780 0.6325 0.485 0.3375 0.190];
        position = [0.020 + (column-1)*columnWidth, rowBottom(row), ...
            span*columnWidth - 0.006, 0.100];
    end

    function limits = forceLimits(values)
        % Symmetric-ish padded limits for the force/weight axis. Always
        % includes 0 and 1 so the "full weight on the pad" line and the
        % zero crossing stay visible even on runs that never leave the pad.
        low = min([0; -0.05; values(:)]);
        high = max([1.05; values(:)]);
        pad = 0.05*(high - low);
        limits = [low - pad, high + pad];
    end

    function handle = addLabel(parent, position, labelText, tip)
        handle = uicontrol('Parent', parent, 'Style', 'text', ...
            'String', labelText, 'HorizontalAlignment', 'left', ...
            'TooltipString', tip, ...
            'BackgroundColor', get(parent, 'BackgroundColor'), ...
            'Units', 'normalized', 'Position', position);
    end

    function handle = addEdit(parent, position, valueText, tip, callback)
        handle = uicontrol('Parent', parent, 'Style', 'edit', ...
            'String', valueText, 'BackgroundColor', 'white', ...
            'TooltipString', tip, ...
            'Units', 'normalized', 'Position', position);
        if ~isempty(callback)
            set(handle, 'Callback', callback);
        end
    end

    function styleAxes(ax, yLabelText, titleText)
        grid(ax, 'on');
        box(ax, 'on');
        xlabel(ax, 'Time (s)');
        ylabel(ax, yLabelText);
        title(ax, titleText);
    end

    function resetVerticalAxes()
        % Same reason as resetPhaseAxes: this axis carries a yyaxis pair, so
        % only cla(...,'reset') clears it without leaving a stale right ruler.
        cla(verticalAxes,'reset');
        styleAxes(verticalAxes, 'z (mm)', ...
            'Vertical displacement, upward positive, and the pad normal force');
    end

    function resetPhaseAxes()
        % cla(...,'reset') is the only clean way to drop a yyaxis pair.
        % It preserves Position and Units, so the layout is untouched.
        cla(phaseAxes,'reset');
        styleAxes(phaseAxes, '\delta (degree)', 'Wrapped phase difference');
        set(phaseAxes, 'YLim', [-180 180]);
    end

    function drawBoundaries(ax,edges,lineColor)
        yLimits = get(ax,'YLim');
        for boundaryIndex = 2:numel(edges)-1
            line(ax,[edges(boundaryIndex) edges(boundaryIndex)],yLimits, ...
                'LineStyle',':','Color',lineColor,'HandleVisibility','off');
        end
        set(ax,'YLim',yLimits);
    end

    function drawMarkers(ax,times,lineColor)
        % Like drawBoundaries but every entry is drawn, because these are
        % event instants rather than the interior edges of a partition.
        yLimits = get(ax,'YLim');
        for markerIndex = 1:numel(times)
            line(ax,[times(markerIndex) times(markerIndex)],yLimits, ...
                'LineStyle','--','Color',lineColor,'HandleVisibility','off');
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
        durations = cellfun(sim.numericValue, data(:,4));
        if all(isfinite(durations))
            set(totalTimeLabel,'String',sprintf('Total time: %.4f s',sum(durations)));
        else
            set(totalTimeLabel,'String','Total time: invalid');
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
