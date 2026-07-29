%% FREQUENCY_TRACKING_STATESPACE_SIM
% Headless engine behind frequency_tracking_statespace_gui.m: the robot
% model, the piecewise frequency command, the four-state integration with
% ground contact, the hover linearization and the text summary. It touches
% no graphics, so it can be called from a script, a test or a sweep.
%
%   x = [delta; omega; z; z_dot],  u = f_field(t)
%
%   delta_dot = 2*pi*f_field(t) - omega
%   omega_dot = (tau_max(f_field)*sin(delta) - k_drag*f_robot*|f_robot|) / I_robot
%   z_ddot    = g*((f_robot/f_hover)^2 - 1)
%   z_ddot    = 0, z = z_dot = 0              while resting on the pad
% Where:
%   delta    field-to-robot phase lag, theta_field - theta_robot (rad)
%   omega    robot spin rate (rad/s); f_robot = omega/(2*pi) (Hz)
%   z        height above the pad (m), upward positive
%   f_field  commanded field frequency (Hz), the only input
%   f_hover  spin frequency where lift = weight (Hz)
%   tau_max  peak magnetic torque m*B(f) at the command frequency (N m)
%   k_drag   quadratic drag-torque coefficient (N m/Hz^2), fit in modelConstants
%   I_robot  spin-axis moment of inertia (kg m^2)
%
% Heave is undamped: air resistance is not modelled, so the vertical channel
% is thrust minus weight alone. The hover calibration k_T*f_hover^2 = m_R*g
% (thrust coefficient k_T, robot mass m_R) divides both out, so z_ddot carries
% no free parameter. A thrust surplus therefore accelerates indefinitely
% instead of settling at a terminal climb rate, and both vertical eigenvalues
% sit at the origin.
%
% Ground contact:
% The robot starts on a pad, so z is a UNILATERAL constraint. While it rests
% there the pad supplies a normal force
%
%   N/W = 1 - (f_robot/f_hover)^2 >= 0
%
% that cancels the net downward force, pinning z and z_dot at zero. N falls to
% zero at f_robot = f_hover: that instant is liftoff, and the free-flight
% z_ddot applies from then on. If lift later drops below weight the robot
% descends, lands inelastically at z = 0 (z_dot -> 0) and the pad reloads. N
% is a fraction of weight because the vertical equation carries no robot mass.
%
% Drive model:
% Series-RLC coil channel, so the peak torque tau_max = m*B moves with the
% drive frequency:
%
%   X(f)       = 2*pi*f*L - 1/(2*pi*f*C)
%   gain(f)    = R/sqrt(R^2 + X(f)^2)          = |I(f)|/|I(f_res)|, <= 1
%   tau_max(f) = m * B_max * gain(f)
%
% gain = 1 at f_res = 1/(2*pi*sqrt(L*C)) and gain -> 0 at DC, where the series
% capacitor blocks current; that rolloff is what limits spin-up from rest.
% Because tau_max moves with the command, the torque margin and the step-out
% ceiling are outputs rather than inputs.
%
% Usage:
%   result = frequency_tracking_statespace_sim(segments, parameters)
%       segments   n-by-5 cell array {Type, Start Hz, End Hz, Duration s, Order/k}
%       parameters struct with fields
%                    drive                   from api.makeDrive(...)
%                    hoverFrequency          Hz
%                    initialVerticalVelocity m/s, upward positive (default 0)
%                    frequencyTolerance      Hz, Hold pass/fail band (default 1)
%                    autoChain               logical (default true)
%       result     struct of traces, scalars and the [A,B] hover linearization
%
%   api = frequency_tracking_statespace_sim()
%       Struct of function handles, for calling the pieces on their own. The
%       struct literal below lists the fields.
%
% Example:
%   api   = frequency_tracking_statespace_sim();
%   d     = api.constants().defaults;
%   drive = api.makeDrive(d.bMaxSI, d.inductanceSI, d.capacitanceSI, ...
%                         d.resistanceSI, d.momentSI);
%   out   = frequency_tracking_statespace_sim( ...
%               {'Polynomial',0,140,20.0,2; 'Hold',140,140,1.00,0}, ...
%               struct('drive',drive,'hoverFrequency',140));
%   disp(char(api.summaryLines(out)));


function varargout = frequency_tracking_statespace_sim(segments, parameters)
    if nargin == 0
        varargout{1} = struct( ...
            'constants',                @modelConstants, ...
            'makeDrive',                @makeDrive, ...
            'coilGain',                 @coilGain, ...
            'coilGainDerivative',       @coilGainDerivative, ...
            'hoverStateSpace',          @hoverStateSpace, ...
            'findStepOutCeiling',       @findStepOutCeiling, ...
            'normalizeSegments',        @normalizeSegments, ...
            'validateSegments',         @validateSegments, ...
            'parseSegments',            @parseSegments, ...
            'evaluateSegmentFrequency', @evaluateSegmentFrequency, ...
            'sampleCommand',            @sampleCommand, ...
            'numericValue',             @numericValue, ...
            'run',                      @runModel, ...
            'summaryLines',             @summaryLines, ...
            'ternary',                  @ternary);
        return;
    end
    
    varargout{1} = runModel(segments, parameters);
end


%% Robot model constants

function constants = modelConstants()
    % Inertia, drag fit and drive defaults. Cached: the drag fit is the same
    % every call and runModel/hoverStateSpace ask for it repeatedly.
    persistent cached
    if ~isempty(cached)
        constants = cached;
        return;
    end
    
    % 3.89E-9 frame moment + two magnet moments
    I_robot = 3.89E-9 + 2.0 * ...
        (1.0/12.0 * 1.17832E-5 * ...
        (3.0 * 0.79375^2.0 * 1E-6 + 0.79375^2.0 * 1E-6) ...
        + 1.17832E-5 * (0.496875^2.0 * 1E-6));
    
    % Rig-measured drag torque against spin frequency, fitted below to -k_drag*f^2.
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
    
    % The two drive magnets are already implicit in I_robot: NdFeB cylinders with
    % r = h = 0.79375 mm and mass 1.17832E-5 kg, so V = pi*r^2*h = 1.5711E-9 m^3
    % and the density works out to 7500 kg/m^3, which is NdFeB. Taking N52
    % (Br = 1.45 T), each magnet has m = Br*V/mu0 = 1.8128E-3 A m^2 and the aligned
    % pair gives 3.6256E-3 A m^2.
    defaults = struct( ...
        'momentMilli',      3.6256, ...   % mA m^2, both magnets together
        'bMaxMilliTesla',   2.50, ...     % mT, field amplitude at resonance
        'inductanceMilliH', 1.4, ...
        'capacitanceMicroF', 500.0, ...
        'resistanceOhm',    1.7, ...
        'hoverFrequency',   140, ...
        'frequencyTolerance', 1, ...
        'initialVerticalVelocity', 0);
    defaults.momentSI      = 1E-3*defaults.momentMilli;
    defaults.bMaxSI        = 1E-3*defaults.bMaxMilliTesla;
    defaults.inductanceSI  = 1E-3*defaults.inductanceMilliH;
    defaults.capacitanceSI = 1E-6*defaults.capacitanceMicroF;
    defaults.resistanceSI  = defaults.resistanceOhm;
    
    cached = struct( ...
        'I_robot',   I_robot, ...
        'k_drag',    k_drag, ...
        'R_squared', 1 - SS_res/SS_tot, ...
        'gravity',   9.80665, ...
        'defaults',  defaults, ...
        'exampleSegments', {{ ...
            'Polynomial',  0, 140, 20.0, 2; ...
            'Hold',        140, 140, 1.00, 0}});
    constants = cached;
end

%% ---- Drive chain -------------------------------------------------------
function drive = makeDrive(bMaxSI, inductanceSI, capacitanceSI, resistanceSI, momentSI)
    % Series-RLC drive channel in SI, with its resonance and Q filled in.
    drive = struct( ...
        'bMaxSI',        bMaxSI, ...
        'inductanceSI',  inductanceSI, ...
        'capacitanceSI', capacitanceSI, ...
        'resistanceSI',  resistanceSI, ...
        'momentSI',      momentSI);
    drive.resonanceHz = 1/(2*pi*sqrt(drive.inductanceSI*drive.capacitanceSI));
    drive.qualityFactor = ...
        sqrt(drive.inductanceSI/drive.capacitanceSI)/drive.resistanceSI;
end

function gainValue = coilGain(frequencyHz, drive)
    % |I(f)| / |I(f_res)| for a series RLC driven at frequencyHz.
    % Exactly 1 at resonance; 0 at DC, where the series cap is open.
    omegaElectrical = 2*pi*abs(frequencyHz);
    if omegaElectrical <= 0
        gainValue = 0;
        return;
    end
    reactance = omegaElectrical*drive.inductanceSI ...
        - 1/(omegaElectrical*drive.capacitanceSI);
    gainValue = drive.resistanceSI / ...
        sqrt(drive.resistanceSI^2 + reactance^2);
end

function derivativeValue = coilGainDerivative(frequencyHz, drive)
    % d(gain)/df, analytic. Needed for the B matrix because the command
    % frequency now modulates the available torque.
    omegaElectrical = 2*pi*abs(frequencyHz);
    if omegaElectrical <= 0
        derivativeValue = 0;
        return;
    end
    reactance = omegaElectrical*drive.inductanceSI ...
        - 1/(omegaElectrical*drive.capacitanceSI);
    dReactanceDomega = drive.inductanceSI ...
        + 1/(omegaElectrical^2*drive.capacitanceSI);
    denominator = (drive.resistanceSI^2 + reactance^2)^1.5;
    derivativeValue = 2*pi * ...
        (-drive.resistanceSI*reactance*dReactanceDomega/denominator);
end

function ceilingHz = findStepOutCeiling(tauMaxOf, hoverFrequency)
    % Highest frequency the field can still hold synchronously:
    % largest f with tau_max(f) >= k_drag*f^2.
    %
    % This is found numerically by sampling values:

    k_drag = modelConstants().k_drag;
    scanTop = max(10*hoverFrequency, 10*1000);
    frequencyGrid = linspace(0, scanTop, 200001);
    frequencyGrid(1) = [];
    sustainable = arrayfun(tauMaxOf, frequencyGrid) >= ...
        k_drag*frequencyGrid.^2;
    lastIndex = find(sustainable, 1, 'last');
    if isempty(lastIndex) || lastIndex == numel(frequencyGrid)
        ceilingHz = Inf;
    else
        ceilingHz = frequencyGrid(lastIndex);
    end
end

%% ---- Hover linearization -----------------------------------------------
function [A, B, deltaHover, reachable] = hoverStateSpace(drive, hoverFrequency)
    % Jacobian of the four-state model at the hover trim.
    % States [delta; omega; z; z_dot], input f_field in Hz.
    % This is the FREE-FLIGHT Jacobian. The hover trim sits exactly on
    % the contact boundary (N = 0), and the constraint is one-sided, so
    % it is the valid linearization for the airborne side only: on the
    % pad the z rows are identically zero instead.
    constants = modelConstants();
    I_robot = constants.I_robot;
    k_drag = constants.k_drag;
    
    omegaTrim = 2*pi*hoverFrequency;
    tauHover = k_drag*hoverFrequency^2;
    tauMaxHover = drive.momentSI*drive.bMaxSI*coilGain(hoverFrequency, drive);
    if tauMaxHover > 0
        sinHover = tauHover/tauMaxHover;
    else
        sinHover = Inf;
    end
    reachable = abs(sinHover) <= 1;
    if reachable
        deltaHover = asin(sinHover);
    else
        % tau_max cannot hold f_hover: no equilibrium exists there, so
        % report the saturated lag and let the caller flag it.
        deltaHover = pi/2;
    end
    
    A = zeros(4);
    A(1,2) = -1;
    A(2,1) = tauMaxHover*cos(deltaHover)/I_robot;
    A(2,2) = -k_drag*omegaTrim/(2*pi^2*I_robot);
    A(3,4) = 1;
    % g/(pi*f_h): thrust sensitivity to ACTUAL spin speed, notes Sec. 8.2.
    A(4,2) = 2*constants.gravity/(2*pi*hoverFrequency);
    % A(4,4) stays 0: no z_dot feedback, so z and z_dot form a bare double
    % integrator.

    % Commanding a different frequency now moves the operating point on
    % the RLC curve, so it changes the available torque as well as the
    % phase. That second path is what makes B(2) nonzero here but zero
    % in the constant-tau_max version of this model.
    dTauMaxdf = drive.momentSI*drive.bMaxSI*coilGainDerivative(hoverFrequency, drive);
    B = [2*pi; dTauMaxdf*sin(deltaHover)/I_robot; 0; 0];
end

%% ---- Ground-contact events ---------------------------------------------
% Real functions rather than anonymous ones: ode45 calls an event function
% with three outputs, and an anonymous @(t,y)deal(...) cannot supply them.
function [value, isterminal, direction] = liftOffCrossing(~, y, hoverFrequency)
    % Leaving the pad: lift/weight reaching 1 is the instant the normal
    % force N = W - L has just been used up.
    value = (y(2)/(2*pi)/hoverFrequency)^2 - 1;
    isterminal = 1;
    direction = 1;
end

function [value, isterminal, direction] = touchDownCrossing(~, y)
    % Coming back down onto the pad: z crossing zero from above.
    value = y(3);
    isterminal = 1;
    direction = -1;
end

%% ---- Simulation --------------------------------------------------------
function result = runModel(segments, parameters)
    % Integrate the four-state model over the segment list and derive the
    % result contract in the file header. Integration restarts at every
    % liftoff and landing, so contact discontinuities land on exact times;
    % a run that chatters between modes gives up after MAX_CONTACT_SWITCHES
    % and flags result.contactSwitchesExhausted.
    constants = modelConstants();
    I_robot = constants.I_robot;
    k_drag = constants.k_drag;
    gravity = constants.gravity;
    
    parameters = fillDefaults(parameters, struct( ...
        'hoverFrequency',          constants.defaults.hoverFrequency, ...
        'initialVerticalVelocity', 0, ...
        'frequencyTolerance',      constants.defaults.frequencyTolerance, ...
        'autoChain',               true));
    if ~isfield(parameters, 'drive')
        error('parameters.drive is required; build it with the makeDrive API.');
    end
    drive = parameters.drive;
    hoverFrequency = parameters.hoverFrequency;
    initialVerticalVelocity = parameters.initialVerticalVelocity;

    segments = normalizeSegments(segments, parameters.autoChain);
    validateSegments(segments);
    [types, starts, ends, durations, shapes, edges] = parseSegments(segments);
    
    fInitial = starts(1);
    
    % tau_max is not a single number: the series RLC passes less current
    % away from resonance, so the peak torque available at drive frequency
    % f is m*B_max*gain(f). It is evaluated at the COMMAND frequency, since
    % that is what drives the coils.
    tauMaxOf = @(f) drive.momentSI * drive.bMaxSI * coilGain(f, drive);
    tauMagMaxAtResonance = drive.momentSI * drive.bMaxSI;
    gainAtHover = coilGain(hoverFrequency, drive);
    tauMagMaxAtHover = tauMaxOf(hoverFrequency);
    tauHoverReference = k_drag * hoverFrequency^2;
    
    % The margin is an OUTPUT, not an input: it is whatever the coils and
    % magnets happen to deliver at the hover frequency.
    effectiveMargin = tauMagMaxAtHover / tauHoverReference;
    stepOutCeiling = findStepOutCeiling(tauMaxOf, hoverFrequency);
    
    % Start phase-locked: the lag that cancels drag at fInitial. At
    % fInitial = 0 both drag and tau_max vanish, so start at delta=0.
    tauMaxInitial = tauMaxOf(fInitial);
    if tauMaxInitial <= 0
        sinInitial = 0;
    else
        sinInitial = k_drag * fInitial^2 / tauMaxInitial;
    end
    initialAboveCeiling = sinInitial > 1;
    deltaInitial = asin(min(max(sinInitial, -1), 1));
    
    % Four-state initial condition. z starts at the pad surface, z = 0; the
    % initial vertical velocity is a state, not a constant of integration
    % bolted on afterwards.
    stateAtStart = [deltaInitial; 2*pi*fInitial; 0; initialVerticalVelocity];
    
    % The vertical channel is unilateral: the pad can push up but not pull
    % down. Two right-hand sides, one per contact mode, and an event that
    % fires at each transition.
    %   ground: z and z_dot are pinned, the pad absorbing N = W - L
    %   flight: the free double integrator
    % Liftoff is lift/weight crossing 1 upward (f_robot = f_hover, so N has
    % just reached zero); touchdown is z crossing 0 downward.
    liftRatioOf = @(omegaValue)(omegaValue/(2*pi)/hoverFrequency)^2;
    spinRows = @(t,y,command)[
        2*pi*command(t) - y(2);
        (tauMaxOf(command(t))*sin(y(1)) ...
        - k_drag*(y(2)/(2*pi))*abs(y(2)/(2*pi))) / I_robot
        ];
    flightODE = @(t,y,command)[spinRows(t,y,command); ...
        y(4); gravity*(liftRatioOf(y(2)) - 1)];
    groundODE = @(t,y,command)[spinRows(t,y,command); 0; 0];
    liftOffEvent = @(t,y)liftOffCrossing(t, y, hoverFrequency);
    touchDownEvent = @touchDownCrossing;
    
    % Resting on the pad unless the robot is already outrunning gravity or
    % has been given an upward kick.
    onGround = liftRatioOf(stateAtStart(2)) < 1 && ...
        initialVerticalVelocity <= 0;
    startedOnGround = onGround;
    if onGround
        % The pad absorbs any downward initial velocity.
        stateAtStart(4) = 0;
    end
    
    totalTime = edges(end);
    outputStep = max(1E-4, totalTime/15000);
    solverMaxStep = min(2E-4, min(durations)/20);
    solverMaxStep = max(solverMaxStep, 1E-7);
    baseOptions = odeset('RelTol',1E-8, ...
        'AbsTol',[1E-9 1E-7 1E-10 1E-9], ...
        'MaxStep',solverMaxStep);
    
    allTime = [];
    allState = [];
    allCommand = [];
    allOnGround = false(0,1);
    liftOffTimes = [];
    touchDownTimes = [];
    contactSwitchesExhausted = false;
    MAX_CONTACT_SWITCHES = 500;
    
    for segmentIndex = 1:numel(durations)
        t0 = edges(segmentIndex);
        t1 = edges(segmentIndex+1);
    
        commandFunction = @(t)evaluateSegmentFrequency( ...
            t, t0, durations(segmentIndex), types{segmentIndex}, ...
            starts(segmentIndex), ends(segmentIndex), shapes(segmentIndex));
    
        % A segment is integrated in one piece per contact mode: the solver
        % stops at each liftoff or landing, the state is handed over, and
        % integration resumes under the other right-hand side. That keeps
        % the discontinuity on an exact time rather than smeared across an
        % output step.
        timeCursor = t0;
        switchCount = 0;
        while timeCursor < t1 - 1E-12
            detectContact = ~contactSwitchesExhausted;
            if onGround
                modeODE = @(t,y)groundODE(t,y,commandFunction);
                modeEvent = liftOffEvent;
            else
                modeODE = @(t,y)flightODE(t,y,commandFunction);
                modeEvent = touchDownEvent;
            end
            if detectContact
                options = odeset(baseOptions, 'Events', modeEvent);
            else
                options = baseOptions;
            end
    
            pointCount = max(2, ceil((t1-timeCursor)/outputStep)+1);
            evaluationTime = linspace(timeCursor,t1,pointCount)';
            [timeChunk, stateChunk, eventTime, eventState] = ...
                ode45(modeODE, evaluationTime, stateAtStart, options);
    
            if ~isempty(eventTime)
                % ode45 stops at the terminal event; make sure the exact
                % switching point lands in the output.
                eventTime = eventTime(end);
                eventState = eventState(end,:);
                if timeChunk(end) < eventTime
                    timeChunk(end+1,1) = eventTime; %#ok<AGROW>
                    stateChunk(end+1,:) = eventState; %#ok<AGROW>
                end
            end
    
            commandChunk = arrayfun(commandFunction, timeChunk);
            groundChunk = repmat(onGround, size(timeChunk));
            chunkEndState = stateChunk(end,:)';
            chunkEndTime = timeChunk(end);
    
            % The first sample of every chunk repeats the last sample of the
            % previous one, at the same instant.
            if ~isempty(allTime)
                timeChunk(1) = [];
                stateChunk(1,:) = [];
                commandChunk(1) = [];
                groundChunk(1) = [];
            end
            allTime = [allTime; timeChunk]; %#ok<AGROW>
            allState = [allState; stateChunk]; %#ok<AGROW>
            allCommand = [allCommand; commandChunk]; %#ok<AGROW>
            allOnGround = [allOnGround; groundChunk]; %#ok<AGROW>
    
            stateAtStart = chunkEndState;
            if isempty(eventTime)
                timeCursor = t1;
                continue;
            end
    
            timeCursor = chunkEndTime;
            if onGround
                liftOffTimes(end+1,1) = timeCursor; %#ok<AGROW>
                onGround = false;
            else
                touchDownTimes(end+1,1) = timeCursor; %#ok<AGROW>
                % Inelastic landing: the pad takes the impact, so the
                % downward velocity is lost rather than bounced.
                stateAtStart(3) = 0;
                stateAtStart(4) = 0;
                % It only stays down if lift is still short of weight;
                % otherwise the pad is already unloaded and the
                % upward-crossing liftoff event would never fire.
                onGround = liftRatioOf(stateAtStart(2)) <= 1;
            end
    
            % A command that hovers on f_hover can make the robot hop
            % indefinitely. Stop chasing every hop rather than stalling the
            % caller, and say so in the summary.
            switchCount = switchCount + 1;
            if switchCount >= MAX_CONTACT_SWITCHES
                contactSwitchesExhausted = true;
            end
        end
    end
    
    delta = allState(:,1);
    fRobot = allState(:,2)/(2*pi);
    verticalDisplacement = allState(:,3);
    verticalVelocity = allState(:,4);
    
    deltaWrapped = atan2(sin(delta),cos(delta));
    frequencyError = fRobot - allCommand;
    coilGainTrace = arrayfun(@(f)coilGain(f, drive), allCommand);
    tauMaxTrace = drive.momentSI*drive.bMaxSI*coilGainTrace;
    fieldTrace = drive.bMaxSI*coilGainTrace;
    tauMagnetic = tauMaxTrace.*sin(delta);
    tauDrag = -k_drag*fRobot.*abs(fRobot);
    angularAcceleration = (tauMagnetic + tauDrag)/I_robot;
    
    liftToWeight = (fRobot./hoverFrequency).^2;
    
    % Every vertical force as a fraction of weight, so they add up on one axis:
    %   lift - 1 + normal = z_ddot/g
    verticalAcceleration = gravity.*(liftToWeight - 1);
    % Standing on the pad the net vertical force is zero, not negative: the
    % normal force makes up the shortfall.
    verticalAcceleration(allOnGround) = 0;
    normalForceRatio = zeros(size(liftToWeight));
    normalForceRatio(allOnGround) = ...
        max(0, 1 - liftToWeight(allOnGround));
    timeOnGround = trapz(allTime, double(allOnGround));
    
    if isempty(liftOffTimes)
        liftOffTime = NaN;
    else
        liftOffTime = liftOffTimes(1);
    end
    
    % Linearization about the hover trim of this same four-state model.
    % Because tau_max depends on the command, the input also reaches
    % omega_dot directly: B(2) is nonzero.
    [A, B, deltaHover, hoverReachable] = hoverStateSpace(drive, hoverFrequency);
    
    result = struct( ...
        'segments',                 {segments}, ...
        'parameters',               parameters, ...
        'drive',                    drive, ...
        'constants',                constants, ...
        'types',                    {types}, ...
        'starts',                   starts, ...
        'ends',                     ends, ...
        'durations',                durations, ...
        'shapes',                   shapes, ...
        'edges',                    edges, ...
        'totalTime',                totalTime, ...
        'time',                     allTime, ...
        'command',                  allCommand, ...
        'state',                    allState, ...
        'onGround',                 allOnGround, ...
        'delta',                    delta, ...
        'deltaWrapped',             deltaWrapped, ...
        'fRobot',                   fRobot, ...
        'verticalDisplacement',     verticalDisplacement, ...
        'verticalVelocity',         verticalVelocity, ...
        'verticalAcceleration',     verticalAcceleration, ...
        'frequencyError',           frequencyError, ...
        'coilGainTrace',            coilGainTrace, ...
        'tauMaxTrace',              tauMaxTrace, ...
        'fieldTrace',               fieldTrace, ...
        'tauMagnetic',              tauMagnetic, ...
        'tauDrag',                  tauDrag, ...
        'angularAcceleration',      angularAcceleration, ...
        'liftToWeight',             liftToWeight, ...
        'normalForceRatio',         normalForceRatio, ...
        'timeOnGround',             timeOnGround, ...
        'liftOffTimes',             liftOffTimes, ...
        'touchDownTimes',           touchDownTimes, ...
        'liftOffTime',              liftOffTime, ...
        'startedOnGround',          startedOnGround, ...
        'contactSwitchesExhausted', contactSwitchesExhausted, ...
        'fInitial',                 fInitial, ...
        'deltaInitial',             deltaInitial, ...
        'initialAboveCeiling',      initialAboveCeiling, ...
        'gainAtHover',              gainAtHover, ...
        'tauMagMaxAtResonance',     tauMagMaxAtResonance, ...
        'tauMagMaxAtHover',         tauMagMaxAtHover, ...
        'effectiveMargin',          effectiveMargin, ...
        'stepOutCeiling',           stepOutCeiling, ...
        'A',                        A, ...
        'B',                        B, ...
        'deltaHover',               deltaHover, ...
        'hoverReachable',           hoverReachable, ...
        'rmsError',                 sqrt(mean(frequencyError.^2)), ...
        'maximumAbsoluteError',     max(abs(frequencyError)), ...
        'netPhaseTurns',            (delta(end)-delta(1))/(2*pi));
end

function parameters = fillDefaults(parameters, defaults)
    if isempty(parameters)
        parameters = struct();
    end
    names = fieldnames(defaults);
    for index = 1:numel(names)
        if ~isfield(parameters, names{index}) || isempty(parameters.(names{index}))
            parameters.(names{index}) = defaults.(names{index});
        end
    end
end

%% ---- Command shaping ---------------------------------------------------
function data = normalizeSegments(data, autoChain)
    % Coerce a raw table to canonical 5-column form: unknown type names become
    % 'Hold', columns 2-5 become numeric, Hold rows get End = Start and
    % Order = 0, Polynomial orders round to integers >= 1. With autoChain,
    % each row's Start is overwritten by the previous row's End.
    if nargin < 2
        autoChain = true;
    end
    if isempty(data)
        return;
    end
    if size(data,2) ~= 5
        error('The signal table must contain exactly five columns.');
    end
    
    validTypes = {'Hold','Polynomial','Exponential'};
    for row = 1:size(data,1)
        try
            typeName = char(data{row,1});
        catch
            typeName = 'Hold';
        end
        match = find(strcmpi(typeName,validTypes),1);
        if isempty(match)
            match = 1;
        end
        typeName = validTypes{match};
        data{row,1} = typeName;
    
        for column = 2:5
            data{row,column} = numericValue(data{row,column});
        end
    
        if autoChain && row > 1
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
        values = cellfun(@numericValue, data(row,2:5));
        if any(~isfinite(values))
            error('Row %d contains a non-finite number.',row);
        end
        if values(1) < 0 || values(2) < 0
            error('Row %d contains a negative frequency.',row);
        end
        if values(3) <= 0
            error('Row %d must have a positive duration.',row);
        end
        if strcmpi(data{row,1},'Polynomial') && ...
                (values(4) < 1 || abs(values(4)-round(values(4))) > 1E-10)
            error(['Row %d Polynomial order must be a positive ' ...
                'integer: 1 = linear, 2 = quadratic, 3 = cubic.'],row);
        end
    end
end

function [types,starts,ends,durations,shapes,edges] = parseSegments(data)
    % Split the table into per-column vectors. edges holds the n+1 boundary
    % times, [0; cumsum(durations)], so segment k spans edges(k) to edges(k+1).
    types = cellfun(@char, data(:,1), 'UniformOutput', false);
    starts = cellfun(@numericValue, data(:,2));
    ends = cellfun(@numericValue, data(:,3));
    durations = cellfun(@numericValue, data(:,4));
    shapes = cellfun(@numericValue, data(:,5));
    edges = [0; cumsum(durations)];
end

function frequency = evaluateSegmentFrequency(time,segmentStartTime, ...
        duration,typeName,startFrequency,endFrequency,shape)
    % Frequency within one segment, blending Start to End by segment shape.
    % s = (time - segmentStartTime)/duration, clamped to [0,1]:
    %   Hold         blend = 0, holds Start
    %   Polynomial   blend = s^shape, shape >= 1 (1 is a linear ramp)
    %   Exponential  blend = (1-exp(-shape*s))/(1-exp(-shape)), linear as shape -> 0
    % time may be a vector.
    s = (time-segmentStartTime)./duration;
    s = min(max(s,0),1);
    switch lower(typeName)
        case 'hold'
            blend = zeros(size(s));
        case 'polynomial'
            % Order 1 reproduces a plain linear ramp.
            blend = s.^max(1,round(shape));
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

%% ---- Text summary ------------------------------------------------------
function lines = summaryLines(result)
    % Full run report: scalars, the hover state-space block and the per-Hold
    % pass/fail table, as one cell array of lines.
    lines = [alignRows(resultRows(result)); {''}; ...
        stateSpaceSummary(result); {''}; holdSummary(result)];
end

function rows = resultRows(result)
    drive = result.drive;
    rows = {
        'Segments', sprintf('%d', numel(result.durations))
        'Total command time', sprintf('%.6f s', result.totalTime)
        'Initial command frequency', sprintf('%.6f Hz', result.fInitial)
        'Final command frequency', sprintf('%.6f Hz', result.command(end))
        'B_max (at resonance)', sprintf('%.6f mT', 1E3*drive.bMaxSI)
        'Magnet moment m', sprintf('%.6f mA m^2', 1E3*drive.momentSI)
        'Coil L / C / R', sprintf('%.4f mH / %.2f uF / %.4f ohm', ...
            1E3*drive.inductanceSI, 1E6*drive.capacitanceSI, drive.resistanceSI)
        'LC resonance / Q', sprintf('%.4f Hz / %.4f', ...
            drive.resonanceHz, drive.qualityFactor)
        'tau_max at resonance', sprintf('%.6e N m  (m*B_max)', result.tauMagMaxAtResonance)
        'RLC gain at f_hover', sprintf('%.6f  -> B = %.6f mT', ...
            result.gainAtHover, 1E3*drive.bMaxSI*result.gainAtHover)
        'tau_max at f_hover', sprintf('%.6e N m', result.tauMagMaxAtHover)
        'Effective margin at hover', sprintf('%.6f  (derived, was an input)%s', ...
            result.effectiveMargin, ternary(result.effectiveMargin < 1, ...
                '  <-- BELOW 1: cannot hold hover', ''))
        'Step-out ceiling', ternary(isfinite(result.stepOutCeiling), ...
            sprintf('%.6f Hz  (max f with tau_max(f) >= k_drag*f^2)', result.stepOutCeiling), ...
            'none -- tau_max(f) never reaches the drag it must beat')
        'RLC gain range over run', sprintf('%.6f to %.6f', ...
            min(result.coilGainTrace), max(result.coilGainTrace))
        'Field range over run', sprintf('%.6f to %.6f mT', ...
            1E3*min(result.fieldTrace), 1E3*max(result.fieldTrace))
        'Peak |tau_mag| used', sprintf('%.6e N m  (%.2f%% of the local ceiling)', ...
            max(abs(result.tauMagnetic)), 100*max(abs(sin(result.delta))))
        'Initial phase lag', sprintf('%.6f deg%s', result.deltaInitial*180/pi, ...
            ternary(result.initialAboveCeiling, ...
                '  <-- SATURATED: first Start is above the step-out ceiling', ''))
        'Final robot frequency', sprintf('%.6f Hz', result.fRobot(end))
        'Robot frequency range', sprintf('%.6f to %.6f Hz', ...
            min(result.fRobot), max(result.fRobot))
        'Hover frequency', sprintf('%.6f Hz', result.parameters.hoverFrequency)
        'Contact at t = 0', ternary(result.startedOnGround, ...
            'on the pad, N = W - L holds z at 0', ...
            'airborne (lift already beats weight, or launched upward)')
        'Initial normal force', sprintf('%.6f of weight', result.normalForceRatio(1))
        'Liftoff', ternary(isnan(result.liftOffTime), ...
            sprintf('never -- f_robot peaks at %.6f Hz, short of f_hover', ...
                max(result.fRobot)), ...
            sprintf('t = %.6f s  (f_robot = f_hover, N -> 0)', result.liftOffTime))
        'Landings after liftoff', sprintf('%d%s', numel(result.touchDownTimes), ...
            ternary(result.contactSwitchesExhausted, ...
                '  <-- CONTACT CHATTER: switch cap hit, later hops not resolved', ''))
        'Time resting on the pad', sprintf('%.6f s  (%.2f%% of the run)', ...
            result.timeOnGround, 100*result.timeOnGround/result.totalTime)
        'Final vertical acceleration', sprintf('%.6f m/s^2%s', ...
            result.verticalAcceleration(end), ...
            ternary(result.onGround(end), '  (held at 0 by the pad)', ''))
        'Initial vertical velocity', sprintf('%.6f m/s', ...
            result.parameters.initialVerticalVelocity)
        'Final vertical velocity', sprintf('%.6f m/s', result.verticalVelocity(end))
        'Final vertical displacement', sprintf('%.6f mm', ...
            1000*result.verticalDisplacement(end))
        'Vertical displacement range', sprintf('%.6f to %.6f mm', ...
            1000*min(result.verticalDisplacement), 1000*max(result.verticalDisplacement))
        'Vertical acceleration range', sprintf('%.6f to %.6f m/s^2', ...
            min(result.verticalAcceleration), max(result.verticalAcceleration))
        'Final tracking error', sprintf('%.6f Hz', result.frequencyError(end))
        'RMS tracking error', sprintf('%.6f Hz', result.rmsError)
        'Maximum |tracking error|', sprintf('%.6f Hz', result.maximumAbsoluteError)
        'Angular acceleration range', sprintf('%.6e to %.6e rad/s^2', ...
            min(result.angularAcceleration), max(result.angularAcceleration))
        'Net relative phase turns', sprintf('%.4f', result.netPhaseTurns)
        };
end

function lines = stateSpaceSummary(result)
    A = result.A;
    B = result.B;
    lines = {'State-space model linearized about the hover trim:'};
    lines{end+1,1} = '  x = [delta; omega; z; z_dot],  u = f_field (Hz)';
    if result.hoverReachable
        lines{end+1,1} = sprintf('  Hover phase lag delta_h : %.6f deg', ...
            result.deltaHover*180/pi);
    else
        lines{end+1,1} = ['  Hover phase lag delta_h : NO TRIM -- tau_max ' ...
            'cannot sustain f_hover (step-out)'];
    end
    lines{end+1,1} = '  A =';
    for row = 1:4
        lines{end+1,1} = sprintf('    %14.6f %14.6f %14.6f %14.6f', A(row,:)); %#ok<AGROW>
    end
    lines{end+1,1} = sprintf('  B'' = [%.6f  %.6f  %.6f  %.6f]', B);
    eigenvalues = eig(A);
    for index = 1:numel(eigenvalues)
        lines{end+1,1} = sprintf('  eig %d = %+0.6f %+0.6fi', ...
            index, real(eigenvalues(index)), imag(eigenvalues(index))); %#ok<AGROW>
    end
    oscillatory = eigenvalues(abs(imag(eigenvalues)) > 1E-9);
    if ~isempty(oscillatory)
        lambda = oscillatory(1);
        lines{end+1,1} = sprintf( ...
            '  Phase-lock mode: %.4f Hz, zeta = %.4f', ...
            abs(imag(lambda))/(2*pi), -real(lambda)/abs(lambda));
    end
    % The block-triangular factorization of notes Sec. 8.3: the two complex
    % roots above are the phase lock, and the vertical block contributes a
    % double root at the origin -- undamped, with no open-loop height feedback.
    lines{end+1,1} = '  Heave block   : undamped double integrator, both roots at 0';
end

function lines = holdSummary(result)
    types = result.types;
    edges = result.edges;
    time = result.time;
    errorSignal = result.frequencyError;
    tolerance = result.parameters.frequencyTolerance;
    
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
                lines{end+1,1} = sprintf( ... %#ok<AGROW>
                    '  Hold %d (segment %d): max tail error %.5f Hz, %s', ...
                    holdCounter,segmentIndex,tailMaxError, ...
                    ternary(tailMaxError <= tolerance, ...
                        'within tolerance','outside tolerance'));
            end
        end
    end
    if holdCounter == 0
        lines{end+1,1} = '  No Hold segments.';
    end
end

function lines = alignRows(rows)
    % Pad every label to the widest one so the colons line up in the
    % monospace result box, whatever labels get added later.
    format = sprintf('%%-%ds: %%s', max(cellfun(@numel, rows(:,1))) + 1);
    lines = cell(size(rows,1),1);
    for index = 1:size(rows,1)
        lines{index} = sprintf(format, rows{index,1}, rows{index,2});
    end
end

%% ---- Small helpers -----------------------------------------------------
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

function out = ternary(condition, whenTrue, whenFalse)
    if condition
        out = whenTrue;
    else
        out = whenFalse;
    end
end
