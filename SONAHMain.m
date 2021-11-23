%% EC EN 671 Application Project: 
% Statistically Optimized Near-field Acoustical Holography
% Logan Mathews
%
%% Initialize Variables and Packages

% Dependency on T-7a Core Files repository for t7ageo.m function
usePackage t-7a-core-files
% Dependency on numsourcemodel repository for doRayleighIntegration.m, 
% calcHypTanWavepacket.m.
usePackage numsourcemodel
% Dependency on t-7a-holography repository for AnalyticContinuation.m, 
% CalculateAMSONAH.m, CalculateAlphaMSONAH.m
usePackage t-7a-holography

f = 125; % Frequency
w = 2*pi*f; % Angular Frequency
pref = 20e-6; % Reference Pressure

% Set Source Parameters
source.type = 'htan';
source.parameters.b1 = 4.4;
source.parameters.b2 = 14.2;
source.parameters.g1 = 13;
source.parameters.g2 = 2.7;
source.parameters.Uc = 478;
source.height = 64.9/12 * .3048;
source.conditions.rho = 1.21;
source.conditions.c = 343;

% Array coordinates from T-7A measurement
[x,y,z] = T7Ageo('Imaging');
x = x(1:end-3);
y = y(1:end-3);
z = z(1:end-3);
coord = [x,y,z];

dx = .05; % Monopole spacing
xm = 0:dx:50; % Locations of monopoles

% Numerical Source Simulation from Whiting et al. (2020)
[P] = doRayleighIntegration(w,coord,xm,source);

% Determine Reconstruction Grid
u = linspace(-15,50); % length of cone
v = linspace(-5,40); % Revolution grid
[X,Y] = meshgrid(u,v); % Mesh
Z = zeros(size(X));

% Coordinates of reconstruction locations
recon = [reshape(X,[],1) reshape(Y,[],1) reshape(Z,[],1)]; 

[~,n] = size(P); % Get size of complex pressure matrix
CSM = 1/n*(P*P'); % Create cross-spectral matrix from complex pressures
[U,S,~] = svd(CSM); % Use SVD to separate into partial fields
PF = U.*sqrt(diag(S)).'; % Results of SVD construct the partial fields
PF = PF(:,1); % Use only the first partial field

%% Interpolation (to raise spatial Nyquist frequency)
xq = x; % initialize
newdx = 0.1; % New dx to interpolate to
k = 1; % Indexing
iter = 0; % Iterator
while k < length(xq) % Loop through grid
    iter = iter + 1; % Incriment iterator
    dxlocal = xq(k+1) - xq(k); % Find local spacing
    factor = ceil(dxlocal/newdx); 
    if factor > 1 % If local dx is greater than desired spacing
        insert = linspace(xq(k),xq(k+1),factor-1+2).'; 
        insert = insert(2:end-1);
        xq = [xq(1:k); insert; xq(k+1:end)]; % Insert an interpolation point
    end
    k = k + factor;
end
mi = interp1(x,[y z],xq); % Interpolate y and z with new x grid
mi = [xq mi];
magnitudes = interp1(x,abs(PF),xq); % Interpolate magnitude over new grid
angles = interp1(x,unwrap(angle(PF)),xq); % Interpolate phase over grid
PF = magnitudes.*exp(1i.*angles); % Construct interpolated pressures
coord = mi; % Update coordinates to new grid
%% Perform Aperture Extension (To avoid wraparound error)
% Use Analytic Continuation Method
iterations = 200; % How many times to iterate
extender = 400; % How many points to extend aperture by
r = 1; % Order of Tukey window
[PF,coord] = AnalyticContinuation(PF,coord,extender,r,iterations,0);

%% Calculate A matrix of wavefunctions
[A,kx,kr] = CalculateAMSONAH(f,coord,0,0); % Find A, matrix of measurement wavefunctions

%% Compute regularized inverse of A^H*A using SVD and modified Tichonov filter

[V, Sigma, U] = csvd(A'*A); % SVD of A^H*A
numK = sum(kr > 0); % number of radial wavenumbers that explode at large r
cutoff = min(round(numK*2 + 1), length(Sigma));
Sigma(cutoff:end) = 0; % Zeroing out values that will blow up

alphaARA = logspace(-3,10,100);

J = zeros(1,length(alphaARA)); % Initialize cost function to zeros

parfor n = 1:length(alphaARA)
    % Filter Function from Williams (2001) Eq. (57)
    F1alpha1 = alphaARA(n) ./ (alphaARA(n) + Sigma.^2 .*...
    ((alphaARA(n) + Sigma.^2) / alphaARA(n)).^2);
    % Cost function from Williams (2001) Eq. (58), also Eq. (5.12) in Wall Diss.
    J(n) = norm(diag(F1alpha1)*U'*PF)^2 / sum(F1alpha1)^2;
end

% Minimize Cost Function
[~,index] = min(J); % Minimum of cost function
alphaMin = alphaARA(index); % Find value of alpha at cost function min
% Compute regularization parameter alpha by finding the minimum value on a
% bounded range
ra = fminbnd('modgcvfun', 0.01*alphaMin, 100*alphaMin,...
optimset('Display', 'off'), Sigma, U, PF); 
% Use a Modified Tichonov Filter to achieve the regularized inverse of A^H*A
G = diag(Sigma);
% Compute modified Tichonov filter (high-pass) (Eq. (5.12) in Wall Diss.)
F1a = diag(ra ./ (ra + Sigma.^2 .* ((ra + Sigma.^2) / ra).^2));
% Calculate regularized inverse (Eq. (5.10) in Wall Diss.)
Raha = V * inv(ra * F1a.^2 + G'*G) * G' * V'; 

%% Compute coefficients for the chosen wavefunctions using results

c = PF.'*Raha*A'; % Find min-norm coefficient vector, c

% Find alpha, matrix of reconstruction wavefunctions
alpha = CalculateAlphaMSONAH(f,recon,kx,kr,0,0); 

pr = c*alpha; % Reconstructed pressures

SPLr = 10*log10(sum(abs(pr).^2,1)/pref^2); % Reconstructed SPL

%% Analytical Data (From wavepacket model, Rayleigh integral evaluated at 
% "reconstruction" locations.
[P] = doRayleighIntegration(w,recon,xm,source); % Analytic P values
SPL = 10*log10((abs(P).^2)/pref^2); % Analytic SPL values
%%
plotStyle('StandardStyle','custom','AspectRatio','standard',...
    'PlotSize','medium','WhichMonitor',2,'FontStyle','modern',...
    'Orientation','portrait')
[m,n] = size(X);

upperLim = ceil(max(max(SPLr),max(SPL))); % Find maximum SPL for setting caxis
numColors = 30; % Set the number of colors and number of dB dynamic range
% to achieve 1 color increment per dB.
colvect = P_hot('Flip',false,'NumColors',numColors); % Create colormap

figure
tiledlayout(2,1,'TileSpacing','compact','Padding','compact')
nexttile
fig1 = pcolor(X,Y,reshape(SPL,m,n)); % Plot conical surface with SPL
fig1.FaceLighting = 'none';
axis equal
title('Analytical')
ylabel('y, m')
shading interp
colormap(colvect); % Set colormap
caxis([upperLim-numColors, upperLim])
hold on
plotT7AModel('Imaged',false); % Plot T-7A model with imaged model
hold off


nexttile
fig2 = pcolor(X,Y,reshape(SPLr,m,n)); % Plot conical surface with SPL
fig2.FaceLighting = 'none';
axis equal
title('SONAH Reconstruction')
xlabel('x, m')
ylabel('y, m')
shading interp
colormap(colvect); % Set colormap
caxis([upperLim-numColors, upperLim])
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'SPL (dB re 20 \muPa)';
hold on
plotT7AModel('Imaged',false); % Plot T-7A model with imaged model
hold off