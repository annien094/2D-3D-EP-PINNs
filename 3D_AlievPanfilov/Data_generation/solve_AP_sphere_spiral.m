% Spiral wave in sphere
% Author: Marta Varela
% Edited by: Aditi Roy
% Edited by: Annie
% Date: 13/03/2024

% This edition is to merge Marta's new code with new larger pacing region in
% solveAP_sphesurf2.m, where we have a square of stimulus to initialise the waves,
% with Aditi's spiral wave generation code.

% Run time: ~1173.375556 seconds = ~20 min.

%% Set up
clear
close all

% Symbolic PDE Definition
syms W(x,y,z,t) V(x,y,z,t)
syms aa k mu1 mu2 epsi b D
fV = k*V*(V-aa)*(V-1)+V*W;
fW = (epsi + mu1*W/(V+mu2))*(-W-k*V*(V-b-1));
pdeeq = [diff(V,t) - D*laplacian(V,[x,y,z]) + fV; ...
    diff(W,t) - fW];

symCoeffs = pdeCoefficients(pdeeq,[V;W],'Symbolic',true);
symVars = [aa k mu1 mu2 epsi b D];
symCoeffs = subs(symCoeffs, symVars, [0.01 8.0 0.2 0.3 0.002 0.15 0.1]);
coeffs = pdeCoefficientsToDoubleMV(symCoeffs);
coeffs.f = zeros(2,1,'double'); % make this explicit to avoid crashing solver!!!

%% FEM model setup
tic
APmodel=createpde(2);

% hollow sphere
R=[10 12];
gm1 = multisphere(R,'Void',[true,false]);
gm2 = multicuboid(1,1,1);
translate(gm2,[0 0 R(1)*1.03]);
gm = addCell(gm1, gm2);

APmodel.Geometry=gm;
pdegplot(APmodel,'CellLabels','on','VertexLabels','on',...
    'FaceLabels','on','FaceAlpha',0.8);
hold all

mesh=generateMesh(APmodel,'Hmax',1.4);
pdemesh(APmodel); 
axis equal

specifyCoefficients(APmodel,'m',coeffs.m,'d',coeffs.d, ...
    'c',coeffs.c,'a',coeffs.a,'f',coeffs.f);
applyBoundaryCondition(APmodel,'neumann','face',1:2,'g',[0;0],'q',[0;0]);
% ufun=@(location,~)location.z>8;
setInitialConditions(APmodel,[0;0]);
setInitialConditions(APmodel,[1;0],'Cell',2);

% solve the PDE
Tstart = 0;
Tend = 100;
S2_time = 40;
dt=0.5;

% Part 1: First set of simulation before S2------------------------------
tini1=Tstart;
tfin1=S2_time;

tlist = tini1:dt:tfin1;

APmodel.SolverOptions.RelativeTolerance = 1.0e-3;
APmodel.SolverOptions.AbsoluteTolerance = 1.0e-4;
APmodel.SolverOptions.ReportStatistics='on';

R = solvepde(APmodel,tlist);

disp('DONE solving first part')

figure
subplot(1,2,1)
pdeplot3D(APmodel,"ColorMapData",R.NodalSolution(:,1,end));
colorbar;clim([0, 1]);
title(sprintf('time: %.2f',tfin1))

% altering the result by appling S2 to the solution
xval = R.Mesh.Nodes(1,:,:);
uval = R.NodalSolution(:,1,end);  % the solution of the first PDE variable V 
                                  % for all mesh nodes at the last time step
uval(xval>0)=0;% tried to apply in the middle % set 1 side of the sphere to 0

subplot(1,2,2)
pdeplot3D(APmodel,"ColorMapData",uval);
colorbar;clim([0, 1]);
title(sprintf('S2 at time: %.2f',tfin1))

% set the solution of the first part as inital conditio for the second part
u2 = R.NodalSolution(:,:,end);
u2(:,1) = uval;
newResults = createPDEResults(APmodel,u2(:));
setInitialConditions(APmodel,newResults);

% Part 2: second set of simulation after S2------------------------------
tini2=S2_time;
tfin2=Tend;

tlist = tini2:dt:tfin2;

results35 = solvepde(APmodel,tlist);
toc
disp('DONE solving second part')

% visualize the results over time
Toltal_time = tini1:dt:tfin2;
dimR = size(R.NodalSolution(:,1,:));
% concatinating all the results from first and second part together
solution = zeros(dimR(1),1,length(Toltal_time)); % Part 1 (tend is removed)
solution(:,1,1:dimR(3)-1)=R.NodalSolution(:,1,1:end-1); % Part 2
solution(:,1,dimR(3):end)=results35.NodalSolution(:,1,1:end);

flag_vis_v = 1;

if flag_vis_v==1
    visualize_voltage(APmodel, Toltal_time, solution);
end

%%
% save the solution to vtk format

% (Marta you may have to check this part of the code to see if saves
% correctly)

flag_save_file = 1;

u = results35.NodalSolution; % saving only the second part after S2
tst = tini2;
te = tfin2;

if flag_save_file == 1
    name='spiral_sphere_new';
    dt_disp=2;

    for t=tst+dt:dt:te
        if ~mod(t,dt_disp)
            mywritemeshvtktetra(mesh,squeeze(u(:,t)),[name num2str(t,'%.0f') '.vtk']);
        end
    end
    %%
    V = u(:,1,:);
    V = squeeze(V);
    V = V';

    W = u(:,2,:);
    W = squeeze(W);
    W = W';

    x = mesh.Nodes(1,:);
    x = x';

    y = mesh.Nodes(2,:);
    y = y';

    z = mesh.Nodes(3,:);
    z = z';

    t = 1:dt:tfin2-tini2+1;
    t = t';
    save([name '.mat'],'V','W','x','y','z','t','-v7')
end

