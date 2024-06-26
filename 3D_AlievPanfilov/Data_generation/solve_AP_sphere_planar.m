%% Set up
clear all
close all
%%
% figure
name='planar';

% Model Parameters
% aa = 0.01;
% k = 8.0;
% mu1 = 0.2;
% mu2 = 0.3;
% epsi = 0.002;
% b  = 0.15;
% D = 0.1;

% Symbolic PDE Definition
syms W(x,y,z,t) V(x,y,z,t)
syms aa k mu1 mu2 epsi b D
fV = k*V*(V-aa)*(V-1)+V*W; 
fW = (epsi + mu1*W/(V+mu2))*(-W-k*V*(V-b-1));
pdeeq = [diff(V,t) - D*laplacian(V,[x,y,z]) + fV; ...
    diff(W,t) - fW];

symCoeffs = pdeCoefficients(pdeeq,[V;W],'Symbolic',true);
symVars = [aa k mu1 mu2 epsi b D];
symCoeffs = subs(symCoeffs, symVars, [0.01 8.0 0.2 0.3 0.002 0.15 0.10]);
coeffs = pdeCoefficientsToDoubleMV(symCoeffs);
coeffs.f = zeros(2,1,'double'); % make this explicit to avoid crashing solver!!!

%% FEM model setup
APmodel=createpde(2);
% walls=niftiread('LV.nii');
% fac=2;
% walls=imresize3(walls,fac,'nearest');
% me=isosurface(walls,0.5);

% me=myreadmeshvtk2('walls3MC.vtk');
% gm=geometryFromMesh(APmodel,me.vertices,me.faces);

% hollow sphere
R=[10 12];
gm1 = multisphere(R,'Void',[true,false]);
gm2 = multicuboid(1,1,1);
translate(gm2,[0 0 R(1)*1.03]);
gm = addCell(gm1, gm2);


% add vertex to use as initial condition
% pt=0.04;
% pt2=sqrt(R(2)^2-pt^2*2);
% vv = addVertex(gm,'Coordinates',[0 0 R(2); pt pt pt2; ...
%     -pt pt pt2; pt -pt pt2; -pt -pt pt2; 0.2889 0.0869 11.9802; ...
%     -0.2687 0.1552 11.9797; -0.3004 -0.1303 11.9785]);

% [x,y]=meshgrid(h*cos(phi),h*sin(phi));

% Approach below leads to complaints that the vertices are too far away,
% for all h values tested...
% h=0.06;
% phi=0:0.1:2*pi;
% vv1=[h*cos(phi);h*sin(phi);R(2)-h*ones(size(phi))]';
% vv2=[h*cos(phi);h*sin(phi);R(1)-h*ones(size(phi))]';
% vv=cat(1,vv1,vv2);
% vv=addVertex(gm,'Coordinates',vv);

APmodel.Geometry=gm;
pdegplot(APmodel,'CellLabels','on','VertexLabels','on',...
    'FaceLabels','on','FaceAlpha',0.8);
hold all

% APmodel.Geometry=gm2;
% pdegplot(APmodel,'CellLabels','on','VertexLabels','on',...
%     'FaceLabels','on','FaceAlpha',0.8);

% plot3(vv(:,1),vv(:,2),vv(:,3),'bo','MarkerSize',20)

mesh=generateMesh(APmodel,'Hmax',1.4);
pdemesh(APmodel); 
axis equal

specifyCoefficients(APmodel,'m',coeffs.m,'d',coeffs.d, ...
    'c',coeffs.c,'a',coeffs.a,'f',coeffs.f);
applyBoundaryCondition(APmodel,'neumann','face',1:2,'g',[0;0],'q',[0;0]);
% ufun=@(location,~)location.z>8;
setInitialConditions(APmodel,[0;0]);
setInitialConditions(APmodel,[1;0],'Cell',2);

%% Solve!
tini=0;
tfin=100;
dt=0.5;
dt_disp=2;
tlist = tini:dt:tfin;

APmodel.SolverOptions.RelativeTolerance = 1.0e-3; 
APmodel.SolverOptions.AbsoluteTolerance = 1.0e-4;

APmodel.SolverOptions.ReportStatistics='on';
R = solvepde(APmodel,tlist);
u = R.NodalSolution;

figure
nod=1;
plot(tlist,squeeze(u(nod,1,:)))
hold all
plot(tlist,squeeze(u(nod,2,:)))
grid on
xlabel('Time (s)')
legend('V','W')

figure; 
tshow=30;
plot(squeeze(u(:,1,tshow)),'.')
hold all
plot(squeeze(u(:,2,tshow)),'.')
grid on
xlabel('Node')
legend('V','W')

% ylabel 'u_{heart} (AU)'

for t=tini+dt:dt:tfin
    if ~mod(t,dt_disp)
        mywritemeshvtktetra(mesh,squeeze(u(:,1,t)),[name num2str(t,'%.0f') '.vtk']);
    end 
end
%%
 V = u(:,1,:);
 V = reshape(V,[],length(tlist));
 V = V';

 W = u(:,2,:);
 W = reshape(W,[],length(tlist));
 W = W';

 x = mesh.Nodes(1,:);
 x = x';
 y = mesh.Nodes(2,:);
 y = y';
 z = mesh.Nodes(3,:);
 z = z';
 t = 1:1:201;
 t = t';
    %%
 save([name '.mat'],'V','W','x','y','z','t','-v7')

 %%
 %% Mesh information
numElements = size(mesh.Elements,2);
numNodes = size(mesh.Nodes,2);
edgeLengths = sqrt(sum((mesh.Nodes(:,mesh.Elements(1,:)) - mesh.Nodes(:,mesh.Elements(2,:))).^2,1));
avgEdgeLength = mean(edgeLengths);

disp(['Number of elements: ', num2str(numElements)]);
disp(['Number of nodes: ', num2str(numNodes)]);
disp(['Average edge length: ', num2str(avgEdgeLength)]);
