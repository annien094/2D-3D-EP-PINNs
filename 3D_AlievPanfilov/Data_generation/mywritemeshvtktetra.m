function outp=mywritemeshvtktetra(FEMmsh,u,filename)
% writes msh as .vtk file to be opened in paraview
% msh is the name of the Matlab structure with fields: 
% faces, vertices (as given by Matlab's isosurface)
% filename is the name of the vtk file to be created
% assumes mesh elements are all triangles 
% Marta, 29/01/2021

msh.vertices=FEMmsh.Nodes;
msh.faces=FEMmsh.Elements;
msh.solution=u;

if length(u)~=length(msh.vertices)
    error('Problem with dimensions of u and/or FEM mesh.')
end

% open .vtk file 
outp=0;
fil=fopen(filename,'wt');
if fil==-1
    disp('Problem opening file.')
    outp=1;
end

nfac=length(msh.faces);
nver=length(msh.vertices);

if size(msh.faces,2)==10
    msh.faces=msh.faces';
end
if size(msh.vertices,2)==3
    msh.vertices=msh.vertices';
end
if any(msh.faces(:)==0)
    msh.faces=msh.faces+1;
end
if any(msh.faces>nver)
    error('Faces reference non-existent vertices')
end

extras='';

% write vertices
fprintf(fil,'# vtk DataFile Version 2.0\nTest\nASCII\n');
fprintf(fil,'DATASET UNSTRUCTURED_GRID\nPOINTS %d FLOAT\n',nver);
fprintf(fil,'%f %f %f\n',msh.vertices);

% write faces (starting from 0)
fprintf(fil,'\nCELLS %d %d\n',nfac,11*nfac);
fprintf(fil,'10 %d %d %d %d %d %d %d %d %d %d\n',msh.faces-1);

fprintf(fil,'\nCELL_TYPES %d\n',nfac);
for i=1:nfac
    fprintf(fil,'24\n');
end

% write one additional field if it exists
% for now assumes all are scalar float and only have one element
flds=fieldnames(msh);
if length(flds)>2
    for i=3 % fields 1 and 2 are vertices and faces %:length(flds)
        if length(getfield(msh,flds{i}))==nver
            fprintf(fil,'POINT_DATA %d\nSCALARS %s float 1\nLOOKUP_TABLE default\n',nver,flds{i});
            fprintf(fil,'%f ',getfield(msh,flds{i})); 
        else
            error('Additional field should have the same dimensions as the number of vertices.');
        end
    end
end

fclose(fil);
if ~outp
    disp(['Written ' filename '.']);
end