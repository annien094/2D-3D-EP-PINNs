function T = pdeCoefficientsToDouble(S, U)
%pdeCoefficientsToDouble Convert symbolic pde coefficients to PDE Toolbox format
%   T = pdeCoefficientsToDouble(S) converts the symbolic objects of the struct S
%   into doubles or function handles. The output is a struct T which can then be
%   used to define the coefficients of a PDE model by calling
%   specifyCoefficients(MODEL, T). This latter step requires the PDE Toolbox.
%
%   Example:
%       syms u(x, y)
%       S = pdeCoefficients(laplacian(u) - x, [u], 'Symbolic', true);
%       T = pdeCoefficientsToDouble(S);
%       model = createpde();
%       geometryFromEdges(model,@lshapeg);
%       specifyCoefficients(model, T)
%
%       creates a pde model for the equation laplacian(u) == x.
%       This example requires the PDE Toolbox.
%
%   See also pdeCoefficients, CREATEPDE, pde.PDEModel.

%   Copyright 2020 The MathWorks, Inc.

arguments
    S struct
    U sym = sym([])
end

fields = string(fieldnames(S));
% Fieldnames used by specifyCoefficients
expectedFields = ["a", "c", "m", "d", "f"];
if ~isempty(setdiff(fields, expectedFields))
    error(message('symbolic:pdeCoefficients:InvalidField'));
end

% Interpret missing fields as zero
for fld = setdiff(expectedFields, fields)
    S.(fld) = sym(0);
end


% get number of equations ... from dimension of m, d, a, or f
nequations = max([size(S.m, 1), size(S.d, 1), size(S.a, 1), numel(S.f)]);

% get number of dependent variables (= number of equations)
nvars = nequations;

% get dimen (2 or 3)
allSyms = struct2cell(S);
dimen = 3;
% for k = 1:numel(allSyms)
%     if has(allSyms{k}, sym('z'))
%         dimen = 3;
%     end
% end

% determine U
if isempty(U)
    for k=1:numel(allSyms)
        U = union(U, findSymType(allSyms{k}, "symfun"));
    end
    if isempty(U)
        % naming does not matter
        U = sym('u', [1, nvars]);
    end
end

if numel(formula(U)) ~= nvars
    error(message('symbolic:pdeCoefficients:SystemMustBeSquare'));
end

% Reshape C to a column (blockwise)
if nvars > 1
    blocks = mat2cell(S.c, repmat(dimen, 1, nequations), repmat(dimen, 1, nvars));
    blocks = cellfun(@(B) B(:), blocks, 'UniformOutput', false);
    S.c = vertcat(blocks{:});
end

% convert coefficients into the form needed by specifyCoefficients
T = struct;
for fld = expectedFields
    if ~isempty(setdiff(symvar(S.(fld)), [sym('x'), sym('y'), sym('z'), sym('t')]))
        error(message('symbolic:pdeCoefficients:InvalidVariableInCoefficient'));
    end
    T.(fld) = makeCoefficient(S.(fld), U, dimen);
end

% warn for purely algebraic equations
if isequal(T.m, 0) && isequal(T.d, 0) && isequal(T.c, 0)
    if ~hasSymType(S.a, "diff") && ~hasSymType(S.f, "diff")
        warning(message('symbolic:pdeCoefficients:Algebraic'));
    end
end

end % pdeCoefficientsToDouble


function fun = makeCoefficient(D, U, dimen)

% D must be reshaped to a column vector!
D = D(:);

% constant coefficient: just convert to double
if isempty(symvar(D))
    fun = double(D);
    if ~any(fun)
        % make scalar. This is sometimes needed:
        % e.g., either m or d should be 0.
        fun = 0;
    end
    return;
end

% For the system of N pdes, in N (dependent) variables u1, ..., un,
% we create 3N symbols ukdx, ukdy, ukdz representing diff(uk, x) etc.
% corresponding to the k-th entry of state.ux, state.uy, state.uz
N = numel(formula(U));
u = sym('u', [1 N]);
if N==1
    dudx = sym('u1x');
    dudy = sym('u1y');
    dudz = sym('u1z');
else
    dudx = sym('u%dx', [1, N]);
    dudy = sym('u%dy', [1, N]);
    dudz = sym('u%dz', [1, N]);
end

%     if dimen == 3
%         renamedD = subs(D, [diff(U, sym('x')), diff(U, sym('y')), diff(U, sym('z'))], [dudx, dudy, dudz]);
%     else
%         renamedD = subs(D, [diff(U, sym('x')), diff(U, sym('y'))], [dudx, dudy]);
%     end
renamedD = subs(D, U, u');


% Generate a function handle from symbolic object D.
% Since matlabFunction does not handle column vectors well,
% we make FD return a cell array of component functions
if dimen == 2
    FD = arrayfun(@(row) matlabFunction(row, 'Vars', [sym('t'), sym('x'), sym('y'), u, dudx, dudy]), renamedD, 'UniformOutput', false);
else
    % dimen == 3
    FD = arrayfun(@(row) matlabFunction(row, 'Vars', [sym('t'), sym('x'), sym('y'), sym('z'), u, dudx, dudy, dudz]), renamedD, 'UniformOutput', false);
end


% return value:
    function dmatrix = coefficientFunction(region, state)

        % With the special argument 'show', the coefficient function
        % displays itself symbolically.
        if strcmpi(region, 'show')
            dmatrix = D;
            return;
        end

        % extract rows from u, ux, uy
        su = num2cell(state.u, 2);
        sux = num2cell(state.ux, 2);
        suy = num2cell(state.uy, 2);

        if dimen == 3
            suz = num2cell(state.uz, 2);
        end

        % It can happen that the time field is missing for time-constant
        % problems.
        if ~isfield(state, 'time')
            state.time = 0;
        end

        % Apply each element of FD to vectors of length equal to the number of points in
        % region. To expand any scalars, we add zero.
        zer = zeros(size(region.x));
        if dimen == 2
            dcell = cellfun(@(rowfunc) zer + rowfunc(state.time, region.x, region.y, su{:}, sux{:}, suy{:}), FD, 'UniformOutput', false);
        else
            dcell = cellfun(@(rowfunc) zer + rowfunc(state.time, region.x, region.y, region.z, su{:}, sux{:}, suy{:}, suz{:}), FD, 'UniformOutput', false);
        end
        dmatrix = vertcat(dcell{:});

    end

fun = @coefficientFunction;

end %makeCoefficient

