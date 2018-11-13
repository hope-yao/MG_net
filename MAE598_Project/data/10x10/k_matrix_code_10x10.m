
tic()   
    nx = 10;              % # of elements along  X- direction.
    ny = 10;              % # of elements along  Y- direction. 
    lx=0.05;              % length in x direction
    ly=0.05;              % length in y direction
        D = 16 * eye(2);    % W/m-K
    mesh = uniform_mesh(nx,ny,lx,ly);
    qbar = 10000;        % Heat flux (leaving domain at bottom).
    Tbar = 0;          % Prescribed temperature.
    
    qpts = [-1 1 1 -1; -1 -1 1 1]/sqrt(3);   
    
    % Compute conductivity matrix and nodal heat source fluxes.
    K = spalloc(mesh.num_nodes, mesh.num_nodes, 9*mesh.num_nodes);
 %     K = zeros(mesh.num_nodes, mesh.num_nodes);
    f = zeros(mesh.num_nodes, 1);
    for c = mesh.connectivity
        xe = mesh.x(:,c);
        Ke = zeros(4);
        for q = qpts
            [N, dNdp] = shape(q);
            J = xe * dNdp;
            B = dNdp/J;
            Ke = Ke + B * D * B' * det(J);
            x = xe(1,:) * N;
%             f(c) = f(c) + N * 1000*x * det(J);
        end
        K(c,c) = K(c,c) + Ke;
    end
    toc()
      right_side = find(mesh.x(1,:) == lx);
     [~, order] = sort(mesh.x(1, right_side));
    right_side = right_side(order);
     
    % Apply heat flux on right edge of domain.

    edge_conn = [right_side(1:end-1); right_side(2:end)];
    for c = edge_conn
        xe = mesh.x(:, c);
        le = norm(xe(:,2) - xe(:,1));
        N = [0.5; 0.5];
        f(c) = f(c) + N * qbar * le; % sign changed from - to +
    end
     
    % Apply essential boundary condition on left edge of domain.
    left_side = find(mesh.x(1,:) == 0);
    
    for i = left_side
%         K(i,:) = 0.0;
%         K(i,i) = 1.0;
        f(i) = Tbar;
        K(i,:) = [];
        K(:,i) = [];
        f(i,:)=[];
    end
    % Solve discrete system.
    toc()
    % deleting rows
%     f(10,1)=f(20,1);
    d = K\f;
    toc()
%     % Plot temperature contours.
%     clf;
%     p.vertices = mesh.x';
%     p.faces = mesh.connectivity';
%     p.facecolor = 'interp';
%     p.facevertexcdata = d;    
%     patch(p)   
%     colorbar
%     
% %     Plot heat fluxes at element centers.
%     qq = [];
%     xx = [];    
%     for c = mesh.connectivity
%         xe = mesh.x(:,c);
%         de = d(c)';
%         [N,dNdp] = shape([0;0]);
%         J = xe * dNdp;
%         B = dNdp / J;
%         qq(end+1, :) = -de * B * D;
%         xx(end+1, :) = xe * N;
%     end
%     hold on;
%     quiver(xx(:,1), xx(:,2), qq(:,1), qq(:,2), 0.50, 'color', 'k');
%     rank_of_k=rank(K)
    K_forceboundary_elements10x10 = K;
    f_forceboundary_elements10x10 = f;
    x0_elements10x10 = zeros(length(K),1);
    save('K_forceboundary_elements10x10.mat', 'K_forceboundary_elements10x10')
    save('f_forceboundary_elements10x10.mat', 'f_forceboundary_elements10x10')
    save('x0_elements10x10.mat', 'x0_elements10x10')
  toc()




% Creates a unifom mesh of ex by ey elements
% with length of lx and ly in the x and y directions.
% The origin (lower left) can be optionally specified as x0, y0.
function [m] = uniform_mesh(ex, ey, lx, ly, x0, y0)
    % if origin is not specified, then set it to zero.
    if nargin < 5, x0 = 0; end
    if nargin < 6, y0 = 0; end       
        
    m.num_nodes = (ex+1)*(ey+1);
    % Nodal reference coordinates.
    m.x    = zeros(2, m.num_nodes);   
    for j=1:ey+1
        for i=1:ex+1
            m.x(:,i+(j-1)*(ex+1)) = [(i-1)*lx/ex + x0; (j-1)*ly/ey + y0];
        end
    end
    m.num_elements = ex*ey;
    m.connectivity = zeros(4, m.num_elements);
    for j=1:ey
        for i=1:ex            
            % first node in element
            n0 = i+(j-1)*(ex+1);            
            m.connectivity(:,i+(j-1)*ex) = [n0; n0+1; n0+ex+2; n0+1+ex];            
        end
    end
        
    % 4 point Gaussian quadrature rule.
    m.quad_points = [-1, 1, 1,-1;
                     -1,-1, 1, 1] / sqrt(3);
    m.quad_points = [m.quad_points; 1,1,1,1];

    % Call as mesh.draw() to draw this mesh.
    m.draw =       @() plot_mesh(m);
    % Call as mesh.draw_nodal(f) to plot mesh colored by nodal value f.
    m.draw_nodal = @(f) plot_nodal(m,f);
    
    m.shape = @shape;
end

function [] = plot_mesh(m)
    p.vertices = m.x';
    p.faces = m.connectivity';
    p.facecolor = 'none';
    patch(p);
end

function[] = plot_nodal(m, f)
    p.vertices = m.x';
    p.faces = m.connectivity';
    p.facecolor = 'interp';
    p.facevertexcdata = f;
    patch(p);    
end

function [N, dNdp] = shape(p)
    N = 0.25*[(1-p(1)).*(1-p(2));
              (1+p(1)).*(1-p(2));
              (1+p(1)).*(1+p(2));
              (1-p(1)).*(1+p(2))];
          
    dNdp = 0.25*[-(1-p(2)), -(1-p(1));
                  (1-p(2)), -(1+p(1));
                  (1+p(2)),  (1+p(1));
                 -(1+p(2)),  (1-p(1))];
end