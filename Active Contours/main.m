clear all;

% Parameters (play around with different images and different parameters)
N = 200;
alpha = 0.4;
beta = 0.2;
gamma = 0.5;
kappa = 0.15;
Wline = 0.5;
Wedge = 1.0;
Wterm = 0.5;
sigma = 0.5;

% alpha = 0.4;
% beta = 0.2;
% gamma = 1.0;
% kappa = 0.15;
% Wline = 0;
% Wedge = 0.5;
% Wterm = 0;
% sigma = 0.5;

% Load image
% I = imread('images/star.png');
I = imread('images/circle.jpg');
if (ndims(I) == 3)
    I = rgb2gray(I);
end

% Initialize the snake
[x, y] = initializeSnake(I);

% Calculate external energy
I_smooth = double(imgaussfilt(I, sigma));
Eext = getExternalEnergy(I_smooth,Wline,Wedge,Wterm);

% Calculate matrix A^-1 for the iteration
Ainv = getInternalEnergyMatrixBonus(size(x,2), alpha, beta, gamma);
disp(size(Ainv))
disp(size(x))
% Iterate and update positions
displaySteps = floor(N/10);

for i=1:N
    % Iterate
    [x,y] = iterate(Ainv, x, y, Eext, gamma, kappa);
    
    fprintf('x(1): %d, y(1): %d \n', x(1), y(1))
    
    % Plot intermediate result
    imshow(I); 
    hold on;
    plot([x x(1)], [y y(1)], 'r');
        
    % Display step
    if(mod(i,displaySteps)==0)
        fprintf('%d/%d iterations\n',i,N);
    end
    
    pause(0.0001)
end
 
if(displaySteps ~= N)
    fprintf('%d/%d iterations\n',N,N);
end