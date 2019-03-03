clear all;

% Parameters (play around with different images and different parameters)
% Random generate parameters and save results to file
% shape = 'star/'; f= fopen(strcat(shape,'parameters.txt'),'a');
% for count=1:30
% alpha = rand(); beta = rand(); gamma = rand(); kappa = rand()*0.5; Wline = rand(); Wedge = rand(); Wterm = rand();
% fprintf(f, strcat('play round ',num2str(count),'\n')); fprintf(f, strcat('alpha=',num2str(alpha),'\n')); fprintf(f, strcat('beta=',num2str(beta),'\n'));fprintf(f, strcat('gamma=',num2str(gamma),'\n'));fprintf(f, strcat('kappa=',num2str(kappa),'\n'));fprintf(f, strcat('Wline=',num2str(Wline),'\n'));fprintf(f, strcat('Wedge=',num2str(Wedge),'\n'));fprintf(f, strcat('Wterm=',num2str(Wterm),'\n\n'));

N = 200;
alpha = 0.55; 
beta = 0.98; 
gamma = 0.55;
kappa = 0.15;
Wline = 0.62;
Wedge = 0.36;
Wterm = 0.75;
sigma = 0.5;

% Load image
I = imread('images/circle.jpg');
% I = imread('images/brain.png');
% I = imread('images/square.jpg');
% I = imread('images/star.png');
% I = imread('images/shape.png');
% I = imread('images/dental.png');
% I = imread('images/vase.tif');

if (ndims(I) == 3)
    I = rgb2gray(I);
end

% Initialize the snake
[x, y] = initializeSnake(I);

% Calculate external energy
I_smooth = double(imgaussfilt(I, sigma));
Eext = getExternalEnergy(I_smooth,Wline,Wedge,Wterm);

% Calculate matrix A^-1 for the iteration
Ainv = getInternalEnergy(size(x,2), alpha, beta, gamma);

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

% save fig and close file 
% saveas(gcf, strcat(shape,'final_',num2str(count),'.png'));
% close(gcf)
% end
% fclose(f);
