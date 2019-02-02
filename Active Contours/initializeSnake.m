function [x, y] = initializeSnake(I)

% Show figure
figure, imshow(I);

% Get initial user selected points
[xInit, yInit] = getpts();

% Form a closed loop
P = transpose([xInit yInit]);  % 2 x n
numOfPts = size(P,2) + 1;
P(:,numOfPts) = P(:,1);  % copy the first point to end 

% Interpolate
k = 1:numOfPts;  %adding another axis k
stepSize = 0.1;
kstep = 1:stepSize:numOfPts;
Pstep = spline(k,P,kstep);

x = Pstep(1,:);
y = Pstep(2,:); 

% Clamp points to be inside of image
[imgHeight, imgWidth] = size(I);
x(x>imgWidth) = imgWidth;
y(y>imgHeight) = imgHeight;
x(x<1) = 1;
y(y<1) = 1;

hold on

plot(xInit,yInit,'ro',x,y,'b.');

end

