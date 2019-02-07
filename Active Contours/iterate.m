function [newX, newY] = iterate(Ainv, x, y, Eext, gamma, kappa)

% Get fx and fy
[fx, fy] = gradient(Eext);

% Iterate
newX = gamma*x - kappa*interp2(fx,x,y);
newY = gamma*y - kappa*interp2(fy,x,y);

%calculating the new position of snake
newX = newX * Ainv;
newY = newY * Ainv;

% Clamp to image size
[imgHeight, imgWidth] = size(Eext);
newX(newX>imgWidth) = imgWidth;
newY(newY>imgHeight) = imgHeight;
newX(newX<1) = 1;
newY(newY<1) = 1;

end

