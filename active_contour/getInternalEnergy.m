function [Ainv] = getInternalEnergyMatrixBonus(nPoints, alpha, beta, gamma)

A = zeros(nPoints,nPoints);
temp = zeros(1,nPoints);

temp(1,1:3) = [(2*alpha + 6 *beta) -(alpha + 4*beta) (beta)];
temp(1,nPoints-1:nPoints) = [(beta) -(alpha + 4*beta)];

for i=1:nPoints
    A(i,:) = temp;
    temp = circshift(temp, 1, 2);
end
I = eye(nPoints);
Ainv = inv(A + gamma*I); 
%Ainv = inv(I-gamma*A); 

end

