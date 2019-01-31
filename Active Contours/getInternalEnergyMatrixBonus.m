function [Ainv] = getInternalEnergyMatrix(nPoints, alpha, beta, gamma)
A = zeros(nPoints,nPoints);
b = [(2*alpha + 6 *beta) -(alpha + 4*beta) beta];
brow = zeros(1,nPoints);
brow(1,1:3) = brow(1,1:3) + b;
brow(1,nPoints-1:nPoints) = brow(1,nPoints-1:nPoints) + [beta -(alpha + 4*beta)];
for i=1:nPoints
    A(i,:) = brow;
    brow = circshift(brow',1)';
end

[L, U] = lu(A + gamma .* eye(nPoints));
Ainv = inv(U) * inv(L); 
end

