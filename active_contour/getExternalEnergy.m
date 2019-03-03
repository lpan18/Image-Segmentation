function [Eext] = getExternalEnergy(I,Wline,Wedge,Wterm)

% Eline
Eline = I;

% Eedge
[Gmag,~]= imgradient(I,'sobel');
Eedge = Gmag;

% Eterm
d1 = [-1 1];
d2 = [-1;1];
d3 = [1 -2 1];
d4 = [1;-2;1];
d5 = [1 -1;-1 1];

Ix = conv2(I,d1,'same');
Iy = conv2(I,d2,'same');
Ixx = conv2(I,d3,'same');
Iyy = conv2(I,d4,'same');
Ixy = conv2(I,d5,'same');

Eterm = (Iyy.*Ix.^2 -2*Ixy.*Ix.*Iy + Ixx.*Iy.^2)./((1+Ix.^2 + Iy.^2).^(3/2));

% Eext
Eext = Wline*Eline + Wedge*Eedge + Wterm * Eterm;

end

