function [Eext] = getExternalEnergy(I,Wline,Wedge,Wterm)

% Eline
Eline = I;

% Eedge
[Gmag,~]= imgradient(I,'sobel');
Eedge = Gmag;

% Eterm
m1 = [-1 1];
m2 = [-1;1];
m3 = [1 -2 1];
m4 = [1;-2;1];
m5 = [1 -1;-1 1];

Ix = conv2(I,m1,'same');
Iy = conv2(I,m2,'same');
Ixx = conv2(I,m3,'same');
Iyy = conv2(I,m4,'same');
Ixy = conv2(I,m5,'same');

Eterm = (Iyy.*Ix.^2 -2*Ixy.*Ix.*Iy + Ixx.*Iy.^2)./((1+Ix.^2 + Iy.^2).^(3/2));

% Eext
Eext = Wline*Eline + Wedge*Eedge + Wterm * Eterm;

end

