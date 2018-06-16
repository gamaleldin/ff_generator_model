% useage:
%   h = arrow3D(prevPoint, point, nextPoint)
%       where each of the three points is [x,y,z].  The three points are used to set the orientation
%       of the arrow so that it looks good when plotted on top of a curve, and roughly follows the
%       tangent of the curvature.
%
%       If you don't want to use the third point (e.g., if there isn't one) then use:
%   h = arrowMMC(prevPoint, point, [])
%
%       You can also specify these additional arguments (must be done in order, but you don't have to use all)
%   h = arrowMMC(prevPoint, point, nextPoint, sizeArrow, axisRange, faceColor, edgeColor)
%       Most importantly, axisRange tells arrowMMC how big to make the arrow, and allows it to have
%       appropriately scaled proportions.  If you don't supply this, arrowMMC will get it using
%       'gca'.  But, if the axes are then later rescaled (e.g. due to more points plotting) things
%       will get wonky.  So either supply 'axisRange', or make sure the axes don't change after
%       plotting the arrow.
%
%       A reasonable starting size for the arrow is 6.
%
function h = arrow3D(prevPoint, point, nextPoint, sizeArrow, axisRange, faceColor, edgeColor)


roughScale = 0.004;  % setting this empirically so that 'sizeArrow' works roughly like 'markerSize'

xVals1 = [0 -1.5 4.5  -1.5 0 ];
yVals1 = [0  2    0   -2   0  ];
zVals1 = [-2 0    0    0   -2 ]/2;

xVals2 = [0 -1.5 4.5  -1.5 0 ];
yVals2 = [0  2    0   -2   0  ];
zVals2 = [2 0    0    0   2 ]/2;



%% do some parsing of the inputs, and use defaults if not provided
if isempty(nextPoint)
    nextPoint = point + point-prevPoint;
end

if nargin<4, sizeArrow = 6; end

if nargin<5
    xRange = range(get(gca,'XLim'));
    yRange = range(get(gca,'YLim'));
else
    xRange = range(axisRange(1:2));
    yRange = range(axisRange(3:4));
    zRange = range(axisRange(5:6));
end

if nargin<6, faceColor = [0 0 0]; end
if nargin<7, edgeColor = faceColor; end

%% do a bit of scaling
mxX = max(xVals1);
xVals1 = roughScale*sizeArrow * xVals1/mxX * xRange;
yVals1 = roughScale*sizeArrow * yVals1/mxX * yRange;
zVals1 = roughScale*sizeArrow * zVals1/mxX * zRange;
xVals2 = roughScale*sizeArrow * xVals2/mxX * xRange;
yVals2 = roughScale*sizeArrow * yVals2/mxX * yRange;
zVals2 = roughScale*sizeArrow * zVals2/mxX * zRange;
%% now do the rotation

vector = nextPoint - prevPoint;
thetaZ = atan2(vector(2),vector(1));
rotMZ = [cos(thetaZ) -sin(thetaZ) 0; sin(thetaZ), cos(thetaZ) 0; 0 0 1];
thetaY = atan2(vector(3),vector(1));
rotMY = [cos(thetaY) 0 sin(thetaY);0 1 0; -sin(thetaY) 0 cos(thetaY)];
thetaX = atan2(vector(3),vector(2));
rotMX = [1 0 0 ; 0 cos(thetaX) -sin(thetaX); 0 sin(thetaX) cos(thetaX)];
rotM = rotMZ*rotMX;
% rotM  = [1;0;0]'\vector(:)';rotM = rotM./norm(rotM,'fro')*sqrt(3);
newVals1 = rotM*[xVals1; yVals1; zVals1];
xVals1 = newVals1(1,:);
yVals1 = newVals1(2,:);
zVals1 = newVals1(3,:);

newVals2 = rotM*[xVals2; yVals2; zVals2];
xVals2 = newVals2(1,:);
yVals2 = newVals2(2,:);
zVals2 = newVals2(3,:);
%% now plot
xVals1 = xVals1 + point(1);
yVals1 = yVals1 + point(2);
zVals1 = zVals1 + point(3);

xVals2 = xVals2 + point(1);
yVals2 = yVals2 + point(2);
zVals2 = zVals2 + point(3);

h = fill3(xVals1, yVals1, zVals1, faceColor,xVals2, yVals2, zVals2, faceColor);
set(h, 'edgeColor', edgeColor');




end
