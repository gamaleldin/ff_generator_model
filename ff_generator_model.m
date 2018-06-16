function [R] = ff_generator_model(all_times, targetOn, goCue, moveStarts, numNeus, numConds, xP, yP, colors)
%% Method for generator responses from the generator model.
% all_times: vector with time points
% targetOn: target onset time
% goCue: go cue time
% moveStarts: movement start time
% numNeus: number of neruons in the data.
% numConds: number of conditions in the data.
% xP: kinematics x position.
% yP: kinematics y position.
% colors: matrix of size conditions by 3 contain RGB values for different
%   conditions.

tprepStarts = targetOn+150;
tprepEnds = targetOn+150+300;
tmoveStarts = moveStarts-50;
tmoveEnds = moveStarts+250;
all_times = [all_times(:)' all_times(end)]; % extend time
prepTimes = tprepStarts:10:tprepEnds;
moveTimes = tmoveStarts:10:tmoveEnds;
rn = 0.014;%0.01; %
lkM = -0.002; %-0.0025;%
lkP =  -0.9;%-1;%
ff =  0.04; % 0.03;%

RNN = [lkM -rn;
       rn lkM];
FF = ff*ones(size(RNN,1),2);
Mprep = zeros(4);Mprep(1:2,1:2) = lkP*eye(2);
MFF = zeros(4);MFF(3:4,1:2) = ff*ones(2);
Mmove = zeros(4);Mmove(3:4,3:4) = RNN;
B = [eye(2); zeros(2)];

xP = xP(end, :) - mean(xP(1,:));
yP = yP(end, :) - mean(yP(1,:));
xP = xP-mean(xP);
yP = yP-mean(yP);
sc = max(normVects([xP; yP]));
xP = xP./sc;
yP = yP./sc;
Q = 50*(randn(numNeus,4));
Qoffset = 50*rand(numNeus, 1);
inTemplate1 = zeros(length(all_times),1);
inTemplate2 = zeros(length(all_times),1);
inTemplate1(all_times>=targetOn+100) = 1;
% inTemplate2(all_times>=goCue+100) = 1;

% inTemplate = gaussFilter(20, inTemplate1)-gaussFilter(50, inTemplate2);
inTemplate = gaussFilter(20, inTemplate1);%-gaussFilter(50, inTemplate2);
time_activate = goCue+120; %goCue+100;%
time_inactivate = goCue+120; %goCue+150;%
goSignal = zeros(1, length(all_times));
goSignal(all_times>=time_activate) = 1;
goSignal = gaussFilter(50, goSignal);
% offPrepSignal  = (1-goSignal);

offPrepSignal = ones(1, length(all_times));
offPrepSignal(all_times>=time_inactivate) = 0;
offPrepSignal = gaussFilter(50, offPrepSignal); % 50

for c = 1:numConds
a = [xP(c); yP(c)];
% a = a./norm(a);
uc = bsxfun(@times, repmat(inTemplate(:).', 2, 1), a); 

x = [0; 0; 0; 0 ];
for i = 2:length(all_times)
%     x(:,i) = x(:,i-1)+ M*x(:,i-1);
    x(:,i) = x(:,i-1)+ Mprep*x(:,i-1)+Mmove*x(:,i-1)+goSignal(i)*MFF*x(:,i-1)+offPrepSignal(i)*B*uc(:, i-1);
% %     figure(1);
% %     subplot(121);
% %     plot(x(1,1:i),x(2,1:i), 'color', colors(c,:))
% %     hold on
% %     plot(x(1,i),x(2,i),'*','color',colors(c,:))
% %     hold off
% %     axis([-1 1 -1 1])
% %     axis square
% %     title(num2str(i))
% %     subplot(122);    
% %     plot(x(3,1:i),x(4,1:i), 'color', colors(c,:))
% %     hold on
% %     plot(x(3,i),x(4,i),'*','color',colors(c,:))
% %     hold off
% %     axis([-1 1 -1 1])
% %     axis square
% %     title(num2str(i))
% %     pause(0.01)
end
% X(:,:,c) = gaussFilter(20,[repmat(x(:,1), 1, 350) x]);
X(:,:,c) = x(:,1:10:end);
R(:,:,c) = (Q*x(:,1:10:end))';%+(Qoffset*goSignal(1:10:end))';
end

all_times = all_times(1:10:end);


%%
hf(1) = figure;
for c = 1:numConds
    subplot(321);
    hold on
    plot(X(1,:,c),X(2,:, c), 'color', colors(c,:))
    axis(1.5*[-1 1 -1 1])
    axis square
    subplot(322); 
    hold on
    plot(X(3,:,c),X(4,:, c), 'color', colors(c,:))
    axis(1.5*[-1 1 -1 1])
    axis square
    title(num2str(i))
    
    subplot(312); 
    hold on
    plot(all_times,[X(1,:,c)' X(2,:, c)'], 'color', colors(c,:))
    xlim([all_times(1) all_times(end)])
    
    subplot(313); 
    hold on
    plot(all_times,[X(3,:,c)' X(4,:, c)'], 'color', colors(c,:))
    xlim([all_times(1) all_times(end)])

%     axis(1.5*[-1 1 -1 1])
%     axis square
%     title(num2str(i))
    
%     subplot(224); 
%     hold on
%     plot(1:length(X(3,:,c)), [X(3,:,c)' X(4,:, c)'], 'color', colors(c,:))
%     axis(1.5*[-1 1 -1 1])
%     axis square
    title(num2str(i))
end

subplot(312); 
hold on
plot(targetOn*[1 1],[-1 1], 'k-')
plot(goCue*[1 1],[-1 1], 'k-')
plot(moveStarts*[1 1],[-1 1],  'k-')
    
subplot(313); 
hold on
plot(targetOn*[1 1],[-1 1],  'k-')
plot(goCue*[1 1],[-1 1],  'k-')
plot(moveStarts*[1 1],[-1 1], 'k-')



%%
varPrep_t = var(squeeze(X(1,:,:)),[],2)+var(squeeze(X(2,:,:)),[],2);
varMove_t = var(squeeze(X(3,:,:)),[],2)+var(squeeze(X(4,:,:)),[],2);


axLim = [all_times(1)-20 all_times(end) -0.15 3.1];
hf(60) = figure;
set(gca,'visible', 'off');
set(hf(60), 'color', [1 1 1]);
axis(axLim);
hold on
set(hf(60),'position',[100 100 300 200])
axisParams.axisLabel = 'Time (ms)';
axisParams.axisOrientation = 'h';
axisParams.tickLocations = [all_times(1) all_times(end)];
axisParams.longTicks = 0;
axisParams.fontSize = 16;
axisParams.axisOffset = -0.1;
AxisMMC(all_times(1), all_times(end), axisParams);
axisParams.axisLabel = 'normalized var (a. u.)';
axisParams.axisOrientation = 'v';
axisParams.tickLocations = [0 1 2 3];
axisParams.longTicks = 0;
axisParams.fontSize = 16;
axisParams.axisOffset = -10;
AxisMMC(0, 3, axisParams);



plot(targetOn,-0.05,'wo', 'markerfacecolor','r','markersize',8)
plot(moveStarts,-0.05,'wo', 'markerfacecolor',[0 0.5 0],'markersize',8)
plot([tprepStarts tprepEnds], -0.025*[1 1], 'color',[0.5 0.5 0.5], 'linewidth', 3)
plot([tmoveStarts tmoveEnds], -0.025*[1 1] ,'color',[0.5 0.5 0.5], 'linewidth', 3)

plot(all_times, varPrep_t, 'color', 'r')
hold on
plot(all_times, varMove_t, 'color', [0 0.5 0])
plot(time_activate, 1, '.', 'color', [0 0.5 0])
plot(time_inactivate, 1-0.1, '.', 'color', 'r')

%%
for i = 1:9
Ri = reshape(R(:, randi(numNeus), :), [], 8);

hf(2) = figure;
set(hf(2), 'color', [1 1 1]);

hold on
for c = 1:numConds
plot(Ri(:,c), 'color', colors(c,:))
end
end
%%
tprepStarts = targetOn+150;
tprepEnds = targetOn+150+300;
tmoveStarts = moveStarts-50;
tmoveEnds = moveStarts+250;
prepTimes = tprepStarts:10:tprepEnds;
moveTimes = tmoveStarts:10:tmoveEnds;
prepIndices = (all_times>=prepTimes(1) & all_times<=prepTimes(end))';
prepMask = repmat(prepIndices(:), numConds,1);
moveIndices = (all_times>=moveTimes(1) & all_times<=moveTimes(end))';
moveMask = repmat(moveIndices(:), numConds,1);

RN = reshape(permute(R, [1 3 2]), [], numNeus);
RN = bsxfun(@times, RN, 1./(range(RN)+5));

P = reshape(permute(RN(prepIndices, :, :), [1 3 2]), [], numNeus);
M = reshape(permute(RN(moveIndices, :, :), [1 3 2]), [], numNeus);



[prepPCs, ~, ~] = pcaBYsvd(P);  % apply PCA to the analyzed times
[movePCs,~, ~] = pcaBYsvd(M);  % apply PCA to the analyzed times

dim = 2;
prepPCs = prepPCs(:, 1:dim);
movePCs = movePCs(:, 1:dim);

P = bsxfun(@minus, P, mean(P));
M = bsxfun(@minus, M, mean(M));

P_prep = P*prepPCs;
P_move = P*movePCs;

M_prep = M*prepPCs;
M_move = M*movePCs;


%% 
DataStruct(1).A = P;
DataStruct(1).dim = 2;
DataStruct(2).A = M;
DataStruct(2).dim = 2;
[QSubspaces] = getSubspaces(DataStruct);
prepPCsOrth = QSubspaces(1).Q;
movePCsOrth = QSubspaces(2).Q;
%% link
R_prep = bsxfun(@minus, RN, mean(RN))*prepPCsOrth;
R_move = bsxfun(@minus, RN, mean(RN))*movePCsOrth;
% mskCV1 = repmat(all_times(:)>=all_times(300) & all_times(:)<all_times(301), numConds, 1);
% mskCV2 = repmat(all_times(:)>=all_times(400) & all_times(:)<all_times(401), numConds, 1);
mskCV1 = repmat(all_times(:)>=prepTimes(end-1) & all_times(:)<=prepTimes(end), numConds, 1);
mskCV2 = repmat(all_times(:)>=moveTimes(round(length(moveTimes)/2)) & all_times(:)<moveTimes(round(length(moveTimes)/2)+1), numConds, 1);

[prepMoveLinkAnalysis] = prepMoveLink(R_prep(mskCV1, :), R_move(mskCV2, :), numConds, 0);

R2Train = prepMoveLinkAnalysis.Training.R2;
R2TrainShfl = prepMoveLinkAnalysis.Training.R2Shfl;
R2Test = prepMoveLinkAnalysis.Testing.R2;
R2TestShfl = prepMoveLinkAnalysis.Testing.R2Shfl;
pValsTrain = sigTest(R2Train, R2TrainShfl, 'upper', 'gamma');
pValsTest = sigTest(R2Test, R2TestShfl, 'upper', 'gauss');

hf(24) = figure;
hold on
set(hf(24), 'color', [1 1 1]);
set(gca, 'xtick', 1:2, 'xlim', [0.5 2.5], 'ytick', 0:0.5:1, 'ylim', [-0.4 1.35])
set(gca, 'xtickLabel', {'Training','Testing'})
ylabel('R^2')
h = bar([1 2], [R2Train median(R2TrainShfl);
R2Test median(R2TestShfl)]);
set(h(1),'facecolor','k','barwidth',0.95,'edgecolor','none')
set(h(2),'facecolor',[0.5 0.5 0.5],'barwidth',0.95,'edgecolor','none')
R2mtx = [R2TrainShfl(:) R2TestShfl(:)];
pctpts = prctile(R2mtx , [5 50 95], 1); 
dot_pt = pctpts(2, :);
topBar = pctpts(3,:)-dot_pt;
botBar = -(pctpts(1,:)-dot_pt);
plot([1 2]+0.14 , dot_pt , 'k.', 'markersize',20);
he= errorbar( [1 2]+0.14 , dot_pt , zeros(size(botBar)) ,topBar, 'k.', 'linewidth',2);
% errorbar_tick(he,10);
sigstar({1+[-0.14 0.14], 2+[-0.14 0.14]},[pValsTrain, pValsTest])
legend(h, 'Data', 'Shuffled')
set(gca,'plotboxaspectratio',[3 5 1])
ti = get(gca,'TightInset');
set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
legend boxoff




%%
% % % R1_prep = reshape(R_prep(:,1), [], numConds);
% % % R1_move = reshape(R_move(:,1), [], numConds);
% % % R2_move = reshape(R_move(:,2), [], numConds);
theta = 0;
rotMtx = [cos(theta/180*pi) -sin(theta/180*pi); sin(theta/180*pi) cos(theta/180*pi)];
Xp = reshape(permute(X([1 2], :, :), [2 3 1]), [],2);
R1_prep = reshape(bsxfun(@minus, Xp, mean(Xp))*rotMtx(:,1), [], numConds);
R1_move = reshape(permute(X(3, :, :), [2 3 1]), [], numConds);
R2_move = reshape(permute(X(4, :, :), [2 3 1]), [], numConds);
ix = 1:numConds;
R1_prep = R1_prep-mean(R1_prep(:));
R1_move = R1_move - mean(R1_move(:));
R2_move = R2_move - mean(R2_move(:));

% [~, ix] = sort(R1_prep(1, :), 'descend');
mskTrans = all_times>=moveStarts-200 & all_times<= moveStarts+10;
mskMove = all_times>=moveStarts & all_times<= moveStarts+50;
A1 = R1_prep(mskTrans,ix);%A1 = A1(1:10:end, :);
A2 = R1_move(mskTrans,ix);%A2 = A2(1:10:end, :);
A3 = R2_move(mskTrans,ix);%A3 = A3(1:10:end, :);

A11 = R1_prep(mskMove,ix);%A11 = A11(1:10:end, :);
A22 = R1_move(mskMove,ix);%A22 = A22(1:10:end, :);
A33 = R2_move(mskMove,ix);%A33 = A33(1:10:end, :);
hf(22) = figure;
set(hf(22), 'color', [1 1 1]);
plot3(0,0,0,'k.', 'markersize', 6)
hold on
for c = 1:size(A1,2); 
    plot3(A2(:,c), A3(:,c), A1(:,c), 'color',colors(c,:),'linewidth',0.5, 'linestyle', ':')
end

for c = 1:size(A1,2); 
    plot3(A2(1,c), A3(1,c), A1(1,c), '*', 'color',colors(c,:),'markerSize',10,'markerfacecolor',colors(c,:),'linewidth',1.5)
end

for c = 1:size(A1,2); 
    plot3(A22(:,c), A33(:,c), A11(:,c), 'color',colors(c,:),'linewidth',0.5, 'linestyle', '-')
    plot3(A22(1,c),A33(1,c), A11(1,c), 'k.', 'markerSize',12)
    arrow3D([A22(end-1,c) A33(end-1,c) A11(end-1,c)], [A22(end,c) A33(end,c) A11(end,c)], [], 18, [-1 1 -1 1 -1 1], colors(c,:), [0 0 0]);
end

set(gca,'plotboxaspectratio',[4 4 5])
set(gca, 'xticklabel', [], 'yticklabel', [], 'zticklabel', [])
grid on
xlim([-1.1000    1.6000])
ylim([ -1.8000    1.3500])
zlim(1.1*[ -1 1])
xlabel('Move. dim. 1')
ylabel('Move. dim. 2')
zlabel('Prep. dim. 1')
view(-60, 20)


hf(23) = figure;
set(hf(23), 'color', [1 1 1]);
plot3(0,0,0,'k.', 'markersize', 6)
hold on
for c = 1:size(A1,2); 
%     plot3(A2(:,c), A3(:,c), A1(:,c), 'color',colors(c,:),'linewidth',0.5, 'linestyle', ':')
end

for c = 1:size(A1,2); 
    plot3(A2(1,c), A3(1,c), A1(1,c), '*', 'color',colors(c,:),'markerSize',10,'markerfacecolor',colors(c,:),'linewidth',1.5)
end

for c = 1:size(A1,2); 
    plot3(A22(:,c), A33(:,c), A11(:,c), 'color',colors(c,:),'linewidth',0.5, 'linestyle', '-')
    plot3(A22(1,c),A33(1,c), A11(1,c), 'k.', 'markerSize',12)
    arrow3D([A22(end-1,c) A33(end-1,c) A11(end-1,c)], [A22(end,c) A33(end,c) A11(end,c)], [], 18, [-1 1 -1 1 -1 1], colors(c,:), [0 0 0]);
end

set(gca,'plotboxaspectratio',[4 4 5])
set(gca, 'xticklabel', [], 'yticklabel', [], 'zticklabel', [])
grid on
xlim([-1.1000    1.6000])
ylim([ -1.8000    1.3500])
zlim(1.1*[ -1 1])
xlabel('Move. dim. 1')
ylabel('Move. dim. 2')
zlabel('Prep. dim. 1')
view(-60, 20)
axis vis3d
view(-90, 90)
set(gca,'yaxislocation','right')
ti = get(gca,'TightInset');

%%
% % alignIx_P = sum(var(P_move))./sum(var(P_prep))
% % alignIx_M = sum(var(M_prep))./sum(var(M_move))

alignIx_P = alignIx(bsxfun(@minus,P, mean(P)).', movePCs)
alignIx_M = alignIx(bsxfun(@minus,M, mean(M)).', prepPCs)
%%
end
