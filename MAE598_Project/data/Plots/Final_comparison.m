figure
title('Plot of Computaion Time Vs Dimensionality')
xlabel('Dimensionality (No. of Elements in one direction)') 
ylabel('Time in Sec') 
% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
grid on
hold on


% Tensorflow: Conjugate Gradient Method - WS CPU
y4=[0.72283,5.92201, 55.55385];% put the time in sec for line 1
x4=[10,100, 1000];%no of elements line 1
plot (x4,y4,'--c','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method - WS GPU
y5=[0.73323,5.92948, 54.82987];% put the time in sec for line 1
x5=[10,100, 1000];%no of elements line 1
plot (x5,y5,':c','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method (Convolution) - WS CPU
y7=[0.99609,8.19984,75.82120];% put the time in sec for line 1
x7=[10,100,1000];%no of elements line 1
plot (x7,y7,'--k','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method (Convolution) - WS GPU
y8=[1.01424,8.34345,75.62940];% put the time in sec for line 1
x8=[10,100,1000];%no of elements line 1
plot (x8,y8,':k','LineWidth',1.5)
hold on

% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
% Replace the name in corresponding legend for respective method
legend({'Tensorflow: Conjugate Gradient Method - WS CPU','Tensorflow: Conjugate Gradient Method - WS GPU','Tensorflow: Conjugate Gradient Method (Convolution) - WS CPU','Tensorflow: Conjugate Gradient Method (Convolution) - WS GPU'},'Location','northwest')