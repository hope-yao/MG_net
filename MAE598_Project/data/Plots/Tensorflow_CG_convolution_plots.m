figure
title('Plot of Computaion Time Vs Dimensionality')
xlabel('Dimensionality (No. of Elements in one direction)') 
ylabel('Time in Sec') 
% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
grid on
hold on

% Tensorflow: Conjugate Gradient Method (Convolution) - Intel Core i5-8350U CPU
y1=[0.72087,47.44112,961.71353];% put the time in sec for line 1
x1=[10,100,1000];%no of elements line 1
plot (x1,y1,'-k','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method (Convolution) - WS CPU
y2=[0.99609,8.19984,75.82120];% put the time in sec for line 1
x2=[10,100,1000];%no of elements line 1
plot (x2,y2,'--k','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method (Convolution) - WS GPU
y3=[1.01424,8.34345,75.62940];% put the time in sec for line 1
x3=[10,100,1000];%no of elements line 1
plot (x3,y3,':k','LineWidth',1.5)
hold on

% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
% Replace the name in corresponding legend for respective method
legend({'Tensorflow: Conjugate Gradient Method (Convolution) - Intel Core i5-8350U CPU','Tensorflow: Conjugate Gradient Method (Convolution) - WS CPU','Tensorflow: Conjugate Gradient Method (Convolution) - WS GPU'},'Location','northwest')