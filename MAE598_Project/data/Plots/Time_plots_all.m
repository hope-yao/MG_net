figure
title('Plot of Computaion Time Vs Dimensionality')
xlabel('Dimensionality (No. of Elements in one direction)') 
ylabel('Time in Sec') 
% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
grid on
hold on

% Python Numpy: Conjugate Gradient Method - Intel Core i5-8350U CPU
y1=[0.00074,0.10062,112.65326];% put the time in sec for line 1
x1=[10,100,1000];%no of elements line 1
plot (x1,y1,'-r','LineWidth',1.5)
hold on

% Python Numpy: Conjugate Gradient Method - WS CPU
y2=[0.00084,0.06895,63.62284];% put the time in sec for line 2
x2=[10,100,1000];%no of elements line 2
plot (x2,y2,'--r','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method - Intel Core i5-8350U CPU
y3=[0.47984,8.74552,650.71783];% put the time in sec for line 2
x3=[10,100,1000];%no of elements line 2
plot (x3,y3,'-c','LineWidth',1.5)
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

% Tensorflow: Conjugate Gradient Method (Convolution) - Intel Core i5-8350U CPU
y6=[0.72087,47.44112,961.71353];% put the time in sec for line 1
x6=[10,100,1000];%no of elements line 1
plot (x6,y6,'-k','LineWidth',1.5)
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
legend({'Python Numpy: Conjugate Gradient Method - Intel Core i5-8350U CPU','Python Numpy: Conjugate Gradient Method - WS CPU','Tensorflow: Conjugate Gradient Method - Intel Core i5-8350U CPU','Tensorflow: Conjugate Gradient Method - WS CPU','Tensorflow: Conjugate Gradient Method - WS GPU','Tensorflow: Conjugate Gradient Method (Convolution) - Intel Core i5-8350U CPU','Tensorflow: Conjugate Gradient Method (Convolution) - WS CPU','Tensorflow: Conjugate Gradient Method (Convolution) - WS GPU'},'Location','northwest')