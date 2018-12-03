figure
title('Plot of Computaion Time Vs Dimensionality')
xlabel('Dimensionality (No. of Elements in one direction)') 
ylabel('Time in Sec') 
% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
grid on
hold on

% Tensorflow: Conjugate Gradient Method - Intel Core i5-8350U CPU
y1=[0.47984,8.74552,650.71783];% put the time in sec for line 2
x1=[10,100,1000];%no of elements line 2
plot (x1,y1,'-c','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method - WS CPU
y2=[0.72283,5.92201, 55.55385];% put the time in sec for line 1
x2=[10,100, 1000];%no of elements line 1
plot (x2,y2,'--c','LineWidth',1.5)
hold on

% Tensorflow: Conjugate Gradient Method - WS GPU
y3=[0.73323,5.92948, 54.82987];% put the time in sec for line 1
x3=[10,100, 1000];%no of elements line 1
plot (x3,y3,':c','LineWidth',1.5)
hold on

% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
% Replace the name in corresponding legend for respective method
legend({'Tensorflow: Conjugate Gradient Method - Intel Core i5-8350U CPU','Tensorflow: Conjugate Gradient Method - WS CPU','Tensorflow: Conjugate Gradient Method - WS GPU'},'Location','northwest')