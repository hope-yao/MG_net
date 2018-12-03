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

% legend({'y1 = method 1','y2 = method 2','y3 = method 3','y4 = method 4','y5 = method 5','y6 = method 6'},'Location','northwest')
% Replace the name in corresponding legend for respective method
legend({'Python Numpy: Conjugate Gradient Method - Intel Core i5-8350U CPU','Python Numpy: Conjugate Gradient Method - WS CPU'},'Location','northwest')