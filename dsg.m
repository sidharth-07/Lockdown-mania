 %One-versus-all classification algorithm
 
 
 Q=csvread('coronaTrain.csv');%Loads the Training set into Q
 data=Q(2:end,2:end);%deletes the first row and first column
 X=data(:,1:1000);
 y=data(:,1001);
 m=size(X,1);%Total no. of training examples
 X=[ones(m,1),X];%adds an extra column of ones to X
 opt_theta=zeros(3,1001);%Initializes the matrix comprising of optimum values of theta(at which the minimum of the costfunctions of three separate classes occurs)
 lambda=100;%regularization parameter
 count=0;%counts the no. of correct outputs for the training set
 pred=ones(m,1);%'pred' stores the predicted output for each training example
 for i=1:3,% To find the optimum parameters for each hypothesis of the three classes
   options = optimset('GradObj', 'on', 'MaxIter', 50);
   initial_theta=zeros(1001,1);%Initial guess
   z=fmincg(@(t)(costfunction(t,X,(y==(i-1)),lambda)),initial_theta,options);%Calls the function fmincg to minimize the cost function
   opt_theta(i,:)=z';
 endfor

 for g=1:m,
   [u,v]=max((opt_theta)*(X(g,:)'));%u and v return the maximum value and the index at which it occurs of the 3X1 matrix
   pred(g)=v-1;
   if pred(g)==y(g),
     count=count+1;%count increases by one if it predicts the correct output
   end
   
   
 endfor

 disp((count/m)*100);%displays the accuracy of the trained classifier on the training set

 p=csvread('coronaTest.csv');%Loads the Test set into p
 a=p(2:end,2:end);%deletes the first row and first column
 a=[ones((size(a,1)),1),a];%adds an extra column of ones to a
 predict=zeros(size(a,1),1);%predict' stores the predicted output for each entry of the test set
 for j=1:size(a,1),
   [c,d]=max(sigmoid(opt_theta*((a(j,:))')));
   predict(j)=d-1;
 endfor