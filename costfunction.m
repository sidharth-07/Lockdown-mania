function [Jval,grad]=costfunction(theta,X,y,lambda)
  m=size(X,1);
  Jval=(-1/m)*(sum(y.*(log(sigmoid(X*theta)))+(1-y).*(log(1-sigmoid(X*theta)))))+(lambda/(2*m))*(sum((theta).^2)-(theta(1)).^2);%returns the value of cost function and calls the sigmoid function
  pqr=theta;
  pqr(1)=0;
  grad=(1/m)*((X')*(sigmoid(X*theta)-y))+(lambda/m)*pqr;%returns the matrix comprising of partial derivatives of Jval w.r.t theta(i)
endfunction
