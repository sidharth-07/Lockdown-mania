function [y]=sigmoid(X);
  y=1./(1+exp(-X));
endfunction
