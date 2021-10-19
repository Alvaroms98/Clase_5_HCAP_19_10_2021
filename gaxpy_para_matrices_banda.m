% Algoritmo de cálculo Gaxpy (matriz x vector) de una matriz banda
% con un ancho de banda "p"
p=1;
n = 20;

A = rand(n);
b = rand(n,1);

for i=1:n
    for j=max(1,i-p):min(n,i+p)
        y(i) = y(i) + A(i,j)*x(j);
    end
end
        
