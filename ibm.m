function H = ibm(trainset,testset,backgroundset,label)

%Improved Bayesian Method
%trainset: m*p matrix with m observations and p features
%testset: n*p matrix with n observations and p features
%backgroundset: g*p matrix with g observations and p features
%  backgroundset is used to calculate the correlation coefficient matrix of
%  the p features, and the g observations are independent of the m + n
%  obversations in trainset and testset.
%label: 1*m vector, including the classification labels of the
%  m observations in trainset.
%H: output result, n*k matrix, where k is the number of classifications

R = corr(backgroundset);
R(isnan(R)) = 0;
[P,D] = eig(R);
for i = 1 : length(D)
    if D(i,i) < 0.001 * D(1,1)
        break
    end
end
D = D(1:i-1, 1:i-1);
P = P(:, 1:i-1);

l = unique(label);
for i = 1 : length(l)
    mu = mean(trainset);
    muk = mean(trainset(label == l(i),:));
    sigma = std(trainset);
    sigmak = std(trainset(label == l(i),:));
    for j = 1 : size(testset,1)
        beta = ((testset(j,:) - mu)./(sigma + 0.001)) * P;
        gamma = ((testset(j,:) - muk)./(sigmak + 0.001)) * P;
        H(j,i) = sum((beta.^2 - gamma.^2)./(2 * diag(D)')) + ...
            log(sum(label == l(i)) / length(label));
    end
end