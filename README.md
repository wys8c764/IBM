Improved Bayesian Method (MATLAB)
function H = ibm(trainset,testset,backgroundset,label)
trainset: m*p matrix with m observations and p features
testset: n*p matrix with n observations and p features
backgroundset: g*p matrix with g observations and p features
backgroundset is used to calculate the correlation coefficient matrix of the p features, and the g observations are independent of the m + n obversations in trainset and testset
label: 1*m vector, including the classification labels of the m observations in trainset
H: output result, n*k matrix, where k is the number of classifications
