n <- 10
p <- 100

X <- matrix(rnorm(n*p), ncol=p, nrow=n)
y <- matrix(rnorm(n), ncol=1, nrow=n)

xtx <- t(X) %*% X

xtx[1,]

library(pracma)
library(MASS)
Rank(xtx)


bhat <- pinv(X) %*% y
yhat <- X %*% bhat

lmod <- lm(y ~ X)
# Coeffs way different but fitted values are the same
lmod$fitted.values
yhat