# LOCO example

set.seed(327)

d <- 500
n <- 200

B_a <- rnorm(5, 0, 2)
B <- matrix(c(B_a, rep(0, d-5)), nrow = d, ncol = 1)

X <- rnorm(n*d, 0, 1)
X <- matrix(X, nrow = n, ncol = d, byrow = T)
e <- rnorm(n, 0, 1)

# Y <- B %*% X + e
Y <- X %*% B + e

dim(Y)

library(glmnet)

#Penalty type (alpha=1 is lasso and alpha=0 is the ridge)
cv.lambda.lasso <- cv.glmnet(x=X, y=Y, alpha = 1)
plot(cv.lambda.lasso)

cv.lambda.lasso
