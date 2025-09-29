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


mat <- matrix(c(3,4,4,6), 2, 2)
egn <- eigen(mat)
t(matrix(egn$values)) %*% egn$vectors
egn$vectors %*% t(matrix(egn$values))


qrx <- qr(X)
qrx$qr





x <- rnorm(100) #+ rexp(100, rate=4)
# e <- rexp(100, rate=4)
y <- rnorm(100)


mod <- lm(y ~ x + I(x^2))
par(mfrow=c(2,2))
plot(mod)


dev.off()
qqnorm(x)
qqline(x)



A <- matrix(c(1, 1, 1, -1), 2, 2)

Ap <- cbind(rbind(A, A), rbind(A, A*-1))

Ap %*% t(Ap)

A %*% t(A)

eigen(A)




A <- matrix(c(1, 2, 3, 4), 2, 2)

t(A) %*% A


det(t(A) %*% A)
