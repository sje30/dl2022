## Solve the XOR problem

## let's build a sigmoidal activation
g <- function(x) { 1 / (1+ exp(-x)) }

## be careful about interpreting gprime!!!
gprime <- function(x) { y <- g(x); y * (1 - y) }

#curve(g, xlim=c(-3, 3))
#curve(gprime, xlim=c(-3, 3))
bias = -1

epsilon = 0.5

data = matrix( c(0, 0, bias, 0,
                 0, 1, bias, 1,
                 1, 0, bias, 1,
                 1, 1, bias, 0), 4, 4, byrow=TRUE)

inputs = t(data[,1:3])
targets = data[,4]


I=2 #excluding bias
J=2 #eclusing bias
K=1

W1 = matrix(runif(J*(I+1)), J, I+1) #+1 for bias
W2 = matrix(runif(K*(J+1)), K, J+1)


z_j = matrix(0, J+1, 1)
delta_j = rep(0,J)

nepoch = 2000
errors = rep(0, nepoch)

for (epoch in 1:nepoch) {
  DW1 = matrix(0, J, I+1)
  DW2 = matrix(0, K, J+1)

  epoch_err = 0.0


  for (i in 1:ncol(inputs)) {
    ## forward activation
    z_i = inputs[,i,drop=FALSE] # keep as col vector
    t_k = targets[i]
    
    ## input to hidden
    x_j = W1 %*% z_i

    for (q in 1:J) {
      z_j[q] = g(x_j[q])
    }
    z_j[J+1] = bias
    

    ## hidden to output

    x_k = W2 %*% z_j
    z_k = g(x_k)

    error = sum(0.5 * (t_k - z_k)^2)

    epoch_err = epoch_err + error
    
    ## backward error propagation.
    delta_k = gprime(x_k) * (t_k - z_k)
    DW2 = DW2 + outer(as.vector(delta_k), as.vector(z_j))

    ## Now get deltas for hidden layer.
    for (q in 1:J) {
      delta_j[q] = gprime(x_j[q]) * delta_k[1] * W2[1,q]
    }
    
    ## for (p in 1:(I+1)) {
    ##   for (q in 1:J) {
    ##     DW1[q,p] = DW1[q,p] + delta_j[q] * z_i[p]
    ##   }
    ## }
    DW1 = DW1 + outer( delta_j, as.vector(z_i))
    ##m = as.vector(z_i)
    ##m = rep(0,3)
    ##DW1 = DW1 + outer(delta_j, rep(0,3))
    ##DW1 = DW1 + outer(delta_j, m)
  }

  

  ## end of an epoch.
  errors[epoch] = epoch_err
  if ((epoch %%50)==0) {
    print(epoch_err)
  }
  W2 = W2 + (epsilon*DW2)
  W1 = W1 + (epsilon*DW1)
  ##print(W1)
}

plot(errors)



print_activations <-  function() {

  z_j = matrix(0, J+1, 1)
  
  n_inputs = ncol(inputs)
  ncol = I + J + K
  activations = matrix(0, n_inputs, ncol)
  for (i in 1:n_inputs) {
    z_i = inputs[,i, drop=FALSE]
    x_j = W1 %*% z_i
    for (q in 1:J) {
      z_j[q] = g(x_j[q])
    }
    z_j[J+1] = bias

    x_k = W2 %*% z_j
    z_k = g(x_k)
    ##browser()
    activations[i,] = c( z_i[1:I], z_j[1:J], z_k)
  }
  activations
}
