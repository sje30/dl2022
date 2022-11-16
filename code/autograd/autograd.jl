using Plots

## The problem with finite approximation

logistic(x) = 1.0 / (1.0 + exp(0-x) )
deriv(x) = exp(-x) / (1+exp(-x))^2  ## used for ground-truth!


function myerror(x, h)
    est = (logistic(x+h) - logistic(x) ) / h
    abs(deriv(x) - est)
end

## Finite difference approximation to gradient.
hs = [10^-x for x in 1.0:16.0]
fderrors = [myerror(1.0, h) for h in hs]
plot(hs, fderrors, xlab="step size", ylab="abs. error", xaxis=:log, yaxis=:log)

struct D <: Number  # D is a function-derivative pair
    f::Tuple{Float64,Float64}
end

import Base: +, -, *, /, convert, promote_rule
+(x::D, y::D) = D(x.f .+ y.f)
-(x::D, y::D) = D(x.f .- y.f)
*(x::D, y::D) = D((x.f[1]*y.f[1], (x.f[2]*y.f[1] + x.f[1]*y.f[2])))
/(x::D, y::D) = D((x.f[1]/y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
convert(::Type{D}, x::Real) = D((x,zero(x)))
promote_rule(::Type{D}, ::Type{<:Number}) = D

Base.show(io::IO,x::D) = print(io,x.f[1]," + ",x.f[2]," ϵ")

""" Let E be an error fuction given two parameters x and y"""
E(x, y) = x*y - 3x +x^2 - y^2

x = D( (1.0, 1.0) )
y = D( (2.0, 0.0) )
E(x, y)

x = D( (1.0, 0.0) )
y = D( (2.0, 1.0) )
E(x, y)



## Computations can be differentiated, as shown here by
## Babylonian algorithm for computing square roots.


function Babylonian(x; N = 10)
    t = (1+x)/2
    for i = 2:N; t=(t + x/t)/2  end
    t
end


α = π
Babylonian(α), √α

x=49; Babylonian(D((x,1))), (√x,.5/√x)
x=π; Babylonian(D((x,1))), (√x,.5/√x)

## Taken from https://github.com/JuliaAcademy/JuliaAcademyMaterials/blob/main/Courses/Foundations%20of%20machine%20learning/20.Automatic-Differentiation-in-10-Minutes.jl


######################################################################
import Base: exp
## homework -- why is this following rule true? (Taylor expand).
exp(x::D) = exp(x.f[1])* D((1, x.f[2]))


ϵ = D((0,1))
x = 0.5 + ϵ
logistic(x)
logistic(x).f[2] ≈ deriv(x.f[1])



xs = collect(-5:0.01:5)
ys = logistic.(xs)

xs2 = [x + ϵ for x in xs]
op2 = map(logistic, xs2)

ims = [x.f[2] for x in op2]
op = map(logistic, xs)

plot(xs, ys)
plot!(xs, ims)




