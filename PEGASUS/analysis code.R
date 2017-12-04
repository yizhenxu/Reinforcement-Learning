# change the probability of each direction to 0.225, so it is 0.9 prob for g(s,a,p) to use a


library("Rcpp")
library("RcppArmadillo")
library("parallel")

sourceCpp("~/codefoo.cpp")


map = matrix(0,5,5)
map[5,1] = 1 # start S
map[1,5] = 2 # end G

# 1up: i-1,j
# 2left: i,j-1
# 3down: i+1,j
# 4right: i,j+1

action.value = cbind(c(-1,0,1,0),c(0,-1,0,1)) 

# eachcat was 0.05 in the paper example, P(delta(s,a)) in g(s,a,p) would be 1 - eachcat*4

# location -> type of position 
#up, left, down, right
#i0,j0,  ,   --- 6
#  ,j0,i6,   --- 3 (S)
#  ,  ,i6,j6 --- 8
#  ,  ,  ,   --- 0 ->5
#i0,  ,  ,   --- 4
#  ,j0,  ,   --- 2
#  ,  ,i6,   --- 1 
#  ,  ,  ,j6 --- 7
#i0,  ,  ,j6 --- 11 (G)
# 1,2,3,4,5,6,7,8,11(G)

position = function(s){
  s.new = s + action.value
  type = sum(c(4,NA,1,NA,NA,2,NA,7)[s.new %in% c(0,6)])
  if(type == 0) type = 5
  return(type)
}

position.matrix = apply( expand.grid(1:5,1:5),1, position)
position.matrix = matrix(position.matrix,nrow=5)

# define policy class based on position
# policy: position -> action 1,2,3,4
# each column corresponds to a position -- 8 positions
policy.class = expand.grid(1:4,1:4,1:4,1:4,1:4,1:4,1:4,1:4)


gamma = 0.99
H = 100 # number of time steps

#mlist # number of simulated sample in each trial
#policy.num = 65536

#####################################################
###Train
maxm = 30
eachcat = 0.05

tmp = simplify2array(policy.class)
calc_all = function(mlist,eachcat){
  plist = matrix(runif(mlist*H), ncol = H)# plist is k specific
  look = pertrial_c(gamma, plist, tmp, position.matrix, action.value, eachcat  )
  return(look)
}
calc_all_rand = function(mlist,eachcat){
  plist = matrix(runif(mlist*65536*H), ncol = H)# plist is k specific
  look = pertrial_rand_c(gamma, plist, tmp, position.matrix, action.value, eachcat  )
  return(look)
}


set.seed(1)
opt_pi = matrix(NA,nrow = maxm, ncol = 8)
for(m in 1:maxm){
  res = calc_all(m, eachcat) # 65536 x m
  VM = apply(res, 1, mean)
  opt_pi[m,] = tmp[which.max(VM),]
}


opt_pi_rand = matrix(NA,nrow = maxm, ncol = 8)
for(m in 1:maxm){
  res = calc_all_rand(m, eachcat) # 65536 x m
  VM = apply(res, 1, mean)
  opt_pi_rand[m,] = tmp[which.max(VM),]
}


# maxm x K
K=10000

ptm <- proc.time()
randmat = matrix(runif(2*H*maxm*K),ncol=H)
opt_pi_mat = rbind(opt_pi, opt_pi_rand)
eval = Test_c(K, gamma, opt_pi_mat, position.matrix, action.value, eachcat, randmat)
meanPV = apply(eval, 1, mean)
proc.time() - ptm

save(meanPV,file = paste0("RLProj_K10000_TrainTest_seed1.RData"))

setwd("C:\\Users\\Yizhen Xu\\Google Drive\\Desktop\\2015 summer\\Reinforcement learning in HIV\\Project\\project code")


load("RLProj_K10000_TrainTest.RData")
load("RLProj_K10000_TrainTest_seed1.RData")

plot(1:30,meanPV[1:30],type = "o",ylim=c(min(meanPV),-7.5),xlab = "m", ylab="Mean Policy Value",main = "Figure 1b Replication (K=10,000)")
lines(1:30,meanPV[31:60],type = "o",col="red")
gamma =0.99
VH = function(E) -1*(1-gamma^E)/(1-gamma)
abline(h=VH(8))
text(locator(), "red line: random p \n black line: PEGASUS")
