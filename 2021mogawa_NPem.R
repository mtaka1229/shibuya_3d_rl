### RL Estimation & OD estimation by EM algorithm_NPL
### @author:Kenta Ishii
### coding:SHIFT-JIS
### 2020/11/02
### 2021/07/19


rm(list=ls(all=TRUE))
library(Matrix)
options(warn=-1) # Ignore all warnings
readdata.number <- -1 # -1 = all


print('OK')
#####for PC#####
# Input path
input_area<-"/home/ogawam/test/estimation5"
input_NW <- "NW"
input_data <- "data"

# network data ### toyosu multiscale nw
inputnodepass<-paste(input_area, input_NW, "mult_node_val_ishii.csv", sep = "/")
inputlinkpass<-paste(input_area, input_NW, "mult_link_val_withT_ishii.csv", sep = "/")
inputdemandpass<-paste(input_area, input_data, "ODsample7_23.csv", sep = "/")
inputflowpass<-paste(input_area, input_data, "test_mobile_spaceflowData_9_21_2020.csv", sep = "/")
inputtransitpass<-paste(input_area, input_data, "test_mobile_spacetransitData_9_21_2020.csv", sep = "/") ## 遷移確率のデータ(k→aの遷移確率, 和は1)

#inputdatapass<-paste(input_area, input_user2, input_project, input_data, "500m_routeOD4_3.csv", sep = "/")

#####for s31#####
# Input path
#input_area<-"NW/500mesh_toyosu"
#input_data<-"input_data/50mesh_route_dual"
#
## network data
#inputnodepass<-paste(input_area, "500mesh_node_ishii.csv", sep = "/")
#inputlinkpass<-paste(input_area,"500mesh_link_ishii.csv", sep = "/")
##inputdemandpass<-paste(input_area, "reduced_OD4.csv", sep = "/")
#inputdatapass<-paste(input_data, "500m_routeOD4_3.csv", sep = "/")

########## Calculation Time #############
message("PROGRAM START!\n")
time<-proc.time()

########## node data reading ############
node <- read.csv(inputnodepass)
nodeid <- sort(unique(node$fid))
N <- length(nodeid)
########## link data reading ############
link <- read.csv(inputlinkpass)
linkid <- sort(unique(link$id))
L <- length(linkid)

########## ODlist reading ##########
ODlist <- read.csv(inputdemandpass)
OD <- nrow(ODlist)

########## transit data reading ##########
## 流量の観測値
flow <- read.csv(inputflowpass, nrows = readdata.number)
flow <- flow[, -which (colnames(flow) %in% c("X", "Unnamed..0"))]
flow <- matrix(as.matrix(flow), nrow(flow), ncol(flow))

## 遷移確率の経験分布
transit <- read.csv(inputtransitpass, nrows = readdata.number)
transit <- transit[, -which (colnames(transit) %in% c("X", "Unnamed..0"))]
transit <- matrix(as.matrix(transit), nrow(transit), ncol(transit))

print('OK2')
#for(i in 1:Trip){
#  sub <- subset(data, data$TripID == i)
#  kdata[[i]] <- sort(unique(sub$k)) 
#  dataList[[i]] <- sub
#}
#for(i in 1:OD){
#  sub <- subset(data, data$ODID == i)
#  kdata[[i]] <- sort(unique(sub$k)) 
#  dataList[[i]] <- sub
#}

########## init #############
feature <-  c(-1, 1.5, 0.5, 0.5, 0.8) # 特徴量＝説明変数ということ
#w_od<- c(0.4, 0.6)#, 0.6)

#beta <- c(feature, w_od)#true parameter

f_init <- rep(0, length=length(feature))
w_init <- rep(1/OD, length=OD) # ���K���p�����[�^

binit <- c(f_init, w_init) # init parameter # 前半がRL説明変数，後半がOD交通量
n_feature <- length(feature)
theta <- 1
TS <- 100

print('OK3')
###########features##########
I <- array(0, dim = c(N, N))
length <- array(0, dim = c(N, N))
store <-array(0, dim = c(N, N))
park <-array(0, dim = c(N, N))
building <-array(0, dim = c(N, N))
station <-array(0, dim = c(N, N))
print('OK4')

#transit utility
for(i in 1:L){
  kn <- link$k[i]
  an <- link$a[i]
  k <- (1:length(nodeid))[nodeid == kn]
  a <- (1:length(nodeid))[nodeid == an]
  scale <- (an%/%1000)/5

  I[k,a] <- 1
  length[k,a] <- link$length[i]/1000
  length[a,k] <- link$length[i]/1000
  
  anode <- node[node$fid==an,]
  knode <- node[node$fid==kn,]
  
  store[k,a] <- anode$store/(10000*scale)
  park[k,a] <- anode$park/(1000*scale)
  building[k,a] <- anode$building/(10*scale)
  station[k,a] <- anode$station/scale
  
  
  store[a,k] <- knode$store/(10000*scale)
  park[a,k] <- knode$park/(1000*scale)
  building[a,k] <- knode$building/(10*scale)
  station[a,k] <- knode$station/scale

}
print('OK5')
#stay utility
for(i in 1:N){
  I[i,i] <- 1
  #length[i,i, 1] <- 0
  n <- node[i,]
  scale <- (n$fid%/%1000)/5
  
  store[i,i] <- n$store/(10000*scale)
  park[i,i] <- n$park/(1000*scale)
  building[i,i] <- n$building/(10*scale)
  station[i,i] <- n$station/scale
}
print('OK6')
#time space network
TSNW <- function(ODlist){
  Ilist <- list()
  
  # �N�_O����̓��B�\���Ɋւ���w���ϐ�It�̒�`
  Itlist <- list()
  II <- I # ��ԓI�ڑ��s��
  Itlist[[1]] <- diag(N)#sparseMatrix(seq(1, N), seq(1, N), x=1, dims = c(N, N)) # N*N�̒P�ʍs��
  for(ts in 2:(TS+1)){
    Itlist[[ts]] <- II
    II <- II%*%I # �s��̐ρi�e�����̐ς�*�j
    II <- (II > 0)*1 + (II == 0)*0
  }
  
  # �I�_D����̓��B�\���Ɋւ���w���ϐ�Itt�̒�`
  Ittlist <- list()
  for(ts in 1:(TS+1)){
    Ittlist[[ts]] <- t(Itlist[[(TS+1)-ts+1]]) # t()�͍s��̓]�u
  }
  
  # �w���ϐ������Ȃ��玞��ԓI�ڑ��s����쐬
  for(od in 1:OD){
    
    # OD�y�A���ƂɎ���ԓI�ڑ��s����쐬
    Id <- list()
    on <- ODlist$O[od]
    dn <- ODlist$D[od]
    o <- (1:N)[nodeid == on]
    d <- (1:N)[nodeid == dn]
    
    for(ts in 1:TS){
      Its <- array(0, dim = c(N, N))#sparseMatrix(1, 1, x=0, dims = c(N, N))
      if(ts == 1){
        Its[o, ] <- I[o, ]
        Id[[ts]] <- Its
        next
      }
      alist <- (1:N)[(Itlist[[ts+1]][o, ] == 1) & (Ittlist[[ts+1]][d, ]==1)]
      for(a in alist){
        if(Itlist[[ts+1]][o, a] == Ittlist[[ts+1]][d, a]){
          klist <- (1:N)[I[ ,a] == 1]
          for(k in klist){
            if(length((1:N)[Id[[ts-1]][, k] == 1]) != 0){
              Its[k, a] <- 1
            }
          }
        }
      }
      Id[[ts]] <- Its
    }
    Ilist[[od]] <- Id
  }
  return(Ilist)
}
print('OK7')
# calc observe link flow
assignment_proposal <- function(ODlist, beta){ ## betaは効用パラメータ（推定対象）
  Ilist <- TSNW(ODlist)
  
  #linkflow
  #F_link <- list()
  F_link <- array(0, dim = c(N, N))
  z_list <- calc_z(beta, Ilist)
  z <- z_list[[1]]
  z_re <- z_list[[2]]
  
  fe<-exp(length*beta[1])*exp(store*beta[2])*exp(park*beta[3])*exp(building*beta[4])*exp(station*beta[5])
  for(od in 1:OD){
    # time-space prism
    Id <- Ilist[[od]]
    
    # instantaneous utility M
    M <- list()
    for(ts in 1:TS){
      Mts <- Id[[ts]]*fe
      #Id[[ts]]
      M[[ts]] <- Mts
    }
    
    #calc link flow
    for(tt in 1:TS){
      ZD <- array(rep(z[tt+1,,od]), dim = c(N, N))
      ZD_re <- array(rep(z_re[tt,,od]), dim = c(N, N))
      ZD <- t(ZD)
      
      flow <- beta[(n_feature+od)]*ZD*M[[tt]]*ZD_re/z[1,(1:N)[nodeid==ODlist$O[od]],od]
      F_link <- F_link + as.array(flow)
      
      #F_link[tt,,] <- F_link[tt,,] + as.array(flow)
    }
  }
  return(F_link)
}


print('OK8')

# Calculation of Value funciton ## 価値関数と逆状態価値関数の両方をここで計算している
calc_z <- function(x, Ilist){ # ���p�֐�theta�Ɠ���
  
  z <- array(1, dim = c(TS+1,N,OD)) #exp(Vd)
  z_re <- array(1, dim = c(TS+1,N,OD))
  fe <- exp(length*x[1])*exp(store*x[2])*exp(park*x[3])*exp(building*x[4])*exp(station*x[5])
  z_list <- list()
  for(od in 1:OD){
    # instantaneous utility M
    M <- list()
    for(ts in 1:TS){
      Id <- Ilist[[od]]
      Mts <- Id[[ts]]*fe
      M[[ts]] <- Mts
    }
    
    #value function z
    for(tt in TS:1){ ## 逆向きに計算
      zi <- M[[tt]]%*%(z[tt+1,,od])
      z[tt,,od] <- as.vector((zi == 0)*1 + (zi != 0)*zi)
    }
    
    #proposal value function z_re
    for(tt in 1:TS){ ## 順向きに計算
      zii <- t(M[[tt]])%*%(z_re[tt,,od]) ### Mを転置して逆向きを表している
      z_re[tt+1,,od] <- as.vector((zii == 0)*1 + (zii != 0)*zii)
    }
  }
  z_list[[1]]<- z
  z_list[[2]]<-z_re
  return(z_list)
}
print('OK9')

######################################
######################################
####### q_zの算出：ここが重要そう ####### 観測多様体，e-stepのやつ
######################################
######################################
calc_qz <- function(x,Ilist){
  w <- x[(n_feature+1):length(x)] ### パラメタのうち，後ろの方の交通量の部分

  # od flow #### ここも含め，全てノードベースで考えている
  F_link <- array(0, dim = c(N, N, OD))
  F_a_kod <- array(0, dim = c(N, N, OD))
  F_transit <- array(0, dim = c(N, N, OD))
  
  # p(z|y,x)
  q_z <- array(0, dim = c(N, N, OD))
  
  flag <- array(1, dim = c(OD))
  fe <- exp(length*x[1])*exp(store*x[2])*exp(park*x[3])*exp(building*x[4])*exp(station*x[5])
  for(od in 1:OD){
    #linkflow
    F_link_od <- array(0, dim = c(N, N))
    
    # time-space prism
    Id <- Ilist[[od]]
    
    for(ts in 1:TS){
      Mts <- Id[[ts]]*fe
      
      rate <- Mts*ZD_ZD_re_list[ts,,,od] #### ZD_ZD_re_listが逆状態価値関数な気がするが
      flow <- w[od]*rate/sum(rate) ### w[od]はこのodの交通量（推定対象）
      F_link_od <- F_link_od + as.array(flow) ###### v/sum(v)のやつ！！！ #######
    }
    #flow
    F_link[,,od] <- F_link_od ###### F_link=p(z|x)な気がする ######
  }

  
  for(i in 1:N){ # 各ノードについて
    #flow
    sum <- F_link[i,,]%*%flag
    sum2 <-array(rep(sum), dim = c(N, OD))
    qz_n <- F_link[i,,]/sum2 
    q_z[i,,] <- qz_n
  }
  q_z[is.nan(q_z)] <-0
  return(q_z)
}
print('OK10')

# objective function ### これが全て，遷移確率および交通量を観測として
fr <- function(x){ # ������x
  z_list <- calc_z(x, Ilist) # x��theta��w�������Ă���̂œ������肵�Ă���
  z <- z_list[[1]]
  z_re <- z_list[[2]]
  ZD_ZD_re_list <- ZD_ZD_re(z, z_re)
  
  
  w <- x[(n_feature+1):length(x)] ### xの後ろの方は交通量
  F_link_od <- array(0, dim = c(OD, N, N)) 
  F_transit_od <- array(0, dim = c(OD, N, N))
  
  observe_flow_od <- array(0, dim = c(OD, N, N))
  observe_transit_od <- array(0, dim = c(OD, N, N))
  
  ### xの前の方はRLのモデルパラメータ
  fe <- exp(length*x[1])*exp(store*x[2])*exp(park*x[3])*exp(building*x[4])*exp(station*x[5])
  flag <- array(1, dim = c(N, 1))
  
  for(od in 1:OD){
    ##observe transit in each od
    observe_transit_od[od,,] <- observe_transit*q_z[,,od] # q_transit[,,od] ### observe_transitは観測された遷移確率, これにq_zをかける
    #observe flow in each od
    observe_flow_od[od,,] <- observe_flow*q_z[,,od] ## 各リンクの観測交通量（経験分布）にq_z(=q(z|x))を掛け，それをodペアの交通量としている
    # time-space prism
    Id <- Ilist[[od]]
    
    for(ts in 1:TS){
      # instantaneous utility M
      Mts <- Id[[ts]]*fe

      rate <- Mts*ZD_ZD_re_list[ts,,,od] # rateはODペアodにおいて各リンク（ノードk→a）を選択する確率
      F_link_od[od,,] <- F_link_od[od,,] + as.array(w[od]*rate/z[1,(1:N)[nodeid==ODlist$O[od]],od])
      ## ここでwで交通量がパラメタとして動かされる
      ## z[1,(1:N)[nodeid==ODlist$O[od]],od]はペアodのOにおける期待効用（割り算すればexp内で引かれる）
      ### F_link_od = p(x,z|gsi)
    }
  }
  #transit
  for(i in 1:N){
    sum <- sum(F_link_od[,i,]) ### 遷移を扱っているので元のノードiごとに処理している？？？→条件付きで扱えるようにしているぽい?
    F_transit_od[,i,]<-F_link_od[,i,]/sum # F_transit_od = p(a,z|k,gsi) 
  }

  #flow 
  F_list <- F_link_od/sum(F_link_od) ##### F_slit = p(x,z|gsi)
  
  F_list <- (F_list ==0)*1 + (F_list != 0)*F_list
  F_transit <- (F_transit_od==0)*1 + (F_transit_od!=0)*F_transit_od

  ## 二つのKLを足している
  CE <- sum(observe_transit_od*log(F_transit)) + sum(observe_flow_od*log(F_list))
  
  print(x)
  print(CE)
  
  return(CE)
}


print('OK11')
# calc ZD*ZD_re ### 状態価値関数と逆状態価値関数を合わせて処理→リンク交通量算出に用いる
ZD_ZD_re <- function(z, z_re){
  ZD_ZD_re_list <- array(0, dim = c(TS+1, N, N, OD))
  
  for(od in 1:OD){
    for(tt in 1:TS){
      ZD <- array(rep(z[tt+1,,od]), dim = c(N, N))
      ZD_re <- array(rep(z_re[tt,,od]), dim = c(N, N))
      ZD <- t(ZD)
      
      ZD_ZD_re_list[tt,,,od] <- ZD*ZD_re #### 単に要素ごとに掛け算してるだけ！
      #### ただしexpがかかっている．Mもexpに入った後なのでx=exp(V)として処理している，だからVとV_invを足し算してるのと同じ
    }
  }
  return(ZD_ZD_re_list)
}
print('OK3')

###### ここから推定回り始める ######
# Estimation by cross entropy, NPL
message("PROGRAM START!\n")

# supervised data #### 事前に取得したデータを読み込む
observe_flow <- flow #assignment_proposal(ODlist, beta)
observe_transit <- transit
b_flow<-sum(observe_flow)
observe_flow<-observe_flow/b_flow #### 観測交通量を正規化
### 以下各関数はobserve_flowをglobal変数として読み込んで処理する

# time space NW
Ilist <- TSNW(ODlist)

# init value function
b0 <- binit
z_list <- calc_z(b0, Ilist)
z0 <- z_list[[1]]
z0_re <- z_list[[2]]

ZD_ZD_re_list <- ZD_ZD_re(z0, z0_re)
q_z0 <- calc_qz(binit, Ilist)
#qz_list <- calc_qz(binit, Ilist)
#q_transit0<-qz_list[[1]]
#q_z0<-qz_list[[2]]

KL_bf <- 100
dL_KL <- 100
count <- 0

start <- proc.time()
# EM algorithm
############ Optimizing Time ############
time_opt<-proc.time()
#########################################
#ZD_ZD_re_list <- ZD_ZD_re(z, z_re)
print('OK4')
#KL_c <- 0
# EL algorithm
while(dL_KL >=2){
  #q_transit <- q_transit0
  q_z <- q_z0
  #print(sum(q_z))
  
  #m step ### 交通量とモデルパラメータを同時に動かしている
  res <- optim(b0, fr, method = "L-BFGS-B", lower=c(rep(-Inf, length=length(feature)), rep(0.01, length=OD)), upper=c(rep(5, length=length(feature)), rep(0.99, length=OD)), hessian = TRUE, control=list(fnscale=-1))
  
  b <- res$par
  z_list <- calc_z(b, Ilist) ### ���l�֐�
  z <- z_list[[1]]
  z_re <- z_list[[2]]
  ZD_ZD_re_list <- ZD_ZD_re(z, z_re)
  
  #e step
  q_z0<-calc_qz(b, Ilist)

  # KL error
  dL_KL <- abs(res$value - KL_bf)
  KL_bf <-res$value
  hhh <- res$hessian
  #print(b)
  print('hessian')
  print(hhh)
  print('KL div')
  print(KL_bf)
  
}
############ Optimizing Time ############
print("Optimized!!")
print(proc.time()-time_opt)
#########################################
# estimation result in NPL
b <- res$par
cat("b  =")
print(b)
cat("lnL=")
print(res$value)

e_flow <- assignment_proposal(ODlist, b)
t_flag <- (observe_flow !=0)*1 + (observe_flow==0)*0
e_flow_ob <-  e_flow*t_flag

#b_flow <- sum(observe_flow)
w <- b[(n_feature+1):length(b)]#### パラメタベクトルのうち交通量の部分
w <- w*b_flow/sum(e_flow_ob) 
b[(n_feature+1):length(b)]<-w  

#RMSE_flow
RMSE <- sqrt((observe_flow-e_flow_ob*b_flow/sum(e_flow_ob))^2/L)
cat("RMSE  =")
print(RMSE)

# ���ʂ̕\��
print(res)

hhh <- res$hessian
tval <- b/sqrt(-diag(solve(hhh)))
LL <- res$value
Lc <- fr(binit)

print(w)
print(Lc) 
print(LL)
print((Lc-LL)/Lc) 
print((Lc-(LL-length(b)))/Lc) 
print(b)

print(tval)

duration <- proc.time()-start
print(duration)

# ---------- �\������(NFXP)�F�v�Z�I�� ---------- #
message("ALL COMPLETED!\n")

