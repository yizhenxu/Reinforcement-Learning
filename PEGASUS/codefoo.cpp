#include <cstdlib> // standard c library, includes memory allocation, misc math, process control, string manipulation etc
#include <stdio.h>
#include <iostream>
#include <RcppArmadillo.h>
#include <math.h>  // for -INFINITY, NAN, isnan(), ceil

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// [[Rcpp::depends(RcppArmadillo)]]

using namespace std;//standard namespace, includes stuff like cout, cin, string, vector, map, etc
using namespace arma;

// [[Rcpp::export]]
arma::vec g_c(arma::vec s,double a,double p,arma::mat actionvalue, double eachcat){
  double asim;
  arma::vec snew = arma::zeros<arma::vec>(2);
  int count = 0;
  
  asim = std::ceil(p/eachcat);
  if(asim > 4) asim = a; 
  for(int k=0;k < 2;k++){
    snew(k) = s(k) + actionvalue(asim-1,k);
    if(snew(k) == 0 || snew(k) == 6){
      count += 1;
    }
  }
  if(count > 0){
    snew = s;
  }
  return(snew);
}

// [[Rcpp::export]]
arma::mat pertrial_c(double gamma,arma::mat plist,arma::mat policyclass,arma::mat positionmat, arma::mat actionvalue, double eachcat ){
  double a;
  double pnum = policyclass.n_rows;
  double H = plist.n_cols;
  double mlist = plist.n_rows;
  arma::mat straj = arma::zeros(2,H+1);
  arma::mat Vmat = arma::zeros(pnum,mlist);
  double E;
  
  for(int i=0; i<pnum; i++){
    for(int m=0; m<mlist; m++){
      straj(0,0) = 5; straj(1,0) = 1; E = H;
      
      for(int time=0; time<H; time++){

        a = policyclass(i, positionmat(straj(0,time)-1, straj(1,time)-1) -1);
        
        straj.col(time+1) = g_c(straj.col(time),a,plist(m,time),actionvalue,eachcat);
        
        if(straj(0,time+1) == 1 && straj(1,time+1) == 5){
          E = time+1;
          
          break;
        }
       
      }
     
      Vmat(i,m) = -1*(1-pow(gamma,E))/(1-gamma);
      
    }
  }
  return(Vmat);
}

// [[Rcpp::export]]
arma::mat Test_c(double K, double gamma,arma::mat policymat,arma::mat positionmat, arma::mat actionvalue, double eachcat, arma::mat randmat){
  double a;
  double maxm = policymat.n_rows;
  double H = randmat.n_cols;
  arma::mat straj = arma::zeros(2,H+1);
  arma::mat Vmat = arma::zeros(maxm,K);
  double E;
  int cnt = 0;
  
  for(int i=0; i<maxm; i++){
    for(int k=0; k<K; k++){
      straj(0,0) = 5; straj(1,0) = 1; E = H;
      
      for(int time=0; time<H; time++){
        
        a = policymat(i, positionmat(straj(0,time)-1, straj(1,time)-1) -1);
        
        straj.col(time+1) = g_c(straj.col(time),a,randmat(cnt,time),actionvalue,eachcat);
        
        if(straj(0,time+1) == 1 && straj(1,time+1) == 5){
          E = time+1;
          break;
        }
        
      }
      cnt++;
      Vmat(i,k) = -1*(1-pow(gamma,E))/(1-gamma);
    }
  }
  
  return(Vmat);
}

// [[Rcpp::export]]
arma::mat pertrial_rand_c(double gamma,arma::mat plist,arma::mat policyclass,arma::mat positionmat, arma::mat actionvalue, double eachcat ){
  double a;
  double pnum = policyclass.n_rows;
  double H = plist.n_cols;
  double mlist = plist.n_rows / pnum;
  arma::mat straj = arma::zeros(2,H+1);
  arma::mat Vmat = arma::zeros(pnum,mlist);
  double E;
  int cnt = 0;
  
  for(int i=0; i<pnum; i++){
    for(int m=0; m<mlist; m++){
      straj(0,0) = 5; straj(1,0) = 1; E = H;
      
      for(int time=0; time<H; time++){
        
        a = policyclass(i, positionmat(straj(0,time)-1, straj(1,time)-1) -1);
        
        straj.col(time+1) = g_c(straj.col(time),a,plist(cnt,time),actionvalue,eachcat);
        
        if(straj(0,time+1) == 1 && straj(1,time+1) == 5){
          E = time+1;
          
          break;
        }
        
      }
      cnt++;
      Vmat(i,m) = -1*(1-pow(gamma,E))/(1-gamma);
      
    }
  }
  return(Vmat);
}

