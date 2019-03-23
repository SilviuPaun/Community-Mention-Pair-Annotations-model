package CommunityMPA;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.special.Gamma;

public class Model {

	private int I; //the number of mentions
	private int J; //number of annotators
	private int K; //number of classes
	private int C; //max number of communities
	private int[] Mi; //number of labels per mention
	private int[][] Nim; //number of decisions per mention
	
	//variational parameters - same notation as in the paper
	private double[] lambda, eta;
	private double[][] gamma, mu;
	private double[][] theta, epsilon;
	private double[][] tau_mu, tau_sigma;
	private double[][] nu_mu, nu_sigma;
	private double[][] sigma_alpha, sigma_beta;
	private double[][] omega_alpha, omega_beta;
	private double[][] kappa;
	private double[] f, g;
	private double b_alpha, b_beta;
	
	private double[][] phi, zeta;
	
	private double[][][] delta, rho;
	
	//hyper-parameters
	private double a0,a1,d0,d1,e0,e1,t0,t1,u0,u1, s0, s1;
	
	//useful speed params
	private double[][] log_phi, log_zeta, log_kappa;
	
	private double[] digamma_lambda, digamma_eta; 
	private double[] digamma_lambdaPLUSeta; 
	private double[] lambdaPLUSeta;
	
	//sufficient statistics
	double[] ss_lambda, ss_eta;
	double[][] ss_gamma, ss_mu;
	double[][] ss_theta, ss_epsilon;
	double[][] ss_tau_mu, ss_nu_mu, ss_sigma_beta, ss_omega_beta;
	double[] ss_indicator;
	double[] ss_f, ss_g;
	double ss_b_beta;
	
	
	//data collections
	private List<String> items;
	private List<String> classes; //the classes the labels are part of; e.g.: DN, DO
	private Map<String, List<String>> itemInterpretations; //the mention-level labels; e.g.: the antecedents
	private List<String> annotators;
	
	private Map<Pair, Integer> itemInterpretationClass; // the z_i,m
	private Map<Pair, List<Integer>> itemInterpretationValidations; // the y_i,m,n
	private Map<Triple, Integer> annotatorOfItemInterpretationValidation; // for jj[i,m,n]
	
	private Map<Integer, List<Integer>> itemAnnotations;
	private Map<Pair, Integer> annotatorOfItemAnnotation; // for jj(i,m,n)
	
	private Map<String, String> itemGoldStandard; //the gold labels for each mention
	private Map<String, String> itemDocument;
	
	//general settings
	private int RANDOM_SEED;
	private double convergence_threshold = Math.pow(10, -6);
	private String convergenceInfo;
	private Double runTime;
	
	public void setSeed(Integer seed) {
		RANDOM_SEED = seed;
	}
	
	public void Launch()
	{
		String relativePath = "/JMLR 2019/";
		String annotationsFile = "example.csv";
		
		LoadAnnotations(relativePath + annotationsFile);
		
		InitializeParameterSpace();
		InitializeParameters(relativePath + "mpa-profiles-" + annotationsFile);
		
		System.out.println();
		System.out.println("Processed file: " + annotationsFile);
		System.out.println("Items: " + I);
		System.out.println("Annotators: " + J);
		System.out.println(K + " classes: " + classes);
		System.out.println("Truncation level: " + C);
		System.out.println();
		
		RunModel();
		
		int noGoldLabels = itemGoldStandard.keySet().size();
		Map<Integer, Integer> inferredLabels = GetInferredLabels();
		double accuracy = ComputeAccuracy(inferredLabels);
		System.out.println();
		System.out.println("CMPA  accuracy: " + accuracy + " " + (int)(accuracy*noGoldLabels) + "/" + noGoldLabels);
		
		ComputeClassEvaluation(inferredLabels);
		
		int bestSeed = -1;
		double bestAccuracyMV = -1.0;
		for(int seed = 1; seed<=10; seed++)
		{
			Random rand = new Random(seed);
			
			Map<Integer, Integer> majorityVote = GetMajorityVoting(rand);
			
			double accuracyMV = ComputeAccuracy(majorityVote);
			System.out.println("MV seed " + seed + " :" + accuracyMV + " " + (int)(accuracyMV*noGoldLabels) + "/" + noGoldLabels);
			
			if(accuracyMV > bestAccuracyMV)
			{
				bestAccuracyMV = accuracyMV;
				bestSeed = seed;
			}
		}
		System.out.println("Best MV was found at seed " + bestSeed + ": " + bestAccuracyMV);
		
		PrintCommunities(0.02);
	}
	
	private void ComputeClassEvaluation(Map<Integer, Integer> results)
	{
		Map<String, Integer> classMatches = new HashMap<String, Integer>();
		Map<String, Integer> classCountsPrecission = new HashMap<String, Integer>();
		Map<String, Integer> classCountsRecall = new HashMap<String, Integer>();
		for(String itemID : itemGoldStandard.keySet())
		{
			int i = items.indexOf(itemID);
			
			List<String> itemInterpretationsList = itemInterpretations.get(itemID);
			
			String silverLabel = itemInterpretationsList.get(results.get(i));
			String goldLabel = itemGoldStandard.get(itemID);
			
			String classStringPrecission = silverLabel.substring(0, silverLabel.indexOf("("));
			String classStringRecall = goldLabel.substring(0, goldLabel.indexOf("("));
			
			if(silverLabel.equals(goldLabel))
				classMatches.put(classStringPrecission, classMatches.containsKey(classStringPrecission) ? classMatches.get(classStringPrecission) + 1 : 1);
			
			classCountsPrecission.put(classStringPrecission, classCountsPrecission.containsKey(classStringPrecission) ? classCountsPrecission.get(classStringPrecission) + 1 : 1);
			classCountsRecall.put(classStringRecall, classCountsRecall.containsKey(classStringRecall) ? classCountsRecall.get(classStringRecall) + 1 : 1);
		}
		
		System.out.println();
		System.out.println("A per class evaluation:");
		for(String classString : classMatches.keySet())
		{
			int matches = classMatches.get(classString);
			int countPrecission = classCountsPrecission.get(classString);
			int countRecall = classCountsRecall.get(classString);
			
			double P = 1.0 * matches / countPrecission;
			double R = 1.0 * matches / countRecall;
			double F1 = 2.0 * P * R / (R + P);
			
			System.out.println(classString + " P: " + P);
			System.out.println(classString + " R: " + R);
			System.out.println(classString + " F1: " + F1);
		}
		System.out.println();
	}
	
	private  void PrintCommunities(double min_prevalence)
	{
		Map<Double, Integer> prevalence = new HashMap<Double, Integer>();
		for(int r=0; r<C; r++)
		{
			double length = MeasureStick(r);
			prevalence.put(length, r);
		}
		
		List<Double> lengths = new ArrayList<Double>();
		lengths.addAll( prevalence.keySet() ); 
		Collections.sort(lengths, Collections.reverseOrder());
		
		System.out.println();
		System.out.println("Community profiles with a prevalence above " + min_prevalence + ":");
		System.out.println();
		for(int pos = 0; pos < lengths.size(); pos++)
		{
			double length = lengths.get(pos);
			if(length >= min_prevalence)
			{
				int cluster = prevalence.get(length);
				
				System.out.println(cluster + " - " + length);
				
				for(int h=0; h<K; h++)
				{
					String classID = classes.get(h);
					System.out.println(classID + " specificity: " + logistic(nu_mu[cluster][h]));
					System.out.println(classID + " sensitivity: " + logistic(tau_mu[cluster][h]));
				}
				System.out.println();
			}
		}
	}
	
	private void RunModel() 
	{
		boolean convergence = false;
		int nIter = 1;
		double lowerBoundOLD = 0.0;
		double lowerBoundNEW = 0.0;
		long startTime = System.currentTimeMillis();
		
		while(!convergence)
		{
			UpdateParameters();
			
			if(nIter==1)
			{
				lowerBoundOLD = ComputeLowerBound();
				System.out.println(RANDOM_SEED + " - lowerbound at " + nIter + ": " + lowerBoundOLD);
				nIter++;
			}
			else
			{
				int skip = 1; //if you don't want to check every iteration for convergence
				if(nIter % skip == 0) 
				{
					lowerBoundNEW = ComputeLowerBound();
					
					System.out.println(RANDOM_SEED + " - lowerbound at " + nIter + ": " + lowerBoundNEW);
					
					convergence = CheckConvergence(lowerBoundOLD, lowerBoundNEW);
					if(!convergence)
						lowerBoundOLD = lowerBoundNEW;
					else
					{
						convergenceInfo = "Convergence achieved in " + nIter + " iterations with a lowerbound value of " + lowerBoundNEW;
						System.out.println(RANDOM_SEED + " - " + convergenceInfo);
						
						long endTime = System.currentTimeMillis();
						runTime =  (endTime-startTime) / 1000.0;
						System.out.println(RANDOM_SEED + " - Run time: " + runTime + " seconds");
					}
				}
				nIter++;
			}
		}
	}
	
	//note that some constants were discarded
	private double ComputeLowerBound() 
	{
		double lowerbound = 0.0;
		
		for(int i=0; i<I; i++)
		{
			for(int m=0; m<Mi[i]; m++)
			{
				int classIdx = itemInterpretationClass.get(new Pair(i,m)); 
				List<Integer> itemInterpretationValidationsList = itemInterpretationValidations.get(new Pair(i, m));
				
				lowerbound += phi[i][m] * (digamma_lambda[classIdx] - digamma_lambdaPLUSeta[classIdx]); //
				lowerbound += zeta[i][m] * (digamma_eta[classIdx] - digamma_lambdaPLUSeta[classIdx]); //
				
				if(phi[i][m] > 0)
					lowerbound -= phi[i][m] * Math.log(phi[i][m]); //
				if(zeta[i][m] > 0)
					lowerbound -= zeta[i][m] * Math.log(zeta[i][m]); //
				
				for(int n=0; n<Nim[i][m]; n++)
				{
					int annotatorIdx = annotatorOfItemInterpretationValidation.get(new Triple(i, m, n));
					int annotation = itemInterpretationValidationsList.get(n);
					
					double e_log_y_alpha = Math.log(logistic(delta[i][m][n]));
					e_log_y_alpha += gamma[annotatorIdx][classIdx] * annotation;
					e_log_y_alpha -= (0.5) * ( gamma[annotatorIdx][classIdx] + delta[i][m][n] );
					e_log_y_alpha -= lambda(delta[i][m][n]) * ( Math.pow(gamma[annotatorIdx][classIdx], 2)  + mu[annotatorIdx][classIdx] - Math.pow(delta[i][m][n], 2) );
					
					lowerbound += phi[i][m] * e_log_y_alpha; //
					
					double e_log_y_beta = Math.log(logistic(rho[i][m][n]));
					e_log_y_beta -= theta[annotatorIdx][classIdx] * annotation;
					e_log_y_beta += (0.5) * ( theta[annotatorIdx][classIdx] - rho[i][m][n] );
					e_log_y_beta -= lambda(rho[i][m][n]) * ( Math.pow(theta[annotatorIdx][classIdx], 2)  + epsilon[annotatorIdx][classIdx] - Math.pow(rho[i][m][n], 2) );
					
					lowerbound += zeta[i][m] * e_log_y_beta; //
				}
			}
		}
		
		for(int h=0; h<K; h++)
		{
			lowerbound += (a0 - 1) * (digamma_lambda[h] - digamma_lambdaPLUSeta[h]); //
			lowerbound += (a1 - 1) * (digamma_eta[h] - digamma_lambdaPLUSeta[h]); //
			lowerbound += Gamma.logGamma(a0 + a1); //
			lowerbound -= Gamma.logGamma(a0); //
			lowerbound -= Gamma.logGamma(a1); //
			
			lowerbound -= (lambda[h] - 1) * (digamma_lambda[h] - digamma_lambdaPLUSeta[h]); //
			lowerbound -= (eta[h] - 1) * (digamma_eta[h] - digamma_lambdaPLUSeta[h]); //
			lowerbound -= Gamma.logGamma(lambdaPLUSeta[h]); //
			lowerbound += Gamma.logGamma(lambda[h]); //
			lowerbound += Gamma.logGamma(eta[h]); //
		}
		
		lowerbound += s0 * Math.log(s1) - Gamma.logGamma(s0) + (s0 - 1) * ( Gamma.digamma(b_alpha) - Math.log(b_beta) ) - s1 * b_alpha / b_beta;
		lowerbound -= ( b_alpha * Math.log(b_beta) - Gamma.logGamma(b_alpha) + (b_alpha - 1) * ( Gamma.digamma(b_alpha) - Math.log(b_beta) ) - b_alpha );
		
		for(int r=0; r<C; r++)
		{
			if( r != (C-1) )
			{
				lowerbound += ( b_alpha / b_beta - 1) * ( Gamma.digamma(g[r]) - Gamma.digamma(f[r] + g[r]) ); //
				lowerbound += Gamma.digamma(b_alpha) - Math.log(b_beta); //
				
				lowerbound -= (f[r] - 1) * ( Gamma.digamma(f[r]) - Gamma.digamma(f[r] + g[r]) ); //
				lowerbound -= (g[r] - 1) * ( Gamma.digamma(g[r]) - Gamma.digamma(f[r] + g[r] )); //
				lowerbound -= Gamma.logGamma(f[r] + g[r]); //
				lowerbound += Gamma.logGamma(f[r]); //
				lowerbound += Gamma.logGamma(g[r]); //
			}
			
			for(int h=0; h<K; h++)
			{
				lowerbound += (-0.5) * (1/d1) * ( Math.pow(tau_mu[r][h], 2)  + tau_sigma[r][h]  - 2 * d0 * tau_mu[r][h] );  //
				lowerbound += (-0.5) * (1/t1) * ( Math.pow(nu_mu[r][h], 2)  + nu_sigma[r][h]  - 2 * t0 * nu_mu[r][h] ); //
				
				lowerbound += 0.5 * Math.log(tau_sigma[r][h]); //
				lowerbound += 0.5 * Math.log(nu_sigma[r][h]); //
				
				lowerbound += (-e0 - 1) *  ( Math.log(sigma_beta[r][h]) - Gamma.digamma(sigma_alpha[r][h]) ) - e1 * sigma_alpha[r][h]/sigma_beta[r][h]; //
				lowerbound += (-u0 - 1) *  ( Math.log(omega_beta[r][h]) - Gamma.digamma(omega_alpha[r][h]) ) - u1 * omega_alpha[r][h]/omega_beta[r][h]; //
				
				lowerbound += sigma_alpha[r][h] + Math.log( sigma_beta[r][h] ) +  Gamma.logGamma(sigma_alpha[r][h])  - (1 + sigma_alpha[r][h]) * Gamma.digamma(sigma_alpha[r][h]); //
				lowerbound += omega_alpha[r][h] + Math.log( omega_beta[r][h] ) + Gamma.logGamma(omega_alpha[r][h])  - (1 + omega_alpha[r][h]) * Gamma.digamma(omega_alpha[r][h]); //
			}
		}
		
		for(int j=0; j<J; j++)
		{
			for(int r=0; r<C; r++)
			{
				if(r != (C-1))
				{
					lowerbound += kappa[j][r] * ( Gamma.digamma(f[r]) - Gamma.digamma(f[r] + g[r]) ); //
					
					double temp = 0.0;
					for(int r_prime = r+1; r_prime < C; r_prime++)
						temp += kappa[j][r_prime];
					
					lowerbound += temp * ( Gamma.digamma(g[r]) - Gamma.digamma(f[r] + g[r]) ); //
				}
				if(kappa[j][r] > 0)
					lowerbound -= kappa[j][r] * Math.log(kappa[j][r]);
			}
			
			for(int h=0; h<K; h++)
			{
				for(int r=0; r<C; r++)
				{
					double temp_alpha = (-0.5) * ( Math.log(2*Math.PI) + Math.log(sigma_beta[r][h]) - Gamma.digamma(sigma_alpha[r][h]) 
							+ (sigma_alpha[r][h]/sigma_beta[r][h]) *( Math.pow(gamma[j][h], 2) + mu[j][h] - 2.0 * gamma[j][h] * tau_mu[r][h] + Math.pow(tau_mu[r][h],2) + tau_sigma[r][h] ) ) ; //
					lowerbound += kappa[j][r] * temp_alpha;
					
					double temp_beta = (-0.5) * ( Math.log(2*Math.PI) + Math.log(omega_beta[r][h]) - Gamma.digamma(omega_alpha[r][h])
							+ (omega_alpha[r][h]/omega_beta[r][h]) * ( Math.pow(theta[j][h], 2) + epsilon[j][h] - 2.0 * theta[j][h] * nu_mu[r][h] + Math.pow(nu_mu[r][h],2) + nu_sigma[r][h] ) ) ; //
					lowerbound += kappa[j][r] * temp_beta;
				}
				
				lowerbound += 0.5 * Math.log(mu[j][h]); //
				lowerbound += 0.5 * Math.log(epsilon[j][h]); //
			}
		}
		
		return lowerbound;
	}
	
	private void UpdateParameters() 
	{
		//-----------reset sufficient statistics-----------------
		ss_lambda = new double[K];
		ss_eta = new double[K];
		
		ss_gamma = new double[J][K];
		ss_mu = new double[J][K];
		
		ss_theta = new double[J][K];
		ss_epsilon = new double[J][K];
		
		ss_tau_mu = new double[C][K];
		ss_nu_mu = new double[C][K];
		
		ss_sigma_beta = new double[C][K];
		ss_omega_beta = new double[C][K];
		
		ss_indicator = new double[C];
		
		ss_b_beta = 0.0;
		
		ss_g = new double[C]; //we actually need C-1; last remains unused
		//-------------------------------------------------------
		
		for(int i=0; i<I; i++)
		{
			for(int m=0; m<Mi[i]; m++)
			{
				int classIdx = itemInterpretationClass.get(new Pair(i,m)); 
				List<Integer> itemInterpretationValidationsList = itemInterpretationValidations.get(new Pair(i, m));
				
				log_phi[i][m] = digamma_lambda[classIdx] - digamma_lambdaPLUSeta[classIdx];
				log_zeta[i][m] = digamma_eta[classIdx] - digamma_lambdaPLUSeta[classIdx];
				
				for(int n=0; n<Nim[i][m]; n++)
				{
					int annotatorIdx = annotatorOfItemInterpretationValidation.get(new Triple(i, m, n));
					int annotation = itemInterpretationValidationsList.get(n);
					
					delta[i][m][n] = Math.pow(gamma[annotatorIdx][classIdx], 2) + mu[annotatorIdx][classIdx];
					delta[i][m][n] = Math.sqrt(delta[i][m][n]);

					rho[i][m][n] = Math.pow(theta[annotatorIdx][classIdx], 2) + epsilon[annotatorIdx][classIdx];
					rho[i][m][n] = Math.sqrt(rho[i][m][n]);
					
					log_phi[i][m] += Math.log(logistic(delta[i][m][n]));
					log_phi[i][m] += gamma[annotatorIdx][classIdx] * annotation;
					log_phi[i][m] -= (0.5) * ( gamma[annotatorIdx][classIdx] + delta[i][m][n] );
					log_phi[i][m] -= lambda(delta[i][m][n]) * ( Math.pow(gamma[annotatorIdx][classIdx], 2)  + mu[annotatorIdx][classIdx] - Math.pow(delta[i][m][n], 2) );
					
					log_zeta[i][m] += Math.log(logistic(rho[i][m][n]));
					log_zeta[i][m] -= theta[annotatorIdx][classIdx] * annotation;
					log_zeta[i][m] += (0.5) * ( theta[annotatorIdx][classIdx] - rho[i][m][n] );
					log_zeta[i][m] -= lambda(rho[i][m][n]) * ( Math.pow(theta[annotatorIdx][classIdx], 2)  + epsilon[annotatorIdx][classIdx] - Math.pow(rho[i][m][n], 2) );
				}
				
				//normalize
				double maxExp = log_phi[i][m] > log_zeta[i][m] ? log_phi[i][m] : log_zeta[i][m];
				double sumExp = Math.exp(log_phi[i][m] - maxExp) + Math.exp(log_zeta[i][m] - maxExp);
				
				phi[i][m] = Math.exp(log_phi[i][m] - maxExp - Math.log(sumExp));
				zeta[i][m] = Math.exp(log_zeta[i][m] - maxExp - Math.log(sumExp));
				
				//update sufficient statistics
				ss_lambda[classIdx] += phi[i][m];	
				ss_eta[classIdx] += zeta[i][m];
				for(int n=0; n<Nim[i][m]; n++)
				{
					int annotatorIdx = annotatorOfItemInterpretationValidation.get(new Triple(i, m, n));
					int annotation = itemInterpretationValidationsList.get(n);
					
					ss_mu[annotatorIdx][classIdx] += phi[i][m] * lambda(delta[i][m][n]);
					ss_gamma[annotatorIdx][classIdx] += phi[i][m] * (annotation - 0.5); 
					
					ss_epsilon[annotatorIdx][classIdx] += zeta[i][m] * lambda(rho[i][m][n]);
					ss_theta[annotatorIdx][classIdx] += zeta[i][m] * (0.5 - annotation); 
				}
			}
		}
		
		for(int j=0; j<J; j++)
		{
			double maxLogKappa = -Double.MAX_VALUE;
			for(int r=0; r<C; r++)
			{
				log_kappa[j][r] = 0.0;
				
				if(r != (C-1))
					log_kappa[j][r] += Gamma.digamma(f[r]) - Gamma.digamma(f[r] + g[r]);
				
				for(int r_prime = 0; r_prime < r; r_prime++)
					log_kappa[j][r] += Gamma.digamma(g[r_prime]) - Gamma.digamma(f[r_prime] + g[r_prime]);
				
				for(int h=0; h<K; h++)
				{
					double temp_alpha = (-0.5) * ( Math.log(2*Math.PI) + Math.log(sigma_beta[r][h]) - Gamma.digamma(sigma_alpha[r][h]) 
							+ (sigma_alpha[r][h]/sigma_beta[r][h]) *( Math.pow(gamma[j][h], 2) + mu[j][h] - 2.0 * gamma[j][h] * tau_mu[r][h] + Math.pow(tau_mu[r][h],2) + tau_sigma[r][h] ) ) ; 
					log_kappa[j][r] += temp_alpha;
					
					double temp_beta = (-0.5) * ( Math.log(2*Math.PI) + Math.log(omega_beta[r][h]) - Gamma.digamma(omega_alpha[r][h])
							+ (omega_alpha[r][h]/omega_beta[r][h]) * ( Math.pow(theta[j][h], 2) + epsilon[j][h] - 2.0 * theta[j][h] * nu_mu[r][h] + Math.pow(nu_mu[r][h],2) + nu_sigma[r][h] ) ) ; 
					log_kappa[j][r] += temp_beta;
				}
				
				if(log_kappa[j][r] > maxLogKappa)
					maxLogKappa = log_kappa[j][r];
			}
			
			double sumLogKappa = 0.0;
			for(int r=0; r<C; r++)
				sumLogKappa += Math.exp(log_kappa[j][r] - maxLogKappa);
			
			for(int r=0; r<C; r++)
			{
				double exponent = log_kappa[j][r] - maxLogKappa - Math.log(sumLogKappa);
				kappa[j][r] = Math.exp(exponent);
			}
			
			//update sufficient statistics
			for(int r=0; r<C; r++)
			{
				ss_indicator[r] += kappa[j][r];
				
				for(int r_prime = r+1; r_prime < C; r_prime++)
					ss_g[r] += kappa[j][r_prime];
			}
			
			for(int h=0; h<K; h++)
			{
				mu[j][h] = 2.0 * ss_mu[j][h];
				gamma[j][h] = ss_gamma[j][h];
				
				epsilon[j][h] =  2.0 * ss_epsilon[j][h];
				theta[j][h] = ss_theta[j][h];
				
				for(int r=0; r<C; r++)
				{
					mu[j][h] += kappa[j][r] * (sigma_alpha[r][h] / sigma_beta[r][h]);
					gamma[j][h] += kappa[j][r] * tau_mu[r][h] * (sigma_alpha[r][h] / sigma_beta[r][h]);
					
					epsilon[j][h] += kappa[j][r] * (omega_alpha[r][h] / omega_beta[r][h]);
					theta[j][h] += kappa[j][r] *  nu_mu[r][h] * (omega_alpha[r][h] / omega_beta[r][h]) ;
				}
				
				mu[j][h] = 1.0 / mu[j][h];
				gamma[j][h] = mu[j][h] * gamma[j][h];
				
				epsilon[j][h] = 1.0 / epsilon[j][h];
				theta[j][h] = epsilon[j][h]  * theta[j][h];
				
				//update sufficient statistics
				for(int r=0; r<C; r++)
				{
					ss_tau_mu[r][h] += kappa[j][r] * gamma[j][h];
					ss_nu_mu[r][h] += kappa[j][r] * theta[j][h];
					
					ss_sigma_beta[r][h] += kappa[j][r] * ( Math.pow(gamma[j][h], 2) + mu[j][h] - 2.0 * gamma[j][h] * tau_mu[r][h] + Math.pow(tau_mu[r][h],2) + tau_sigma[r][h] );
					ss_omega_beta[r][h] += kappa[j][r] * ( Math.pow(theta[j][h], 2) + epsilon[j][h] - 2.0 * theta[j][h] * nu_mu[r][h] + Math.pow(nu_mu[r][h],2) + nu_sigma[r][h] );
				}
			}
		}
		
		for(int h=0; h<K; h++)
		{
			lambda[h] = a0 + ss_lambda[h];
			eta[h] = a1 + ss_eta[h];
			
			lambdaPLUSeta[h] = lambda[h] + eta[h];
			digamma_lambda[h] = Gamma.digamma(lambda[h]);
			digamma_eta[h] = Gamma.digamma(eta[h]);
			digamma_lambdaPLUSeta[h] = Gamma.digamma(lambdaPLUSeta[h]);
		}
		
		for(int r=0; r<C; r++)
		{
			if(r != (C-1) )
			{
				f[r] = 1 + ss_indicator[r];
				g[r] = ( b_alpha / b_beta ) + ss_g[r];
				
				ss_b_beta += Gamma.digamma(g[r]) - Gamma.digamma(f[r] + g[r]);
			}
			
			for(int h=0; h<K; h++)
			{
				sigma_alpha[r][h] = e0 + ss_indicator[r] / 2.0;
				sigma_beta[r][h] = e1 + 0.5 * ss_sigma_beta[r][h];
				
				tau_sigma[r][h] = 1 / ( 1/d1 + ss_indicator[r] * sigma_alpha[r][h]/sigma_beta[r][h] );
				tau_mu[r][h] = tau_sigma[r][h] * ( d0/d1 + (sigma_alpha[r][h]/sigma_beta[r][h]) * ss_tau_mu[r][h] ) ;
				
				omega_alpha[r][h] = u0 + ss_indicator[r] / 2.0;
				omega_beta[r][h] = u1 + 0.5 * ss_omega_beta[r][h];
				
				nu_sigma[r][h] = 1 / ( 1/t1 + ss_indicator[r] * omega_alpha[r][h]/omega_beta[r][h] );
				nu_mu[r][h] = nu_sigma[r][h] * ( t0/t1 + (omega_alpha[r][h]/omega_beta[r][h]) * ss_nu_mu[r][h] );
			}
		}
		
		b_alpha = s0 + C - 1;
		b_beta = s1 - ss_b_beta;
	}
	
	
	private double[] GenerateRandomDirichletSample(RandomGenerator random, double alpha, int size)
	{
		GammaDistribution gammaDistribution = new GammaDistribution(random, alpha, 1);
		double[] gammaSamples = gammaDistribution.sample(size);
		
		double sum = 0.0;
		for(double gammaSample : gammaSamples)
			sum += gammaSample;
		
		double[] dirichletSample = new double[size];
		for(int sample = 0; sample < size; sample ++)
			dirichletSample[sample] = gammaSamples[sample] / sum;
		
		return dirichletSample;
	}
	
	private void CustomInitialization(String profilePath) 
	{
		try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(profilePath)));
			
			String fileLine = reader.readLine();
			fileLine = reader.readLine();
			while(fileLine != null)
			{
				String[] data = fileLine.split(",");
				
				String annotatorID = data[0];
				int j = annotators.indexOf(annotatorID);
				String classID = data[1];
				int h = classes.indexOf(classID);
				Double specificity = Double.parseDouble(data[2]);
				Double sensitivity = Double.parseDouble(data[3]);

				specificity = logit(specificity);
				sensitivity = logit(sensitivity);
				
				gamma[j][h] = sensitivity;
				mu[j][h] = 1;
				
				theta[j][h] = specificity;
				epsilon[j][h] = 1;
				
				fileLine = reader.readLine();
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		RandomGenerator random = new JDKRandomGenerator();
		random.setSeed(RANDOM_SEED);
		 
		 for(int j=0; j<J; j++)
		 {
			 kappa[j] = GenerateRandomDirichletSample(random, 1, C); 
		 }
		
		//-----------reset sufficient statistics-----------------
				ss_lambda = new double[K];
				ss_eta = new double[K];
				
				ss_gamma = new double[J][K];
				ss_mu = new double[J][K];
				
				ss_theta = new double[J][K];
				ss_epsilon = new double[J][K];
				
				ss_tau_mu = new double[C][K];
				ss_nu_mu = new double[C][K];
				
				ss_sigma_beta = new double[C][K];
				ss_omega_beta = new double[C][K];
				
				ss_indicator = new double[C];
				
				ss_b_beta = 0.0;
				
				ss_g = new double[C]; //we actually need C-1; last remains unused
				//-------------------------------------------------------
				
				for(int i=0; i<I; i++)
				{
					for(int m=0; m<Mi[i]; m++)
					{
						int classIdx = itemInterpretationClass.get(new Pair(i,m)); 
						List<Integer> itemInterpretationValidationsList = itemInterpretationValidations.get(new Pair(i, m));
						
						log_phi[i][m] = 0.0;
						log_zeta[i][m] = 0.0;
						
						for(int n=0; n<Nim[i][m]; n++)
						{
							int annotatorIdx = annotatorOfItemInterpretationValidation.get(new Triple(i, m, n));
							int annotation = itemInterpretationValidationsList.get(n);
							
							delta[i][m][n] = Math.pow(gamma[annotatorIdx][classIdx], 2) + mu[annotatorIdx][classIdx];
							delta[i][m][n] = Math.sqrt(delta[i][m][n]);

							rho[i][m][n] = Math.pow(theta[annotatorIdx][classIdx], 2) + epsilon[annotatorIdx][classIdx];
							rho[i][m][n] = Math.sqrt(rho[i][m][n]);
							
							log_phi[i][m] += Math.log(logistic(delta[i][m][n]));
							log_phi[i][m] += gamma[annotatorIdx][classIdx] * annotation;
							log_phi[i][m] -= (0.5) * ( gamma[annotatorIdx][classIdx] + delta[i][m][n] );
							log_phi[i][m] -= lambda(delta[i][m][n]) * ( Math.pow(gamma[annotatorIdx][classIdx], 2)  + mu[annotatorIdx][classIdx] - Math.pow(delta[i][m][n], 2) );
							
							log_zeta[i][m] += Math.log(logistic(rho[i][m][n]));
							log_zeta[i][m] -= theta[annotatorIdx][classIdx] * annotation;
							log_zeta[i][m] += (0.5) * ( theta[annotatorIdx][classIdx] - rho[i][m][n] );
							log_zeta[i][m] -= lambda(rho[i][m][n]) * ( Math.pow(theta[annotatorIdx][classIdx], 2)  + epsilon[annotatorIdx][classIdx] - Math.pow(rho[i][m][n], 2) );
						}
						
						//normalize
						double maxExp = log_phi[i][m] > log_zeta[i][m] ? log_phi[i][m] : log_zeta[i][m];
						double sumExp = Math.exp(log_phi[i][m] - maxExp) + Math.exp(log_zeta[i][m] - maxExp);
						
						phi[i][m] = Math.exp(log_phi[i][m] - maxExp - Math.log(sumExp));
						zeta[i][m] = Math.exp(log_zeta[i][m] - maxExp - Math.log(sumExp));
						
						//update sufficient statistics
						ss_lambda[classIdx] += phi[i][m];	
						ss_eta[classIdx] += zeta[i][m];
						for(int n=0; n<Nim[i][m]; n++)
						{
							int annotatorIdx = annotatorOfItemInterpretationValidation.get(new Triple(i, m, n));
							int annotation = itemInterpretationValidationsList.get(n);
							
							ss_mu[annotatorIdx][classIdx] += phi[i][m] * lambda(delta[i][m][n]);
							ss_gamma[annotatorIdx][classIdx] += phi[i][m] * (annotation - 0.5);
							
							ss_epsilon[annotatorIdx][classIdx] += zeta[i][m] * lambda(rho[i][m][n]);
							ss_theta[annotatorIdx][classIdx] += zeta[i][m] * (0.5 - annotation);
						}
					}
				}
				
				for(int j=0; j<J; j++)
				{	
					//update sufficient statistics
					for(int r=0; r<C; r++)
					{
						ss_indicator[r] += kappa[j][r];
						
						for(int r_prime = r+1; r_prime < C; r_prime++)
							ss_g[r] += kappa[j][r_prime];
					}
					
					for(int h=0; h<K; h++)
					{
						//update sufficient statistics
						for(int r=0; r<C; r++)
						{
							ss_tau_mu[r][h] += kappa[j][r] * gamma[j][h];
							ss_nu_mu[r][h] += kappa[j][r] * theta[j][h];
							
							ss_sigma_beta[r][h] += kappa[j][r] * ( Math.pow(gamma[j][h], 2) + mu[j][h] - 2.0 * gamma[j][h] * tau_mu[r][h] + Math.pow(tau_mu[r][h],2) + tau_sigma[r][h] );
							ss_omega_beta[r][h] += kappa[j][r] * ( Math.pow(theta[j][h], 2) + epsilon[j][h] - 2.0 * theta[j][h] * nu_mu[r][h] + Math.pow(nu_mu[r][h],2) + nu_sigma[r][h] );
						}
					}
				}
				
				for(int h=0; h<K; h++)
				{
					lambda[h] = a0 + ss_lambda[h];
					eta[h] = a1 + ss_eta[h];
					
					lambdaPLUSeta[h] = lambda[h] + eta[h];
					digamma_lambda[h] = Gamma.digamma(lambda[h]);
					digamma_eta[h] = Gamma.digamma(eta[h]);
					digamma_lambdaPLUSeta[h] = Gamma.digamma(lambdaPLUSeta[h]);
				}
				
				for(int r=0; r<C; r++)
				{
					if(r != (C-1) )
					{
						f[r] = 1 + ss_indicator[r];
						g[r] = s0 / s1 + ss_g[r];
						
						ss_b_beta += Gamma.digamma(g[r]) - Gamma.digamma(f[r] + g[r]);
					}
					
					for(int h=0; h<K; h++)
					{
						sigma_alpha[r][h] = e0 + ss_indicator[r] / 2.0;
						sigma_beta[r][h] = e1 + 0.5 * ss_sigma_beta[r][h];
						
						tau_sigma[r][h] = 1 / ( 1/d1 + ss_indicator[r] * sigma_alpha[r][h]/sigma_beta[r][h] );
						tau_mu[r][h] = tau_sigma[r][h] * ( d0/d1 + (sigma_alpha[r][h]/sigma_beta[r][h]) * ss_tau_mu[r][h] ) ;
						
						omega_alpha[r][h] = u0 + ss_indicator[r] / 2.0;
						omega_beta[r][h] = u1 + 0.5 * ss_omega_beta[r][h];
						
						nu_sigma[r][h] = 1 / ( 1/t1 + ss_indicator[r] * omega_alpha[r][h]/omega_beta[r][h] );
						nu_mu[r][h] = nu_sigma[r][h] * ( t0/t1 + (omega_alpha[r][h]/omega_beta[r][h]) * ss_nu_mu[r][h] );
					}
				}
				
				b_alpha = s0 + C - 1;
				b_beta = s1 - ss_b_beta;
				
	}
	
	private double MeasureStick(int r)
	{
		double res = ( r != (C-1) ) ? f[r] / (f[r] + g[r]) : 1.0;
		
		for(int r_prime = 0;  r_prime < r; r_prime++)
			res = res * g[r_prime] / (f[r_prime] + g[r_prime]) ;
		
		return res;
	}
	
	private void InitializeParameters(String profilePath)
	{
		//flat hyper-parameters
		a0 = a1 = 1.0;
		
		s0 = s1 = 1.0;
		
		d0 = 0;
		d1 = 1;
		t0 = 0;
		t1 = 1;
		
		e0 = 5; 
		e1 = 1;
		u0 = 5;
		u1 = 1;

		CustomInitialization(profilePath);
	}
	
	private double logistic(double x)
	{
		double z = 0.0;
		if(x >= 0)
		{
			z = Math.exp(-x);
			return 1.0 / (1.0 + z);
		}
		else
		{
			z = Math.exp(x);
			return z / (1.0 + z);
		}
	}
	
	private double logit(double x)
	{
		return Math.log(x) - Math.log(1-x);
	}
	
	private double lambda(double x)
	{
		return ( 0.5 / x ) * ( logistic(x) - 0.5) ;
	}
	
	private boolean CheckConvergence(double lowerBoundOLD, double lowerBoundNEW) 
	{
		if(lowerBoundOLD>lowerBoundNEW)
			System.out.println(RANDOM_SEED + " - Non-increasing lowerbound warning");

		return Math.abs(lowerBoundNEW-lowerBoundOLD) < convergence_threshold;
	}
	
	private Map<Integer, Integer> GetMajorityVoting(Random rand) 
	{
		Map<Integer, Integer> majorityVoting = new HashMap<Integer, Integer>();
		
		for(int i=0; i<I; i++)
		{
			Map<Integer, Integer> majorityVote = new HashMap<Integer, Integer>();
			
			for(int m=0; m<Mi[i]; m++)
			{
				List<Integer> validations = itemInterpretationValidations.get(new Pair(i,m));
				int score = 0;
				for(Integer validation : validations)
					score += validation == 1 ? 1 : -1;
				
				majorityVote.put(m, score);
			}
			
			List<Integer> votes = new ArrayList<Integer>();
			votes.addAll(majorityVote.values());
			
			int maxVotes = Collections.max(votes);
			
			List<Integer> classesWithMV = new ArrayList<Integer>();
			for(int classIndex : majorityVote.keySet())
			{
				int classVote = majorityVote.get(classIndex);
				if(classVote == maxVotes)
					classesWithMV.add(classIndex);
			}
			
			int size = classesWithMV.size();
			int randomClass = classesWithMV.get(rand.nextInt(size));
			
			majorityVoting.put(i, randomClass);
		}
		
		return majorityVoting;
	}
	
	private double ComputeAccuracy(Map<Integer, Integer> results) 
	{
		double matches = 0.0;
		for(String itemID : itemGoldStandard.keySet())
		{
			int i = items.indexOf(itemID);
			
			List<String> itemInterpretationsList = itemInterpretations.get(itemID);
			
			String silverLabel = itemInterpretationsList.get(results.get(i));
			String goldLabel = itemGoldStandard.get(itemID);
			
			if(silverLabel.equals(goldLabel))
				matches ++;
		}
		matches = matches / itemGoldStandard.keySet().size();
		return matches;
	}
	
	private Map<Integer, Integer> GetInferredLabels() 
	{
		Map<Integer, Integer> result = new HashMap<Integer, Integer>();
		for(int i=0; i<I; i++)
		{
			double maxProb = 0.0;
			int maxProbIndex = -1;
			
			for(int m=0; m<Mi[i]; m++)
			{
				if(phi[i][m] >= maxProb)
				{
					maxProb = phi[i][m];
					maxProbIndex = m;
				}
			}
			result.put(i, maxProbIndex);
		}
		
		return result;
	}
	
	private void InitializeParameterSpace() 
	{
		C = 10;
		
		phi = new double[I][];
		zeta = new double[I][];
		log_phi = new double[I][];
		log_zeta = new double[I][];
		log_kappa = new double[J][C];
		
		delta = new double[I][][];
		rho = new double[I][][];
		
		Mi = new int[I];
		Nim = new int[I][];
		
		for(int i=0; i<I; i++)
		{
			String itemID = items.get(i);
			Mi[i] = itemInterpretations.get(itemID).size();
			Nim[i] = new int[ Mi[i] ] ;
			
			delta[i] = new double[Mi[i]][];
			rho[i] = new double[Mi[i]][];
			for(int m=0; m<Mi[i]; m++)
			{
				Nim[i][m] = itemInterpretationValidations.get(new Pair(i, m)).size();
				
				delta[i][m] = new double[Nim[i][m]];
				rho[i][m] = new double[Nim[i][m]];
			}
			
			phi[i] = new double[Mi[i]];
			log_phi[i] = new double[Mi[i]];
			
			zeta[i] = new double[Mi[i]];
			log_zeta[i] = new double[Mi[i]];
		}
		
		kappa = new double[J][C];
		f = new double[C-1];
		g = new double[C-1];
		
		lambda = new double[K];
		eta = new double[K];
		lambdaPLUSeta = new double[K];
		digamma_lambda = new double[K];
		digamma_eta = new double[K];
		digamma_lambdaPLUSeta = new double[K];
		
		gamma = new double[J][K];
		mu = new double[J][K];

		theta = new double[J][K];
		epsilon = new double[J][K];
		
		tau_mu = new double[C][K];
		tau_sigma = new double[C][K];
		
		nu_mu = new double[C][K];
		nu_sigma = new double[C][K];
		
		sigma_alpha = new double[C][K];
		sigma_beta = new double[C][K];
		
		omega_alpha = new double[C][K];
		omega_beta = new double[C][K];
	}
	
	private void TransformAnnotationsIntoBinaryDecisions() 
	{
		annotatorOfItemInterpretationValidation = new HashMap<Triple, Integer>();
		itemInterpretationClass = new HashMap<Pair, Integer>();
		classes = new ArrayList<String>();
		itemInterpretationValidations = new HashMap<Pair, List<Integer>>();
		
		System.out.println("processing the annotations...");
		
		for(int i=0; i<I; i++)
		{
			if(i%1000 == 0) System.out.println("processed items: " + i);
			
			String item = items.get(i);
			List<String> itemInterpretationsList = itemInterpretations.get(item);
			List<Integer> itemAnnotationsList = itemAnnotations.get(i);
			
			int Mi = itemInterpretationsList.size();
			int Nim = itemAnnotationsList.size();
			
			for(int m=0; m<Mi; m++)
			{
				List<Integer> itemInterpretationValidationsList = new ArrayList<Integer>(); 
				
				String itemInterpretation = itemInterpretationsList.get(m);
				String interpretationClass = itemInterpretation.substring(0, itemInterpretation.indexOf("(")); //this is the class of the label
				
				if(!classes.contains(interpretationClass))
					classes.add(interpretationClass);
				
//				String itemClassContext = itemClass;
				int itemInterpretationClassIdx = classes.indexOf(interpretationClass); 
				
				itemInterpretationClass.put(new Pair(i,m), itemInterpretationClassIdx);
				
				for(int n=0; n<Nim; n++)
				{
					int annotationIdx = itemAnnotationsList.get(n);
					int annotatorIndex = annotatorOfItemAnnotation.get(new Pair(i, n));
					
					String annotationString = itemInterpretationsList.get(annotationIdx);
					
					int y_imn = annotationString.equals(itemInterpretation) ? 1 : 0;
					itemInterpretationValidationsList.add(y_imn);
					
					annotatorOfItemInterpretationValidation.put(new Triple(i, m, itemInterpretationValidationsList.size() - 1), annotatorIndex);
				}
				
				Pair itemClassPair = new Pair(i, m);
				itemInterpretationValidations.put(itemClassPair, itemInterpretationValidationsList);
			}
		}
		K = classes.size();
	}
	
	private void LoadAnnotations(String path) 
	{
		annotators = new ArrayList<String>();
		items = new ArrayList<String>();
		itemInterpretations = new HashMap<String, List<String>>();	
		itemAnnotations = new HashMap<Integer, List<Integer>>();
		annotatorOfItemAnnotation = new HashMap<Pair, Integer>();
		itemGoldStandard = new HashMap<String, String>();
		itemDocument = new HashMap<String, String>();
		int lineCounter = 0;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
			String fileLine = reader.readLine();
			fileLine = reader.readLine(); //skip header
			while(fileLine != null)
			{
				lineCounter++;
				if(lineCounter%10000 == 0)
					System.out.println("processed lines: " + lineCounter);
				
				String[] data = fileLine.split(","); 
				
				String documentName = "same";
				String item = data[0];
				String annotator = data[1];
				String responseClass = data[3];
				String gold = data[2];
				
				if(!annotators.contains(annotator))
					annotators.add(annotator);
						
				if(!items.contains(item))
					items.add(item);
						
				List<String> iClasses = itemInterpretations.containsKey(item) ? itemInterpretations.get(item) : new ArrayList<String>();
				if(!iClasses.contains(responseClass))
					iClasses.add(responseClass);
				itemInterpretations.put(item, iClasses);
					
				int annotatorIndex = annotators.indexOf(annotator);
				int itemIndex = items.indexOf(item);
				int responseIndex = iClasses.indexOf(responseClass);
					
				List<Integer> itemAnt = itemAnnotations.containsKey(itemIndex) ? itemAnnotations.get(itemIndex) : new ArrayList<Integer>();
				itemAnt.add(responseIndex);
				itemAnnotations.put(itemIndex, itemAnt);
						
				annotatorOfItemAnnotation.put(new Pair(itemIndex, itemAnt.size() - 1), annotatorIndex);
				if(!gold.equals("none"))
					itemGoldStandard.put(item, gold);
					
				itemDocument.put(item, documentName);
				
				fileLine = reader.readLine();
			}
			
			reader.close();
			
			I = items.size();
			J = annotators.size();
			
			TransformAnnotationsIntoBinaryDecisions();
		
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
}
