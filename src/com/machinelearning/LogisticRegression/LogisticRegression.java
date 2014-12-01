package com.machinelearning.LogisticRegression;
/*
 * Authors : Aniket Bhosale and Mayur Tare
 * Description : This class implements Logistic Regression algorithm using ABSCONV conversion criterion.
 */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;


public class LogisticRegression {
	
	public static double[] weights;
	static double step = Double.parseDouble(Config.readConfig("stepSize"));
	private static int numOfRuns = Integer.parseInt(Config.readConfig("iterations"));
	private static double conversionVal = Double.parseDouble(Config.readConfig("conversionValue"));
	
	public static String label = null;
	public static int labelIndex;
	public static String trueClassLabel = "1";
	
	//Calculate Probability using sigmoid function
	private static double sigmoid(double z){
	    return 1 / (1 + Math.exp(-z)); 	
	}
	
	//Return calculate (w.x)
	public static double classifier(Example instance){
		double linearSum = 0.0;
		for (int j = 0; j<DataLoader.numberOfFeatures; j++){
			if(j != labelIndex)
				linearSum += weights[j] * Double.parseDouble(instance.getFeature(j));
		}
		return sigmoid(linearSum);
	}
	
	public static void parameterComputation(ArrayList<Example> ex){
		double oldLikelihood = 0.0;
		double logLikelihood =0.0;
		for (int n = 0; n < numOfRuns; n++){
			double ABSFCONV = 0.0;
			
			for (int i=0; i < ex.size(); i++){
					double predictedValue = classifier(ex.get(i));
					String classLabel = ex.get(i).getFeature(labelIndex);
					if(!classLabel.equalsIgnoreCase(trueClassLabel))
						classLabel = "0";
					for (int j = 0; j<DataLoader.numberOfFeatures; j++){
						if(j != labelIndex){
							weights[j]= weights[j] + step * (Double.parseDouble(classLabel) - predictedValue) * Double.parseDouble(ex.get(i).getFeature(j));
						}
					}
					//Calculate Likelihood for current iteration
					logLikelihood += (1-Integer.parseInt(classLabel)) * Math.log(1-classifier(ex.get(i))) + Integer.parseInt(classLabel) * Math.log(classifier(ex.get(i)));
			}
			//ABSFCONV : Convergence requires a small change in the log-likelihood function in subsequent iterations 
			ABSFCONV = Math.abs(logLikelihood -  oldLikelihood);
		
			if(ABSFCONV < conversionVal){
				System.out.println("Converged after "+n+" iterations for conversion value(ABSCONV) = "+conversionVal);
				break;
			}
			oldLikelihood = logLikelihood;
		}        
	}

	public static  void main(String[] args) {
		
		double  predictedClass1=-1.0;
		double  predictedClass0=-1.0;
		int incorrectCount = 0;
		int correctCount = 0;
		int predictedClassLabel = -1;
		int actualClassLabel = -1;
		//Read the file name for training data from config file
		String trainFilePath = Config.readConfig("trainFileName");
		String testFilePath = Config.readConfig("testFileName");
		
		ArrayList<ArrayList<Example>> lists;
		int numberofFolds = 2;
		ArrayList<Example> Examples = DataLoader.readRecords(trainFilePath);
		
		double accuracy = 0.0;
		Random randomGenerator = new Random();
		
		for(int x=0; x<numberofFolds; x++){
			incorrectCount = 0;
			correctCount = 0;
			lists = nFoldData.getFolds(x, Examples, numberofFolds);
			
			ArrayList<Example> trainExamples_orig = (ArrayList<Example>) lists.get(0).clone();
			ArrayList<Example> testExamples = (ArrayList<Example>) lists.get(1).clone();
			
			for(int i=0; i<10; i++){
				
				ArrayList<Example> trainExamples = new ArrayList<>();
				//ArrayList<Example> testExamples = new ArrayList<>();
				
				for(int k = 0; k<trainExamples_orig.size(); k++){
					int randomIndex = randomGenerator.nextInt(trainExamples_orig.size()-1);
					trainExamples.add(trainExamples_orig.get(randomIndex));
				}
				
				//Index of the class label
				label = Config.readConfig("classLable");
				labelIndex = DataLoader.labels.indexOf(label);
						
				weights = new double [DataLoader.numberOfFeatures];
				
				//Train the algorithm of Train Data Set
				parameterComputation(trainExamples);
				
				HashMap<Integer, Double> probabilities1 = new HashMap<>();
				HashMap<Integer, Double> probabilities0 = new HashMap<>();
				
				//Check the algorithm predictions for Test Data set
				for(int e = 0; e < testExamples.size();e++){
					predictedClass1 =  classifier(testExamples.get(e));
					predictedClass0 = 1 - predictedClass1;
					
					if(probabilities1.containsKey(e)){
						double currProb = probabilities1.get(e);
						probabilities1.put(e, currProb + predictedClass1);
					}
					else{
						probabilities1.put(e, predictedClass1);
					}
					if(probabilities0.containsKey(e)){
						double currProb = probabilities0.get(e);
						probabilities0.put(e, currProb + predictedClass0);
					}
					else{
						probabilities0.put(e, predictedClass0);
					}
					//System.out.println("Win Probability : "+predictedClass1);
				}
				
				for(int e = 0; e < testExamples.size();e++){
					
					double AggClass1Prob = probabilities1.get(e)/10;
					double AggClass0Prob = probabilities0.get(e)/10;
					System.out.println(AggClass1Prob+" : "+AggClass0Prob);
					
					if(AggClass0Prob > AggClass1Prob)
						predictedClassLabel = 0;
					else
						predictedClassLabel = 1;
					
					if(Integer.parseInt(testExamples.get(e).getFeature(labelIndex)) != 1)
						actualClassLabel = 0;
					else
						actualClassLabel = 1;
					if(!(predictedClassLabel == actualClassLabel))
						incorrectCount++;
					else
						correctCount++;
					System.out.println("Predicted : "+predictedClassLabel+" Actual : "+actualClassLabel);
				}
			}
			System.out.println("Total Correct Predcitions = "+correctCount+" out of "+testExamples.size()+" examples");
			System.out.println("Total Incorrect Predcitions = "+incorrectCount+" out of "+testExamples.size()+" examples");
			
			accuracy += 100.0*correctCount/testExamples.size();
		}
		
		System.out.println("Accuracy : "+accuracy/numberofFolds);
		
	}

}
