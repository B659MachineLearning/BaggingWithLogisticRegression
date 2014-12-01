package com.machinelearning.LogisticRegression;

import java.util.ArrayList;

public class nFoldData {

	public static ArrayList<ArrayList<Example>> getFolds(int x, ArrayList<Example> examples, int numberofFolds) {
		int divSize = examples.size()/numberofFolds;
		int startIndex = (divSize*x); 
		int endIndex = ((divSize*x)+divSize-1);
		
		ArrayList<Example> ex = new ArrayList<>();
		ArrayList<Example> foldTest = new ArrayList<>();
		
		for(int m = startIndex; m<=endIndex; m++){
			foldTest.add(examples.get(m));
		}
		
		for(int m = 0; m<startIndex; m++){
			ex.add(examples.get(m));
		}
		for(int n = endIndex+1; n<examples.size(); n++){
			ex.add(examples.get(n));
		}
		ArrayList<ArrayList<Example>> lists = new ArrayList<ArrayList<Example>>();
		
		lists.add(ex);
		lists.add(foldTest);
		
		return lists;
	}
}
