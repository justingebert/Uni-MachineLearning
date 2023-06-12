package de.htw.ml.data;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.List;

import org.jblas.FloatMatrix;

/**
 * There are a lot TODOs here. 
 * The class divides the german credit dataset into train and test data.
 * 
 * @author Nico Hezel
 */
public class CreditDataset implements Dataset {
	
	protected Random rnd = new Random(7);
	
	protected FloatMatrix xTrain;
	protected FloatMatrix yTrain;
	
	protected FloatMatrix xTest;
	protected FloatMatrix yTest;
	
	protected int[] categories;
	
	public CreditDataset() throws IOException {
		
		int predictColumn = 15; // type of apartment
		FloatMatrix data = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		
		// List with all categories in the predictColumn
		final FloatMatrix outputData = data.getColumn(predictColumn);
		categories = IntStream.range(0, outputData.rows).map(idx -> (int)outputData.data[idx]).distinct().sorted().toArray();
		int[] categorySizes = IntStream.of(categories).map(v -> (int)outputData.eq(v).sum()).toArray();
		System.out.println("The unique values of y are "+ Arrays.toString(categories)+" and there number of occurrences are "+Arrays.toString(categorySizes));

		// Array with all rows that are not predictColumn
		int[] xColumns = IntStream.range(0, data.columns).filter(value -> value != predictColumn).toArray();

		// Input and output data
		FloatMatrix x = data.getColumns(xColumns);
		FloatMatrix y = data.getColumn(predictColumn);

		// min and maximum for all columns
		FloatMatrix xMin = x.columnMins();
		FloatMatrix xMax = x.columnMaxs();

		// normalize the data sets and add the bias column
		FloatMatrix xNorm = x.subRowVector(xMin).diviRowVector(xMax.sub(xMin));		
		xNorm = FloatMatrix.concatHorizontally(FloatMatrix.ones(xNorm.rows, 1), xNorm);
		
		// TODO create a training and test set with 90% and 10% of all data respectively
		// TODO the test set should contain images from all categories in equal amount
		int testDataPerCategory = data.getRows() / 10 / categories.length; // 10% test set
		int testDataCount = testDataPerCategory * categories.length;
		System.out.println("Use "+testDataCount+" as test data with "+testDataPerCategory+" elements per category.\n");
		
		// TODO replace these lines with the real train and test data
		//xTrain = xTest = xNorm;
		//yTrain = yTest = y;

		List<Integer> allIndices = new ArrayList<>();
		for (int i = 0; i < data.getRows(); i++) {
			allIndices.add(i);
		}

		List<Integer> testIndices = new ArrayList<>();

		// Add test data indices for each category
		for (int category : categories) {
			List<Integer> categoryIndices = new ArrayList<>();
			for (int i = 0; i < data.getRows(); i++) {
				if (y.get(i) == category) {
					categoryIndices.add(i);
				}
			}
			Collections.shuffle(categoryIndices, rnd);
			testIndices.addAll(categoryIndices.subList(0, testDataPerCategory));
		}

		Collections.shuffle(testIndices, rnd);

		// Get the test data

		int [] testIndicesArray = testIndices.stream().mapToInt(i -> i).toArray();

		xTest = xNorm.getRows(testIndicesArray);
		yTest = y.getRows(testIndicesArray);

		// Get the remaining data as training data
		List<Integer> trainIndices = new ArrayList<>(allIndices);
		trainIndices.removeAll(testIndices);

		int [] trainIndicesArray = trainIndices.stream().mapToInt(i -> i).toArray();
		xTrain = xNorm.getRows(trainIndicesArray);
		yTrain = y.getRows(trainIndicesArray);


	}
	
	public int[] getCategories() {
		return categories;
	}
	
	/**
	 * TODO Produce a subset. It contains all the test data but the y-values are binarized.
	 * The train data should contain as many train entries as possible but the ration 
	 * between data points of the desired category and data points of a different category
	 * should be 50:50. All Y data are binarized:
	 *  - desired category = 1
	 *  - other category = 0
	 * 
	 * @param category
	 * @return {x Matrix,y Matrix}
	 */
	public Dataset getSubset(int category) {
		
		// TODO Find all the indices of the lines in which the desired category occurs. 
		// Search as many other lines with a different category. Remove indices if
		// necessary, to ensure the size of both set are the same
		int[] rowIndizies = new int[] { 1 };

		List<Integer> rowIndices = new ArrayList<>();
		for (int i = 0; i < yTrain.length; i++) {
			if (yTrain.get(i) == category) {
				rowIndices.add(i);
			}
		}

		// Get the desired data points and binarize the Y-values
		FloatMatrix xTrainSubset = xTrain.getRows(rowIndices);
		FloatMatrix yTrainSubset = yTrain.getRows(rowIndices).eq(category);
		
		// Get the desired data points and binarize the Y-values
		return new Dataset() {
			
			@Override
			public FloatMatrix getXTrain() {
				return xTrain.getRows(rowIndizies);
			}
			
			@Override
			public FloatMatrix getYTrain() {
				return yTrain.getRows(rowIndizies).eq(category);
			}
			
			@Override
			public FloatMatrix getXTest() {
				return xTest;
			}
			
			@Override
			public FloatMatrix getYTest() {
				return yTest.eq(category);
			}
		};
	}

	@Override
	public FloatMatrix getXTrain() {
		return xTrain;
	}

	@Override
	public FloatMatrix getYTrain() {
		return yTrain;
	}

	@Override
	public FloatMatrix getXTest() {
		return xTest;
	}

	@Override
	public FloatMatrix getYTest() {
		return yTest;
	}
}
