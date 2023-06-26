package de.htw.ml;

import org.jblas.FloatMatrix;

import static org.jblas.MatrixFunctions.log;

/**
 * A lot of TODOs here.
 * This is a simple logistic regression model.
 *  
 * @author Nico Hezel
 */
public class LogisticRegression {
	
	protected int trainingIterations;
	protected float learnRate;
	protected float[] predictionRates;
	protected float[] trainErrors;	
	
	public LogisticRegression(int trainingIterations, float learnRate) {
		this.trainingIterations = trainingIterations;
		this.learnRate = learnRate;
	}

	public FloatMatrix train(FloatMatrix xTest, FloatMatrix yTest, FloatMatrix xTrain, FloatMatrix yTrain) {
		this.predictionRates = new float[trainingIterations];
		this.trainErrors = new float[trainingIterations];
		
		// initialize the weights
		org.jblas.util.Random.seed(7);
		FloatMatrix theta = FloatMatrix.rand(xTrain.getColumns(), 1);
		
		// current training error
		trainErrors[0] = cost(predict(xTrain, theta), yTrain);

		// best combination of weights
		FloatMatrix bestTheta = theta.dup();
		float bestPredictionRate = predictionRates[0] = predictionRate(predict(xTest, theta), yTest);
		
		// training
		for (int iteration = 0; iteration < trainingIterations; iteration++) {
			
			// TODO training using the logistic regression
			FloatMatrix hypothesis = predict(xTrain, theta);
			FloatMatrix error = hypothesis.sub(yTrain);
			FloatMatrix gradient = xTrain.transpose().mmul(error);
			FloatMatrix gradient_norm = gradient.mul(learnRate / (float) yTrain.length);
			theta = theta.sub(gradient_norm);
			
			// TODO fill the prediction rate and train error arrays
			predictionRates[iteration] = predictionRate(predict(xTest,theta), yTest);
			trainErrors[iteration] = cost(hypothesis, yTrain);

			if (predictionRates[iteration] > bestPredictionRate) {
				bestPredictionRate = predictionRates[iteration];
				bestTheta = theta.dup();
			}
		}
		
		return bestTheta;
	}

	/**
	 * Calculates a prediction of the input data X and the current weights theta
	 * 
	 * @param x
	 * @param theta
	 * @return
	 */
	public static FloatMatrix predict(FloatMatrix x, FloatMatrix theta) {
		FloatMatrix z = x.mmul(theta);
		return sigmoid(z);
	}
		
	/**
	 * Calculates the training error according to the logistical cost function or RMSE.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float cost(FloatMatrix prediction, FloatMatrix y) {
		//Todo look up again
		int m = y.length;
		FloatMatrix cost = y.mul(-1).mul(log(prediction)).sub(y.mul(-1).add(1).mul(log(prediction.mul(-1).add(1))));
		return cost.sum() / m;
	}

	/**
	 * Calculates a prediction rate between the prediction and the desired result Y.
	 * 
	 * @param prediction
	 * @param y
	 * @return
	 */
	public static float predictionRate(FloatMatrix prediction, FloatMatrix y) {

		int m = y.length;
		int correctPredictions = 0;
		for (int i = 0; i < m; i++) {
			float predictedLabel = prediction.get(i);
			float trueLabel = y.get(i);


			if ((predictedLabel >= 0.5 && trueLabel == 1) || (predictedLabel < 0.5 && trueLabel == 0)) {
				correctPredictions++;
			}
		}

		return correctPredictions / (float) m * 100;
	}

	/**
	 * Prediction rates of the last training
	 * 
	 * @return
	 */
	public float[] getPredictionRates() {
		return predictionRates;
	}
	
	/**
	 * error rates of the last training
	 * 
	 * @return
	 */
	public float[] getTrainError() {
		return trainErrors;
	}
	
	/**
	 * Replaces the values in the Input Matrix with their sigmoid variant.
	 * 
	 * @param input
	 * @return
	 */
	public static FloatMatrix sigmoid(FloatMatrix input) {
		for (int i = 0; i < input.data.length; i++)
			input.data[i] = (float) (1. / ( 1. + Math.exp(-input.data[i]) ));
		return input;
	}
}