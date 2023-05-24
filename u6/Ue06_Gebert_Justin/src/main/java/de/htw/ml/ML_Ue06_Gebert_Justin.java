package de.htw.ml;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jblas.FloatMatrix;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;
import org.jblas.JavaBlas;
import org.jblas.ranges.IntervalRange;
import org.jblas.util.Random;

public class ML_Ue06_Gebert_Justin {

	// TODO change the names of the axis
	public static final String title = "Line Chart";
	public static final String xAxisLabel = "Iteration";
	public static final String yAxisLabel = "rmse";
	
	public static void main(String[] args) throws IOException {
//		FloatMatrix cars = FloatMatrix.loadCSVFile("cars_jblas.csv");
//		FloatMatrix cars_parameter = cars.getColumns(new IntervalRange(0, 5));
//		FloatMatrix cars_mpg = cars.getColumn(6);
//
//
//		FloatMatrix c_p_min = cars_parameter.columnMins();
//		FloatMatrix c_p_max = cars_parameter.columnMaxs();
//		FloatMatrix cars_parameter_nrm = cars_parameter.subRowVector(c_p_min).divRowVector(c_p_max.subRowVector(c_p_min));
//
//		FloatMatrix c_m_min = cars_mpg.columnMins();
//		FloatMatrix c_m_max = cars_mpg.columnMaxs();
//		FloatMatrix car_mpg_nrm = cars_mpg.subRowVector(c_m_min).divRowVector(c_m_max.subRowVector(c_m_min));
//
//
//		int iterations = 100;
//		float alpha = 0.1f;
//		Random.seed(7);
//
//		FloatMatrix theta = FloatMatrix.rand(cars_parameter_nrm.getColumns());
//		float [] rmse = new float[100];
//
//		for(int i = 0; i<iterations;i++){
//			float sum_of_abs_change = 0f;
//			System.out.println(theta.toString());
////			FloatMatrix h = cars_parameter_nrm.mulRowVector(theta);
////			h = h.rowSums();
//			FloatMatrix h = cars_parameter_nrm.mmul(theta);
//			FloatMatrix d = h.sub(car_mpg_nrm);
//
//			FloatMatrix h_dnrm = h.mmul(c_m_max.sub(c_m_min)).addRowVector(c_m_min);
//			float currentRMSE = Rmse(h_dnrm, cars_mpg);
//			System.out.println("RMSE: " + currentRMSE);
//			rmse[i] = currentRMSE;
//
//			FloatMatrix t_d = cars_parameter_nrm.transpose().mulRowVector(d).rowSums();
//			FloatMatrix t_d_nrm = t_d.mul((alpha/cars_parameter_nrm.getRows()));
//			theta = theta.sub(t_d_nrm);
//
//		}
//
//
//		// TODO implement your own code here -- SKALARPRODRUKT a .* b'
//		FloatMatrix column6 = cars.getColumn(6);
//		float[] yVals = rmse;

		FloatMatrix gc = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		FloatMatrix gc_p1 = gc.getColumns(new IntervalRange(0, 4));
		FloatMatrix gc_p2 = gc.getColumns(new IntervalRange(4, gc.getColumns()));

		FloatMatrix gc_p = FloatMatrix.concatHorizontally(gc_p1, gc_p2);
		FloatMatrix cA = gc.getColumn(5);


		FloatMatrix c_p_min = gc_p.columnMins();
		FloatMatrix c_p_max = gc_p.columnMaxs();
		FloatMatrix cars_parameter_nrm = gc_p.subRowVector(c_p_min).divRowVector(c_p_max.subRowVector(c_p_min));

		FloatMatrix c_m_min = cA.columnMins();
		FloatMatrix c_m_max = cA.columnMaxs();
		FloatMatrix car_mpg_nrm = cA.subRowVector(c_m_min).divRowVector(c_m_max.subRowVector(c_m_min));


		int iterations = 100;
		float alpha = 0.2f;
		Random.seed(7);

		FloatMatrix theta = FloatMatrix.rand(cars_parameter_nrm.getColumns());
		float [] rmse = new float[100];

		for(int i = 0; i<iterations;i++){
			float sum_of_abs_change = 0f;
			//System.out.println(theta.toString());
//			FloatMatrix h = cars_parameter_nrm.mulRowVector(theta);
//			h = h.rowSums();
			FloatMatrix h = cars_parameter_nrm.mmul(theta);
			FloatMatrix d = h.sub(car_mpg_nrm);

			FloatMatrix h_dnrm = h.mmul(c_m_max.sub(c_m_min)).addRowVector(c_m_min);
			float currentRMSE = Rmse(h_dnrm, cA);
			//System.out.println("RMSE: " + currentRMSE);
			rmse[i] = currentRMSE;

			FloatMatrix t_d = cars_parameter_nrm.transpose().mulRowVector(d).rowSums();
			FloatMatrix t_d_nrm = t_d.mul((alpha/cars_parameter_nrm.getRows()));
			theta = theta.sub(t_d_nrm);

		}


		// TODO implement your own code here -- SKALARPRODRUKT a .* b'
		FloatMatrix column6 = gc.getColumn(6);
		float[] yVals = rmse;

		System.out.println("best RMSE: " + rmse[rmse.length-1]);
		
		// plot the RMSE values
		FXApplication.plot(yVals, "Linear Regression");
		Application.launch(FXApplication.class);
	}


	public static float Rmse(FloatMatrix prediction, FloatMatrix original){
		FloatMatrix numorator = prediction.sub(original);
		FloatMatrix numorator2 = numorator;
		FloatMatrix num = numorator.mul(numorator2);

		int denominator = original.getRows();
		float sum = num.sum()/ denominator;
		return (float) Math.sqrt(sum);
	}
	
	

	// ---------------------------------------------------------------------------------
	// --------------- All changes from here on are at your own risk -------------------
	// ---------------------------------------------------------------------------------
	
	
	/**
	 * We need a separate class in order to trick Java 11 to start our JavaFX application without any module-path settings.
	 * https://stackoverflow.com/questions/52144931/how-to-add-javafx-runtime-to-eclipse-in-java-11/55300492#55300492
	 * 
	 * @author Nico Hezel
	 *
	 */
	public static class FXApplication extends Application {
	
		/**
		 * equivalent to linspace
		 * 
		 * @param lower
		 * @param upper
		 * @param num
		 * @return
		 */
		private static FloatMatrix linspace(float lower, float upper, int num) {
	        float[] data = new float[num];
	        float step = Math.abs(lower-upper) / (num-1);
	        for (int i = 0; i < num; i++)
	            data[i] = lower + (step * i);
	        data[0] = lower;
	        data[data.length-1] = upper;
	        return new FloatMatrix(data);
	    }
		
		// y-axis values of the plot 
		private static List<float[]> dataYList = new ArrayList<>();
		private static List<String> dataYNameList = new ArrayList<>();
		
		/**
		 * Remembers the values and the name of the data
		 * 
		 * @param yValues
		 * @param name
		 */
		public static void plot(float[] yValues, String name) {
			dataYList.add(yValues);
			dataYNameList.add(name);
		}
		
		/**
		 * draw the UI
		 */
		@Override 
		public void start(Stage stage) {
	
			stage.setTitle(title);
			
			final NumberAxis xAxis = new NumberAxis();
			xAxis.setLabel(xAxisLabel);
	        final NumberAxis yAxis = new NumberAxis();
	        yAxis.setLabel(yAxisLabel);
	        
			final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);
			sc.setAnimated(false);
			sc.setCreateSymbols(true);
	
			for (int s = 0; s < dataYList.size(); s++) {				
				XYChart.Series<Number, Number> series = new XYChart.Series<>();
				series.setName(dataYNameList.get(s));
				
				float[] dataY = dataYList.get(s);
				for (int i = 0; i < dataY.length; i++)
					series.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
				sc.getData().add(series);
			}	
	
			Scene scene = new Scene(sc, 500, 400);
			stage.setScene(scene);
			stage.show();
	    }
	}
}
