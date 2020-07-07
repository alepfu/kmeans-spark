package at.alepfu;

import java.io.File;
import java.io.IOException;
import java.util.List;
import org.apache.spark.mllib.linalg.Vector;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class ScatterPlot {

	private XYSeriesCollection dataset;
	private JFreeChart chart;

	public ScatterPlot(List<List<Vector>>  clusters) {
		this.dataset = new XYSeriesCollection();
		int i = 0;
		for (List<Vector> cluster : clusters) {
			XYSeries series = new XYSeries(i++);
			for (Vector p : cluster)
				series.add(p.toArray()[0], p.toArray()[1]);
			this.dataset.addSeries(series);
		}
		this.chart = ChartFactory.createScatterPlot("Average ratings", "track", "album",
				this.dataset, PlotOrientation.VERTICAL, false, false, false);
	}

	public void show() {
		ChartFrame frame = new ChartFrame("", this.chart);
		frame.pack();
		frame.setVisible(true);
	}

	public void save(String filename) {
		int width = 660;
		int height = 420;
		try {
			ChartUtilities.saveChartAsJPEG(new File(filename), this.chart, width, height);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
