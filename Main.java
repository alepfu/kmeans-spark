package at.alepfu;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import scala.Tuple2;

@SuppressWarnings("serial")
public class Main {
	
	/**
	 * The path where the logfile and plot are saved to.
	 */
	private final static String outputPath = "/home/alepfu/testing/music";
	
	/**
	 * The path to the track data file, e.g. trackData2.txt.
	 */
	private final static String trackDataPath = "/home/alepfu/testing/music/trackData2.txt";
	
	/**
	 * The path to the rating data file, e.g. testIdx2.txt.
	 */
	private final static String ratingDataPath = "/home/alepfu/testing/music/trainIdx2.txt";
	
	/**
	 * The number of threads Spark should use.
	 */
	private final static String master = "local[4]";
	
	/**
	 * The number of ratings a track must have to be taken into account.
	 */
	private final static int trackCountCutOff = 100;
	
	/**
	 * The number of cluster KMeans should use.
	 */
	private final static int numClusters = 4; //2;
	
	/**
	 * The number of iterations for the KMeans algorithm.
	 */
	private final static int maxIterations = 20;
	
	/**
	 * The number of runs that KMeans should be executed.
	 */
	private final static int numRuns = 5;
	
	/**
	 * The type of initialization KMeans should use.
	 */
	private final static String initMode = KMeans.K_MEANS_PARALLEL();  //KMeans.RANDOM();
	
	/**
	 * Main method: initializes logging and SparkContext, loads data from text files, performs 
	 * the MapReduce and clustering tasks needed for this assignment, handles runtime measuring
	 * and plotting of the clustering results. 
     *
	 * @param args Arguments are not used here, for testing configure the static members above.
	 */
    public static void main(String[] args) throws Exception {
    	
    	long startTime = System.currentTimeMillis();
    	
    	PrintWriter log = new PrintWriter(outputPath + "/log_" + startTime + ".txt");
    	log.println("Number of threads = " + master);
    	log.println("Track count cutoff = " + trackCountCutOff);
    	
    	SparkConf conf = new SparkConf().setAppName("Music").setMaster(master);
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Load data
        JavaPairRDD<String, String> trackAlbum = loadTrackData(sc, trackDataPath);
        JavaPairRDD<String, Integer> trackRating = loadRatingData(sc, ratingDataPath);
        
        log.println("Number of ratings = " + trackRating.count());
        
        // Calculate average track and albumg ratings
        JavaPairRDD<String, Double> avgTrackRating = calcAverageTrackRating(trackRating);
        JavaPairRDD<String, Integer> albumRating = JavaPairRDD.fromJavaRDD(trackAlbum.join(trackRating).values());
        JavaPairRDD<String, Double> avgAlbumRating = calcAverageAlbumRating(albumRating);
       
        // Join average values together
        JavaPairRDD<String, Double> join1 = JavaPairRDD.fromJavaRDD(trackAlbum.join(avgTrackRating).values());
        JavaPairRDD<Double, Double> join2 = JavaPairRDD.fromJavaRDD(join1.join(avgAlbumRating).values());
        
		JavaRDD<Vector> vector = join2.map(new Function<Tuple2<Double, Double>, Vector>() {
			public Vector call(Tuple2<Double, Double> t) {
				double[] values = new double[2];
				values[0] = t._1;
				values[1] = t._2;
				return Vectors.dense(values);
			}
		});
        
		// Cluster and analyze the data using KMeans
		KMeansModel model = KMeans.train(vector.rdd(), numClusters, maxIterations, numRuns, initMode);
		
		log.println("\nNumber of clusters = " + numClusters);
		log.println("Max. iterations = " + maxIterations);
		log.println("Number of runs = " + numRuns);
		log.println("Init. mode = " + initMode);
        log.println("Number of points = " + vector.count());
        log.println("\nCost = " + model.computeCost(vector.rdd()));
        log.println("Centroids:");
        for (Vector c : model.clusterCenters())
        	log.println(c.toString());

        JavaRDD<Integer> clusterIndices = model.predict(vector);
        Map<Integer, Long> numPointsCluster = clusterIndices.countByValue();        
        log.println("Number of points per cluster:");
        numPointsCluster.forEach((k, v) -> log.println(k + ", " + v));
       
        log.println("\nPre-plotting execution time = " + (System.currentTimeMillis() - startTime) + " ms");
        
        // Plotting
        List<List<Vector>> clusters = new ArrayList<List<Vector>>();
        for (int i = 0; i < numClusters; i++)
        	clusters.add(new ArrayList<Vector>());
       
        for (Vector p : vector.collect())
        	clusters.get(model.predict(p)).add(p);
        
        ScatterPlot plot = new ScatterPlot(clusters);
        plot.save(outputPath + "/plot_" + startTime + ".jpeg");
        
        sc.stop();
        sc.close();

        log.println("\nTotal execution time = " + (System.currentTimeMillis() - startTime) + " ms");
        log.close();
    }
   
    /**
     * Loads track data from text file, filters useless rows (id = None).
     *
     * @param sc The spark context.
     * @param file A string with full path and filename.
     * @return A pair rdd holding the track and album id.
     *
     */
    private static JavaPairRDD<String, String> loadTrackData(JavaSparkContext sc, String file) {
       
        JavaPairRDD<String, String> track = sc.textFile(file)
                .mapToPair(new PairFunction<String, String, String>() {
                    public Tuple2<String, String> call(String line) {
                        String[] fields = line.split("\\|");
                        return new Tuple2<String, String>(fields[0], fields[1]);
                    }
                })
                .filter(a -> !"None".equals(a._1))
                .filter(a -> !"None".equals(a._2));
       
        return track;
    }
   
    /**
     * Loads rating data from text file, filters empty (rating = -1) and useless rows.
     *
     * @param sc The spark context.
     * @param file A string with full path and filename.
     * @return A pair rdd holding the track id and rating value.
     *
     */
    private static JavaPairRDD<String, Integer> loadRatingData(JavaSparkContext sc, String file) {
       
        JavaPairRDD<String, Integer> rating = sc.textFile(file).mapToPair(
                new PairFunction<String, String, Integer>() {
                    public Tuple2<String, Integer> call(String line) {
                        if (line.indexOf("\t") >= 0) {
                            String[] fields = line.split("\t");
                            return new Tuple2<String, Integer>(fields[0], Integer.parseInt(fields[1]));
                        }
                        else
                            return new Tuple2<String, Integer>(null, null);
                }})
                .filter(a -> a._1 != null)
                .filter(a -> a._2 != null)
                .filter(a -> -1 != a._2.intValue());
       
        return rating;
    }
   
    /**
     * Calculates the average rating per track, filters tracks with too 
     * few ratings (configure via static variable trackCountCutOff).
     *
     * @param rating A pair rdd holding track id and rating value.
     * @return A pair rdd holding the track id and average rating values.
     *
     */
    private static JavaPairRDD<String, Double> calcAverageTrackRating(JavaPairRDD<String, Integer> rating) {
       
        JavaPairRDD<String, Integer> count = rating.mapToPair(a -> new Tuple2<String, Integer>(a._1, 1))
            .reduceByKey((a, b) -> a + b)
            .filter(a -> a._2 >= trackCountCutOff);
        
        JavaPairRDD<String, Integer> sum = rating.reduceByKey((a, b) -> a+b);
       
        JavaPairRDD<String, Double> avg = count.join(sum)
                .mapToPair(t -> new Tuple2<String, Double>(t._1, (double)t._2._2 / t._2._1));
       
        return avg;
    }
   
    /**
     * Calculates the average rating per album.
     *
     * @param albumRating A pair rdd holding the album id and rating values.
     * @return A pair rdd holding the album id and average rating values.
     *
     */
    private static JavaPairRDD<String, Double> calcAverageAlbumRating(JavaPairRDD<String, Integer> albumRating) {
       
        JavaPairRDD<String, Integer> count = albumRating.mapToPair(a -> new Tuple2<String, Integer>(a._1, 1))
                .reduceByKey((a, b) -> a + b);
     
        JavaPairRDD<String, Integer> sum = albumRating.reduceByKey((a, b) -> a + b);
       
        JavaPairRDD<String, Double> avg = count.join(sum)
                .mapToPair(t -> new Tuple2<String, Double>(t._1, (double)t._2._2 / t._2._1));
       
        return avg;
    }
}
