/***
 * Md Maruf Hossain
 * mhossa10@uncc.edu
 **/

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.math.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

public class TFIDF extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(TFIDF.class);

    public static void main(String[] args) throws Exception {
        int res = ToolRunner.run(new TFIDF(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        /// Set the number of document in the input to calculate IDF from the fourth parameter
        conf.set("NUMBER_OF_FILE", args[3]);
        /// creating the first job for term frequency
        Job tf_job = Job.getInstance(conf, "termfrequency");
        tf_job.setJarByClass(this.getClass());
        /// Get the input file from the first parameter and add to the input path
        FileInputFormat.addInputPath(tf_job, new Path(args[0]));
        /// Get the output file location from the second parameter and add to the output path
        FileOutputFormat.setOutputPath(tf_job, new Path(args[1]));
        /// Add Mapper Class
        tf_job.setMapperClass(Map.class);
        /// Add Combiner Class
        tf_job.setCombinerClass(Combine.class);
        /// Add Reduce Class
        tf_job.setReducerClass(Reduce.class);
        /// Set intermediate output key as Text
        tf_job.setOutputKeyClass(Text.class);
        /// Set intermediate output value as Integer format
        tf_job.setOutputValueClass(IntWritable.class);
        int code = tf_job.waitForCompletion(true) ? 0 : 1;

        // TF-IDF
        /// creating the second job TFIDF
        Job tfidf_job = Job.getInstance(conf, "tfidf");
        tfidf_job.setJarByClass(this.getClass());
        /// Get the input file from the scond parameter that is the output of term frequency job
        FileInputFormat.addInputPath(tfidf_job, new Path(args[1]));
        /// Get the output file location from the third parameter and add to the output path
        FileOutputFormat.setOutputPath(tfidf_job, new Path(args[2]));
        /// Add Mapper Class
        tfidf_job.setMapperClass(TFIDFMap.class);

        /// Do not need combiner class

        /// Add Reduce Class
        tfidf_job.setReducerClass(TFIDFReduce.class);
        /// Set intermediate output key as Text
        tfidf_job.setOutputKeyClass(Text.class);
        /// Set intermediate output value as Text format
        tfidf_job.setOutputValueClass(Text.class);
        code = tfidf_job.waitForCompletion(true) ? 0 : 1;

        return code;
    }

    /// Map Class for Term Frequency
    public static class Map extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private boolean caseSensitive = false;
        /// Pattern to split the input line
        private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");
        private String input;

        /// Setup the input and case sensitivity
        protected void setup(Mapper.Context context)
                throws IOException,
                InterruptedException {
            if (context.getInputSplit() instanceof FileSplit) {
                this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
            } else {
                this.input = context.getInputSplit().toString();
            }
            /// Make case sensitive false for this project
            this.caseSensitive = false;
        }

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            /// Get the file name, because we need it for the output
            String filename = fileSplit.getPath().getName();
            /// get the input line as string and trim it.
            String line = lineText.toString();
            line = line.trim();
            if (!caseSensitive) {
                /// make it lower case
                line = line.toLowerCase();
            }
            Text currentWord = new Text();
            for (String word : WORD_BOUNDARY.split(line)) {
                word = word.trim();
                /// Check the input line is empty or does not conatain any alphabet
                if (word.isEmpty() || !word.matches(".*[a-zA-Z]+.*")) {
                    continue;
                }
                /// Create desired outout format
                currentWord = new Text(word + "#####" + filename);
                context.write(currentWord, one);
            }
        }
    }

    /// Map Class for TFIDF
    public static class TFIDFMap extends Mapper<LongWritable, Text, Text, Text> {
        private final static IntWritable one = new IntWritable(1);
        private boolean caseSensitive = false;
        /// Pattern to split the input line
        private static final Pattern WORD_BOUNDARY = Pattern.compile("#####");
        private String input;

        /// Setup the input and case sensitivity
        protected void setup(Mapper.Context context)
                throws IOException,
                InterruptedException {
            if (context.getInputSplit() instanceof FileSplit) {
                this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
            } else {
                this.input = context.getInputSplit().toString();
            }
            /// Make case sensitive false for this project
            this.caseSensitive = false;
        }

        public void map(LongWritable offset, Text lineText, Context context)
                throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            /// get the input line as string and trim it.
            String line = lineText.toString();
            line = line.trim();
            if (!caseSensitive) {
                /// make it lower case
                line = line.toLowerCase();
            }
            if (line.isEmpty()) {
                return;
            }

            /// Split the line by "#####"
            String[] tokens = WORD_BOUNDARY.split(line);
            if (tokens.length >= 2) {
                // context.write(new Text(tokens[0]), new Text(tokens[1]));
                String[] values = tokens[1].trim().split("\\s+");
                if (values.length >= 2) {
                    /// Create desired outout format
                    context.write(new Text(tokens[0]), new Text(values[0] + "=" + values[1]));
                }
            }
        }
    }

    /// Reduce class for Term Frequency
    public static class Reduce extends Reducer<Text, IntWritable, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<IntWritable> counts, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable count : counts) {
                sum += count.get();
            }
            double TF = 1.00 + Math.log10(sum);
            if (TF < 0) {
                TF = 0.0;
            }
            context.write(word, new DoubleWritable(TF));
        }
    }

    public static class TFIDFReduce extends Reducer<Text, Text, Text, DoubleWritable> {
        @Override
        public void reduce(Text word, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            /// Retrieve the number of documents in input
            long totalNumberOfDoc = Long.parseLong(context.getConfiguration().get("NUMBER_OF_FILE"));
            long numberOfDocContain = 0;
            /// Cache the values
            List<String> cache = new ArrayList<String>();
            for (Text val : values) {
                /// Find the number of files that containd the word
                numberOfDocContain++;
                cache.add(val.toString());
            }
            /// check the validation
            if (numberOfDocContain == 0)
                return;
            /// Calculate the TF-IDF
            double IDF = Math.log10(1 + (totalNumberOfDoc / numberOfDocContain));
            for (int i = 0; i < cache.size(); ++i) {
                /// Split the values by "="
                String[] tokens = cache.get(i).split("=");
                if (tokens.length >= 2) {
                    /// Write the output as desired output format
                    context.write(new Text(word + "#####" + tokens[0]), new DoubleWritable(Double.valueOf(tokens[1]) * IDF));
                }
            }
        }
    }

    /// Combiner class for Term Frequency
    public static class Combine extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text word, Iterable<IntWritable> counts, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable count : counts) {
                /// Sum all the count
                sum += count.get();
            }
            context.write(word, new IntWritable(sum));
        }
    }
}

