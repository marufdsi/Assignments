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
import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;

import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Pagerank extends Configured implements Tool {

    private static final Logger LOG = Logger.getLogger(Pagerank.class);

    public static void main(String[] args) throws Exception {
        /// XMLParser is used initial parsing of input file.
        XMLParser parseXML = new XMLParser();
        /// Pre-process the data and store them into new temporary file OutputFIle0
        parseXML.parseXMLFile(args[0], args[1]+"0");
        int res = ToolRunner.run(new Pagerank(), args);
        System.exit(res);
    }

    public int run(String[] args) throws Exception {
        int code = 0;
        /// Number of times want to perform pagerank algorithm.
        int loop_count = 10;
        for (int i=0; i<loop_count; ++i) {
            Configuration conf = getConf();
            FileSystem fs = FileSystem.get(conf);
            /// isFinalResult used to check the last iteration.
            conf.set("isFinalResult", i==(loop_count-1) ? "true":"false");
            /// save the output file to cleanup the temporary files.
            conf.set("OUTPUT_FILE", args[1]);
            /// isLimitedPageRank boolean variable is used to decide produce the full output or certain limit.
            boolean isLimitedPageRank = false;
            if (args.length>=3){
                isLimitedPageRank = true;
                /// parameter 3 is used to findout how many page rank a user wants in descending order.
                conf.set("GIVE_K_NUMBER_OF_PAGE_RANK", args[2]);
            }
            conf.set("isLimitedPageRank", String.valueOf(isLimitedPageRank));
            /// creating the job for pagerank
            Job job = Job.getInstance(conf, "pagerank");
            job.setJarByClass(this.getClass());
            /// Set the input file
            FileInputFormat.addInputPath(job, new Path(args[1]+String.valueOf(i)));
            /// Set the output file location
            Path outputFile = (i==(loop_count-1) ? new Path(args[1]) : new Path(args[1]+String.valueOf(i+1)));
            if (fs.exists(outputFile)) {
                LOG.info("Pagerank output file exist");
                fs.delete(outputFile, true);
            }
            FileOutputFormat.setOutputPath(job, outputFile);
            /// Add Mapper Class
            job.setMapperClass(Map.class);
            /// Add Reduce Class
            job.setReducerClass(Reduce.class);
            /// Set reducer task
            job.setNumReduceTasks(1);
            /// Set intermediate output key as Text
            job.setOutputKeyClass(Text.class);
            /// Set intermediate output value as Integer format
            job.setOutputValueClass(Text.class);
            code =  job.waitForCompletion(true) ? 0 : 1;
        }
        return code;
    }

    /// Mapper class
    public static class Map extends Mapper<LongWritable, Text, Text, Text> {
        private static final Logger MapperLog = Logger.getLogger(Map.class);
        private String input;
        protected void setup(Mapper.Context context) throws IOException, InterruptedException {
            if (context.getInputSplit() instanceof FileSplit) {
                this.input = ((FileSplit) context.getInputSplit()).getPath().toString();
            } else {
                this.input = context.getInputSplit().toString();
            }
        }
        /// Used custom pattern different link-pagerank
        private final Pattern WORD_BOUNDARY = Pattern.compile(":#:#:#");
        public void map(LongWritable offset, Text lineText, Context context) throws IOException, InterruptedException {
            FileSplit fileSplit = (FileSplit) context.getInputSplit();
            /// get the input line as string and trim it.
            String line = lineText.toString();
            line = line.trim();
            /// make it lower case
            line = line.toLowerCase();
            /// Split line into page and adjacent link
            String[] lineSegments = line.split("######");
            /// Validation check
            if (lineSegments.length<2)
                return;
            if (!lineSegments[1].trim().contains(":::=:::"))
                return;
            /// Emit link as key and adjacent list as value to use in the reduce class.
            context.write(new Text(lineSegments[0].trim()), new Text(lineSegments[1].trim()));
            for (String word : WORD_BOUNDARY.split(lineSegments[1])) {
                /// word contain a single element of adjacent list:(link:::=:::pagerank)
                word = word.trim();
                /// Check the input line is empty
                if (word.isEmpty()) {
                    continue;
                }
                /// Link and pagerank value separated by ":::=:::" pattern
                String[] link_value = word.trim().split(":::=:::");
                /// Create desired output format link as key and pagerank as value
                if (link_value.length>=2) {
                    context.write(new Text(link_value[0].trim()), new Text(link_value[1].trim()));
                }
            }
        }
    }

    /// Reducer class
    public static class Reduce extends Reducer<Text, Text, Text, Text> {
        private static final Logger ReducerLog = Logger.getLogger(Reduce.class);
        /// This variable is used in the final iteration to sort the pagerank.
        private java.util.Map<String , Double > pageRankData = new HashMap<String , Double>();
        @Override
        public void reduce(Text link, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double pagerank = 0;
            /// Dumping factor
            double d = 0.85;
            String[] adjacentListWithLink = null;
            for (Text value : values) {
                if(value.toString().contains(":::=:::")) {
                    /// This is the condition for key:link, value:adjacent list
                    adjacentListWithLink = value.toString().split(":#:#:#");
                } else {
                    /// This is the condition for key:link, value:pagerank
                    pagerank += Double.parseDouble(value.toString());
                }
            }
            /// PR(A) = (1-d) + d (PR(T1)/C(T1) + ... + PR(Tn)/C(Tn))
            pagerank = (1-d) + d*pagerank;
            Configuration conf = context.getConfiguration();
            boolean isFinalResult = Boolean.parseBoolean(conf.get("isFinalResult"));
            /// Check the final iteration
            if (isFinalResult){
                /// In final iteration save the pagerank in the pageRankData to process for the desired output
                pageRankData.put(link.toString(), pagerank);
            } else if(adjacentListWithLink !=null && adjacentListWithLink.length>0) {
                int count = adjacentListWithLink.length;
                if (count==0)
                    return;
                String newLinkRankString = "";
                /// This is the process to ready the data for the next iteration and output them.
                /// In the next iteration it's neighbor will use this value, that is why update the
                /// adjacent list by its pagerank/out_going_neighbor.
                for(int i = 0 ; i < adjacentListWithLink.length ; i++) {
                    String[] link_value = adjacentListWithLink[i].split(":::=:::");
                    if (link_value != null && link_value.length>=2){
                        double newRank = Double.parseDouble(link_value[1]) / count;
                        newLinkRankString += link_value[0]+":::=:::"+newRank+":#:#:#";
                    }
                }
                /// Remove the last ":#:#:#" string
                if (newLinkRankString.length()>6)
                    newLinkRankString = newLinkRankString.substring(0, newLinkRankString.length()-6);
                /// output as input format Link ###### Adjacent List
                context.write(link, new Text("######" + newLinkRankString));
            }
        }

        /// Cleanup method called at the last of Reduce. So, I perform cleanup and sorting in here.
        public void cleanup(Context context){
            try {
                //Here I am going to write the final output
                Configuration conf = context.getConfiguration();
                FileSystem fs = FileSystem.get(conf);
                boolean isFinalResult = Boolean.parseBoolean(conf.get("isFinalResult"));
                String OutputFile = conf.get("OUTPUT_FILE");
                /// Check final iteration
                if (isFinalResult) {
                    /// Delete all intermediate files if exist.
                    for (int i = 0; i < 10; ++i) {
                        if(fs.exists(new Path(OutputFile + String.valueOf(i)))) {
                            fs.delete(new Path(OutputFile + String.valueOf(i)), true);
                        }
                    }
                    /// Check limited pagerank or full.
                    boolean isLimitedPageRank = Boolean.parseBoolean(conf.get("isLimitedPageRank"));
                    int expectedNumberOfPageRank = 0;
                    if (isLimitedPageRank) {
                        expectedNumberOfPageRank = Integer.parseInt(conf.get("GIVE_K_NUMBER_OF_PAGE_RANK"));
                    }
                    /// Sort the pagerank
                    java.util.Map<String, Double> sortedMap = new HashMap<String, Double>();
                    sortedMap = sortMap(pageRankData, isFinalResult, expectedNumberOfPageRank);
                    /// Output the link and sorted pagerank
                    for (String link : sortedMap.keySet()) {
                        context.write(new Text(link), new Text(String.valueOf(sortedMap.get(link))));
                    }
                }
            }catch (Exception e){
                ReducerLog.error(e);
            }
        }
        /// Sorting method
        public java.util.Map<String , Double > sortMap (java.util.Map<String,Double> unsortMap, boolean isLimitedPageRank, int k){
            List<String> mapKeys = new ArrayList<>(unsortMap.keySet());
            List<Double> mapValues = new ArrayList<>(unsortMap.values());
            /// Sort Descending order
            Collections.sort(mapValues, Collections.reverseOrder());

            int count = 1;

            java.util.Map<String ,Double> sortedMap = new LinkedHashMap<String, Double>();
            Iterator<Double> valueIt = mapValues.iterator();
            while (valueIt.hasNext()) {
                if (isLimitedPageRank && count>k)
                    break;
                Double val = valueIt.next();
                Iterator<String> keyIt = mapKeys.iterator();
                while (keyIt.hasNext()) {
                    String key = keyIt.next();
                    Double comp1 = unsortMap.get(key);
                    Double comp2 = val;

                    if (comp1.equals(comp2)) {
                        keyIt.remove();
                        sortedMap.put(key, val);
                        break;
                    }
                }
                count++;
            }

            return sortedMap ;
        }
    }

    /// Util class to parse the initial xml
    public static class XMLParser {
        private static int numberOfDistinctLink = 0;

        public static void parseXMLFile(String fileName, String outputFileName) {
            try {
                /// Distinct list of link to calculate N
                List<String> distinctLinkList = new ArrayList<>();
                List<HashMap<String, List<String>>> outputList = new ArrayList<>();
                Configuration conf = new Configuration();
                FileSystem fs = FileSystem.get(conf);

                // Check the input and out file
                Path inputFile = new Path(fileName);
                Path outputFile = new Path(outputFileName);
                if (!fs.exists(inputFile)) {
                    LOG.info("File not exist");
                    throw new IOException();
                } if (!fs.isFile(inputFile)) {
                    LOG.info("Input file is not a file");
                    throw new IOException();
                } if (fs.exists(outputFile)) {
                    LOG.info("Pagerank output file exist");
                    fs.delete(outputFile, true);
                }

                BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(inputFile)));
                FSDataOutputStream wr = fs.create(outputFile);

                SAXParserFactory factory = SAXParserFactory.newInstance();
                SAXParser saxParser = factory.newSAXParser();
                String line;
                while ((line = br.readLine()) != null) {
                    /// Handler to parse xml
                    MyXmlHandler handler = new MyXmlHandler();
                    /// Convert a line into valid xml
                    line = "<data>" + line + "</data>";
                    /// Parse a input line
                    saxParser.parse(new InputSource(new StringReader(line)), handler);
                    /// Validation check and save them
                    if (!handler.getLineLinkLine().isEmpty())
                        outputList.add(handler.getLineLinkLine());
                    if (handler.getLinkList() != null && !handler.getLinkList().isEmpty()) {
                        for (String link : handler.getLinkList()) {
                            if (!distinctLinkList.contains(link))
                                distinctLinkList.add(link);
                        }
                    }
                }
                /// numberOfDistinctLink represent the N
                numberOfDistinctLink = distinctLinkList.size();
                /// Prepare the desired output format for mapreduce
                for (HashMap<String, List<String>> output : outputList) {
                    for (String title : output.keySet()) {
                        int numberOfAdjacent = output.get(title).size();
                        if (numberOfAdjacent <= 0)
                            continue;
                        /// Split each link and their adjacent list by "######"
                        String lineData = title + "######";
                        for (String link : output.get(title)) {
                            /// Split each adjacent link and their pagrank by ":::=:::"
                            /// and each element separated by ":#:#:#"
                            /// numberOfDistinctLink represent the N
                            lineData += link + ":::=:::" + (1 / (double) (numberOfDistinctLink * numberOfAdjacent)) + ":#:#:#";
                        }
                        /// Remove last ":#:#:#" string
                        if (lineData.length() > 6)
                            lineData = lineData.substring(0, lineData.length() - 6);
                        /// So each line of the file follow the below format
                        /// Link1######Link2:::=:::pagerank2:#:#:#Link3:::=:::pagerank3:#:#:#Link4:::=:::pagerank4
                        wr.writeBytes(lineData + "\n");
                    }
                }
                br.close();
                wr.close();
            } catch (Exception e) {
                LOG.info("Error" + e.getMessage());
            }
        }

        public static int getNumberOfDistinctLink() {
            return numberOfDistinctLink;
        }
    }

    public static class MyXmlHandler extends DefaultHandler {
        boolean isTitle = false;
        boolean isText = false;
        String titleData = "";
        String textData = "";
        List<String> linkData = new ArrayList<>();

        List<String> distinctLink = new ArrayList<>();
        /// hold XML start tag
        public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
            if (qName.equalsIgnoreCase("title")) {
                isTitle = true;
            }
            if (qName.equalsIgnoreCase("text")) {
                isText = true;
            }
        }
        /// hold XML end tag
        public void endElement(String uri, String localName, String qName) throws SAXException {
            if (qName.equalsIgnoreCase("title")) {
                if (!distinctLink.contains(titleData))
                    distinctLink.add(titleData);
                isTitle = false;
            }
            if (qName.equalsIgnoreCase("text")) {
                try {
                    /// find out the content start with [[ and end with ]]
                    Pattern p = Pattern.compile("\\[\\[([^\\]]+)\\]\\]");
                    Matcher m = p.matcher(textData);
                    while (m.find()) {
                        String link = m.group(1);
                        linkData.add(link);
                        if (!distinctLink.contains(link))
                            distinctLink.add(link);
                    }
                } catch (Exception e) {
                    LOG.error(e);
                }
                isText = false;
            }
        }
        /// hold XML body
        public void characters(char ch[], int start, int length) throws SAXException {
            if (isTitle) {
                titleData += new String(ch, start, length);
            }
            if (isText) {
                textData += (new String(ch, start, length));
            }
        }

        /// Return the each line after parsing and converted into desired format
        public HashMap<String, List<String>> getLineLinkLine() {
            if (titleData != null && !titleData.isEmpty() && linkData != null && !linkData.isEmpty())
                return new HashMap<String, List<String>>() {{
                    put(titleData, linkData);
                }};
            return new HashMap<String, List<String>>();
        }
        /// Return the list of distinct link
        public List<String> getLinkList() {
            if (titleData != null && !titleData.isEmpty() && linkData != null && !linkData.isEmpty())
                return distinctLink;
            return null;
        }
    }
}

