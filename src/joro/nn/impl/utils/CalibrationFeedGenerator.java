package joro.nn.impl.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import joro.nn.impl.core.Feed;

public final class CalibrationFeedGenerator {
  public static List<Feed> generate(String path) {
    return generate(Paths.get(path));
  }

  public static List<Feed> generate(Path path) {
    List<Feed> result = new ArrayList<>();

    try {
      BufferedReader br = Files.newBufferedReader(path, Charset.forName("UTF-8"));
      String line = null;
      while((line = br.readLine()) != null) {
        StringTokenizer tokenizer = new StringTokenizer(line, "=");
        if (tokenizer.countTokens() == 2) {
          String inputToken = tokenizer.nextToken().trim();
          StringTokenizer inputTokenizer = new StringTokenizer(inputToken, ",");
          List<Double> inputs = new ArrayList<Double>();
          while (inputTokenizer.hasMoreTokens()) {
            String input = inputTokenizer.nextToken().trim();
            double value = Double.parseDouble(input);
            inputs.add(value);
          }

          String outputToken = tokenizer.nextToken().trim();
          StringTokenizer outputTokenizer = new StringTokenizer(outputToken, ",");
          List<Double> outputs = new ArrayList<Double>();
          while (outputTokenizer.hasMoreTokens()) {
            String output = outputTokenizer.nextToken().trim();
            double value = Double.parseDouble(output);
            outputs.add(value);
          }

          Feed feed = new Feed(inputs.size(), outputs.size());

          double[] inputsArr = new double[inputs.size()];
          for (int i = 0; i < inputsArr.length; i++) {
            inputsArr[i] = inputs.get(i);
          }
          feed.setInput(inputsArr);

          double[] outputsArr = new double[outputs.size()];
          for (int i = 0; i < outputsArr.length; i++) {
            outputsArr[i] = outputs.get(i);
          }
          feed.setOutput(outputsArr);

          result.add(feed);
        } else {
          throw new IllegalArgumentException("Incorrect format.\n" + 
                                             "Expected: [input 1],[input 2] ... [input n]=[output 1], [output 2] ... [output n]\n" + 
                                             "Received: " + line);
        }
      }
      return result;
    } catch (IOException ioEx) {
      throw new RuntimeException(ioEx);
    }
  }
}
