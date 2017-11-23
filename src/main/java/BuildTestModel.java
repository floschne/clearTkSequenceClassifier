import Annotators.ExamplePosAnnotator;
import org.apache.uima.cas.CAS;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.component.ViewCreatorAnnotator;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.cleartk.corpus.penntreebank.TreebankGoldAnnotator;
import org.cleartk.ml.jar.Train;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class BuildTestModel {
    public static void main(String[] args) throws Exception {
        // A collection reader that creates one CAS per file, containing the file's URI
        File file = new File(ClassLoader.getSystemClassLoader().getResource("data/").getFile());
        List<File> files = Arrays.asList(file.listFiles());
        CollectionReader reader = UriCollectionReader.getCollectionReaderFromFiles(files);

        // The pipeline of annotators
        AggregateBuilder builder = new AggregateBuilder();

        // An annotator that creates an empty treebank view in the CAS
        builder.add(AnalysisEngineFactory.createEngineDescription(ViewCreatorAnnotator.class), ViewCreatorAnnotator.PARAM_VIEW_NAME, "TreebankView");

        // An annotator that reads the treebank-formatted text into the treebank view
        builder.add(UriToDocumentTextAnnotator.getDescription());

        // An annotator that uses the treebank text to add sentences, tokens and POS tags to the CAS
        builder.add(TreebankGoldAnnotator.getDescriptionPOSTagsOnly());

        // The POS annotator, configured to write training data
        String outputDirectory = ".";
        builder.add(ExamplePosAnnotator.getWriterDescription(outputDirectory));

        // Run the pipeline of annotators on each of the CASes produced by the reader
        SimplePipeline.runPipeline(reader, builder.createAggregateDescription());

        // Train a classifier on the training data, and package it into a .jar file
        Train.main(outputDirectory);
    }
}