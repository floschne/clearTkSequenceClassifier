package Annotators;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.type.Sentence;
import org.apache.uima.fit.type.Token;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instances;
import org.cleartk.ml.feature.extractor.CleartkExtractor;
import org.cleartk.ml.feature.extractor.CoveredTextExtractor;
import org.cleartk.ml.feature.extractor.FeatureExtractor1;
import org.cleartk.ml.feature.function.*;
import org.cleartk.ml.jar.DefaultDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.ml.opennlp.maxent.MaxentStringOutcomeDataWriter;
import org.cleartk.ml.viterbi.DefaultOutcomeFeatureExtractor;
import org.cleartk.ml.viterbi.ViterbiDataWriterFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


/**
 * Simple annotator to create features for a POS tagging task
 * Generic output type is String because the outcome of the classifier will be strings corresponding to part-of-speech tags
 */
public class ExamplePosAnnotator extends CleartkSequenceAnnotator<String> {

    public static final String DEFAULT_OUTPUT_DIRECTORY = "tmp/pos";

    private FeatureExtractor1<Token> tokenFeatureExtractor;

    private CleartkExtractor<Token, Token> contextFeatureExtractor;


    /**
     * The initialize method is used to initialize feature extractors (as well as feature extractor functions) and reading
     * parameters as necessary from the UIMA Context.
     *
     * @param context the UIMA context
     * @throws ResourceInitializationException
     */
    @Override
    public void initialize(UimaContext context) throws ResourceInitializationException {
        super.initialize(context);

        // create a function feature extractor that creates features corresponding to the token
        // Note the difference between feature extractors and feature functions here. Feature extractors take an Annotation
        // from the JCas and extract features from it. Feature functions take the features produced by the feature extractor
        // and generate new features from the old ones. Since feature functions don’t need to look up information in the JCas,
        // they may be more efficient than feature extractors. So, the e.g. the CharacterNgramFeatureFunction simply extract
        // suffixes from the text returned by the CoveredTextExtractor.
        this.tokenFeatureExtractor = new FeatureFunctionExtractor<Token>(
                // the FeatureExtractor that takes the token annotation from the JCas and produces the covered text
                new CoveredTextExtractor<Token>(),
                // feature function that produces the lower cased word (based on the output of the CoveredTextExtractor)
                new LowerCaseFeatureFunction(),
                // feature function that produces the capitalization type of the word (e.g. all uppercase, all lowercase...)
                new CapitalTypeFeatureFunction(),
                // feature function that produces the numeric type of the word (numeric, alphanumeric...)
                new NumericTypeFeatureFunction(),
                // feature function that produces the suffix of the word as character bigram (last two chars of the word)
                new CharacterNgramFeatureFunction(CharacterNgramFeatureFunction.Orientation.RIGHT_TO_LEFT, 0, 2),
                // feature function that produces the suffix of the word as character trigram (last three chars of the word)
                new CharacterNgramFeatureFunction(CharacterNgramFeatureFunction.Orientation.RIGHT_TO_LEFT, 0, 3));

        // create a feature extractor that extracts the surrounding token texts (within the same sentence)
        this.contextFeatureExtractor = new CleartkExtractor<Token, Token>(Token.class,
                // the FeatureExtractor that takes the token annotation from the JCas and produces the covered text
                new CoveredTextExtractor<Token>(),
                // also include the two preceding words
                new CleartkExtractor.Preceding(2),
                // and the two following words
                new CleartkExtractor.Following(2));
    }

    /**
     * The process method defines how features and labels are extracted from the annotations of a JCas, and how
     * classifier predictions are used to create new JCas annotations.
     *
     * @param jCas the UIMA CAS containing the document
     * @throws AnalysisEngineProcessException
     */
    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {
        // for each sentence in the document, generate training/classification instances
        for (Sentence sentence : JCasUtil.select(jCas, Sentence.class)) {
            // features for all tokens in the sentence
            List<List<Feature>> tokenFeatureLists = new ArrayList<List<Feature>>();
            // the POS tag label of the tokens in the sentence
            List<String> tokenOutcomes = new ArrayList<String>();

            // for each token, extract features and the outcome
            List<Token> tokensInSentence = JCasUtil.selectCovered(jCas, Token.class, sentence);
            for (Token token : tokensInSentence) {
                // create the list of features for the current token
                List<Feature> tokenFeatures = new ArrayList<Feature>();
                // Note: there could be more featureExtractors (also custom made)
                // apply the tokenFeatureExtractor and add the resulting features to the token's feature list
                tokenFeatures.addAll(this.tokenFeatureExtractor.extract(jCas, token));
                // apply the contextFeatureExtractor and add the resulting features to the token's feature list
                tokenFeatures.addAll(this.contextFeatureExtractor.extractWithin(jCas, token, sentence));
                // add the extracted features of the token to the list of features for all tokens
                tokenFeatureLists.add(tokenFeatures);

                // if we are in training mode add the expected POS tag (= label) to the list of POS tags of the token
                if (this.isTraining())
                    tokenOutcomes.add(token.getPos());
            }

            // for training, write instances to the data write
            if (this.isTraining())
                this.dataWriter.write(Instances.toInstances(tokenOutcomes, tokenFeatureLists));

                // for classification, set the tokens POS tags from the classifier output
            else {
                // classify / predict the POS tags of all tokens in the sentence by using the tokensFeatureLists
                List<String> predictedPosTags = this.classifier.classify(tokenFeatureLists);
                Iterator<Token> tokensIter = tokensInSentence.iterator();

                // assign the predicted POS tags to the tokens in the sentence
                for (String outcome : predictedPosTags)
                    tokensIter.next().setPos(outcome);
            }
        }
    }
    public static AnalysisEngineDescription getClassifierDescription(String modelFileName)
            throws ResourceInitializationException {
        return AnalysisEngineFactory.createEngineDescription(
                ExamplePosAnnotator.class,
                GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
                modelFileName);
    }

    public static AnalysisEngineDescription getWriterDescription(String outputDirectory)
            throws ResourceInitializationException {
        return AnalysisEngineFactory.createEngineDescription(
                ExamplePosAnnotator.class,
                CleartkSequenceAnnotator.PARAM_DATA_WRITER_FACTORY_CLASS_NAME,
                ViterbiDataWriterFactory.class.getName(),
                DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
                outputDirectory,
                ViterbiDataWriterFactory.PARAM_DELEGATED_DATA_WRITER_FACTORY_CLASS,
                DefaultDataWriterFactory.class.getName(),
                DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
                MaxentStringOutcomeDataWriter.class.getName(),
                ViterbiDataWriterFactory.PARAM_OUTCOME_FEATURE_EXTRACTOR_NAMES,
                new String[] { DefaultOutcomeFeatureExtractor.class.getName() });
    }
}

