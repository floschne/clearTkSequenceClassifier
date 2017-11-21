package Annotators;

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.type.Sentence;
import org.apache.uima.fit.type.Token;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkSequenceAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instances;
import org.cleartk.ml.feature.extractor.CleartkExtractor;
import org.cleartk.ml.feature.extractor.FeatureExtractor1;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


/**
 * Simple annotator to create features for a POS tagging task
 * Generic output type is String because the outcome of the classifier will be strings corresponding to part-of-speech tags
 */
public class ExamplePosAnnotator extends CleartkSequenceAnnotator<String> {

    private FeatureExtractor1<Token> tokenFeatureExtractor;

    private CleartkExtractor contextFeatureExtractor;

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

                // add the expected POS tag (= label) to the list of POS tags of the token if we are in training mode
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

    /**
     * The initialize method is typically used to initialize feature extractors and reading parameters as necessary
     * from the UIMA Context.
     *
     * @param context the UIMA context
     * @throws ResourceInitializationException
     */
    @Override
    public void initialize(UimaContext context) throws ResourceInitializationException {
        super.initialize(context);
    }
}

