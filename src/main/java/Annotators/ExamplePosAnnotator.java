package Annotators;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.jcas.JCas;
import org.cleartk.ml.CleartkSequenceAnnotator;


/**
 * Simple annotator to create features for a POS tagging task
 * Generic output type is String because the outcome of the classifier will be strings corresponding to part-of-speech tags
 */
public class ExamplePosAnnotator extends CleartkSequenceAnnotator<String> {

    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {

    }
}

