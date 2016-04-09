package example.deeplearning.nn.util;

import example.deeplearning.nn.layers.ProjectionLayer;

/**
 * Created by b1012059 on 2016/04/03.
 */
public abstract class HyperParameters {

    private int vocab;
    private int nIn;
    private int nOut;
    private int numEpochs;
    private int batchSize;
    private int layerSize;
    private int minWordFrequency;
    private double learningRate;
    private double minLearningRate;
    private double learningDecayRate;
    private ProjectionLayer lookupTable;
    private boolean useAdaGrad;
    private double dropOut;
    private int windowSize;
    private long seed;
    private int[] nHidden;
    private int inputSize;
    private String activation;
    private double corruptionLevel;


    public HyperParameters(HyperParameters.Builder builder) {
        this.vocab = builder.vocab;
        this.nIn = builder.nIn;
        this.nHidden = builder.nHidden;
        this.nOut = builder.nOut;
        this.windowSize = builder.windowSize;
        this.numEpochs = builder.numEpochs;
        this.batchSize = builder.batchSize;
        this.layerSize = builder.layerSize;
        this.learningRate = builder.learningRate;
        this.minLearningRate = builder.minLearningRate;
        this.minWordFrequency = builder.minWordFrequency;
        this.learningDecayRate = builder.learningDecayRate;
        this.seed = builder.seed;
        this.useAdaGrad = builder.useAdaGrad;
        this.dropOut = builder.dropOut;
        this.inputSize = builder.inputSize;
        this.activation = builder.activation;
        this.corruptionLevel = builder.corruptionLevel;

    }

    public HyperParameters(){
    }

    public String getActivation(){
        return activation;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getLearningDecayRate() {
        return learningDecayRate;
    }

    public double getDropOut() {
        return dropOut;
    }

    public boolean isUseAdeGrad() {
        return useAdaGrad;
    }

    public double getMinLearningRate() {
        return minLearningRate;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getLayerSize() {
        return layerSize;
    }

    public int getMinWordFrequency() {
        return minWordFrequency;
    }

    public int getInputSize(){
        return inputSize;
    }

    public int getNIn() {
        return nIn;
    }
    
    public int[] getNHidden(){
        return nHidden;
    }

    public int getNOut() {
        return nOut;
    }

    public int getVocab() {
        return vocab;
    }

    public int getNumEpochs() {
        return numEpochs;
    }

    public ProjectionLayer getLookupTable() {
        return lookupTable;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public double getCorruptionLevel(){
        return corruptionLevel;
    }

    public long getSeed() {
        return seed;
    }

    public void setVocab(int vocab){
        this.vocab = vocab;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setMinLearningRate(double minLearningRate) {
        this.minLearningRate = minLearningRate;
    }

    public void setLearningDecayRate(double learningDecayRate) {
        this.learningDecayRate = learningDecayRate;
    }

    public void setNumEpochs(int numEpochs) {
        this.numEpochs = numEpochs;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public void setMinWordFrequency(int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
    }

    public void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }

    public void setWindowSize(int windowSize) {
        this.windowSize = windowSize;
    }

    public void setLookupTable(ProjectionLayer lookupTable) {
        this.lookupTable = lookupTable;
    }

    public void setNIn(int nIn) {
        this.nIn = nIn;
    }
    
    public void setNHidden(int nHidden[]){
        this.nHidden = nHidden;
    }

    public void setNOut(int nOut) {
        this.nOut = nOut;
    }

    public void setDropOut(double dropOut) {
        this.dropOut = dropOut;
    }

    public void setInputSize(int inputSize){
        this.inputSize = inputSize;
    }

    public void setActivation(String activation){
        this.activation = activation;
    }

    public void setCorruptionLevel(double corruptionLevel){
        this.corruptionLevel = corruptionLevel;
    }

    public static abstract class Builder<T extends HyperParameters.Builder<T>> {

        private int vocab;
        private int nIn;
        private int nOut;
        private int numEpochs;
        private int batchSize;
        private int layerSize = 1;
        private int minWordFrequency;
        private double learningRate;
        private double minLearningRate;
        private double learningDecayRate;
        private ProjectionLayer lookupTable = null;
        private boolean useAdaGrad = false;
        private double dropOut;
        private int windowSize;
        private long seed;
        private int[] nHidden;
        private int inputSize;
        private String activation = "sigmoid";
        private double corruptionLevel;


        public Builder(){
            this.vocab = 0;
            this.nIn = 0;
            this.nOut = 0;
            this.numEpochs = 1000;
            this.batchSize = 50;
            this.minWordFrequency = 1;
            this.learningRate = 1e-1;
            this.minLearningRate = 1e-3;
            this.learningDecayRate = 1e-2;
            this.dropOut = 0.0;
            this.windowSize = 3;
            this.seed = 123;
            this.corruptionLevel = 0.3;
        }

        public Builder<T> vocab(int vocab){
            this.vocab = vocab;
            return this;
        }

        public Builder<T> inputSize(int inputSize){
            this.inputSize = inputSize;
            return this;
        }

        public Builder<T> nIn(int nIn){
            this.nIn = nIn;
            return this;
        }
        
        public Builder<T> nHidden(int nHidden[]){
            this.nHidden = nHidden;
            return this;
        }

        public Builder<T> nOut(int nOut){
            this.nOut = nOut;
            return this;
        }

        public Builder<T> numEpochs(int numEpochs){
            this.numEpochs = numEpochs;
            return this;
        }

        public Builder<T> batchSize(int batchSize){
            this.batchSize = batchSize;
            return this;
        }

        public Builder<T> layerSize(int layerSize){
            this.layerSize = layerSize;
            return this;
        }

        public Builder<T> minWordFrequency(int minWordFrequency){
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder<T> learningRate(double learningRate){
            this.learningRate = learningRate;
            return this;
        }

        public Builder<T> learningDecayRate(double learningDecayRate){
            this.learningDecayRate = learningDecayRate;
            return this;
        }

        public Builder<T> minLearningRate(double minLearningRate){
            this.minLearningRate = minLearningRate;
            return this;
        }

        public Builder<T> lookupTable(ProjectionLayer lookupTable){
            this.lookupTable = lookupTable;
            return this;
        }

        public Builder<T> useAdaGrad(boolean useAdaGrad){
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder<T> dropOut(double dropOut){
            this.dropOut = dropOut;
            return this;
        }

        public Builder<T> windowSize(int windowSize){
            this.windowSize = windowSize;
            return this;
        }

        public Builder<T> seed(long seed){
            this.seed = seed;
            return this;
        }

        public Builder<T> activation(String activation){
            this.activation  = activation;
            return this;
        }

        public Builder<T> corruptionLevel(double corruptionLevel){
            this.corruptionLevel = corruptionLevel;
            return this;
        }

        public abstract <E extends HyperParameters> E build();

    }
}
