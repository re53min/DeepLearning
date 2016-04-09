package example.deeplearning.nn.multilayer;

import example.deeplearning.nn.layers.AutoEncoder;
import example.deeplearning.nn.layers.HiddenLayer;
import example.deeplearning.nn.layers.LogisticRegression;
import example.deeplearning.nn.util.Dropout;
import example.deeplearning.nn.util.HyperParameters;
import example.deeplearning.nn.util.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * StackedAutoEncoderの実現
 * Created by b1012059 on 2015/09/03.
 * @author Wataru Matsudate
 */
public class StackedAutoEncoder extends HyperParameters{

    private int layerSize;
    private int nIn;
    private int nOut;
    private int hiddenSize[];
    private int N;
    private HiddenLayer hLayer[];
    private AutoEncoder aeLayer[];
    private LogisticRegression logLayer;
    private Random rng;
    private String activation;

    public StackedAutoEncoder(StackedAutoEncoder.Builder builder){
        super(builder);

        int inputLayer;
        this.N = builder.N;
        this.nIn = builder.nIn;
        this.nOut = builder.nOut;
        this.hiddenSize = builder.nHidden;
        this.layerSize = hiddenSize.length;
        this.aeLayer = new AutoEncoder[layerSize];
        this.hLayer = new HiddenLayer[layerSize];
        this.activation = builder.activation;
        this.rng = new Random(builder.seed);

        //Hidden layerとAutoEncoder layerの初期化
        for (int i = 0; i < layerSize; i++) {
            if (i == 0) inputLayer = nIn;
            else inputLayer = this.hiddenSize[i-1];

            //AutoEncoder layer
            this.aeLayer[i] = new AutoEncoder(N, inputLayer, hiddenSize[i],
                    null, null, rng, activation);
            //hLayer[i].wIO, hLayer[i].bias, rng, activation);
        }
        this.logLayer = new LogisticRegression(hiddenSize[layerSize-1], nOut, N, rng);


    }

    /**
     * pre-trainingメソッド
     * AutoEncoderの学習を行う
     * @param inputData
     */
    public void preTraining(double inputData[][]) {
        double[] inputLayer = new double[0];
        int prevInputSize;
        double[] prevInput;

        for (int i = 0; i < layerSize; i++) {
            for (int epoch = 0; epoch < getNumEpochs(); epoch++) {
                for (int n = 0; n < N; n++) {
                    for (int j = 0; j <= i; j++) {
                        if (j == 0) {
                            inputLayer = new double[nIn];
                            for (int k = 0; k < nIn; k++) inputLayer[k] = inputData[n][k];
                        } else {
                            if (j == 1) prevInputSize = nIn;
                            else prevInputSize = hiddenSize[j-2];

                            prevInput = new double[prevInputSize];
                            for (int k = 0; k < prevInputSize; k++) prevInput[k] = inputLayer[k];

                            inputLayer = new double[hiddenSize[j-1]];
                            //hLayer[j-1].sampleHgive(prevInput, inputLayer);
                            aeLayer[j-1].encoder(prevInput, inputLayer);
                        }
                    }
                    aeLayer[i].train(inputLayer, getLearningRate(), getCorruptionLevel());
                }
            }
        }
    }

    /**
     * fine-tuningメソッド
     * pre-trainingの結果を使い、出力層を追加して
     * バックプロパゲーション
     * とりあえずDropout
     *
     * @param inputData
     * @param teach
     */
    public void fineTuning(double inputData[][], int teach[][]) {
        int nLayer;
        double layerInput[] = new double[0];
        double[] prevLayerInput;
        double learningRate = getLearningRate();
        List<int[]> dropoutMask;
        List<double[]> layerOutput;

        for (int epoch = 0; epoch < getNumEpochs(); epoch++) {
            for (int n = 0; n < N; n++) {

                dropoutMask = new ArrayList<>(layerSize);
                layerOutput = new ArrayList<>(layerSize+1);

                for (int i = 0; i < layerSize; i++) {
                    if (i == 0) prevLayerInput = inputData[n];
                    else prevLayerInput = layerInput;

                    nLayer = prevLayerInput.length;
                    layerOutput.add(prevLayerInput);

                    //Hidden layer
                    hLayer[i] = new HiddenLayer(nLayer, hiddenSize[i],
                            aeLayer[i].getWightIO(), aeLayer[i].getEncodeBias(), N, rng, activation);
                    layerInput = new double[hiddenSize[i]];
                    hLayer[i].forwardCal(prevLayerInput, layerInput);

                    //Dropout
                    int mask[];
                    mask = Dropout.dropout(hiddenSize[i], getDropOut(), rng);
                    for(int k = 0; k < hiddenSize[i]; k++) layerInput[k] *= mask[k];
                    dropoutMask.add(mask.clone());

                }

                //logistic layer forward and backward
                double prevW[][];
                double dLogLayer[] = logLayer.train(layerInput, teach[n], learningRate);
                layerOutput.add(layerInput);

                //hidden layers backward
                double prevDout[] = dLogLayer;
                double dOutput[] = new double[0];
                for (int k = layerSize-1; k >= 0; k--) {
                    if (k == layerSize-1) {
                        prevW = logLayer.getW();
                        //hLayer[k].backwardCal(input.get(k), dhOutput, output.get(k), dOutput, logLayer.getW(), learningRate);
                    } else {
                        prevDout = dOutput;
                        prevW = hLayer[k+1].getW();
                        //hLayer[k].backwardCal(input.get(k), dhOutput, output.get(k), dhOutput, hLayer[k + 1].getW(), learningRate);
                    }

                    //Dropout
                    for(int j = 0; j < prevDout.length; j++) prevDout[j] *= dropoutMask.get(k)[j];

                    dOutput = new double[hiddenSize[k]];
                    hLayer[k].backwardCal(layerOutput.get(k), dOutput, layerOutput.get(k+1),
                            prevDout, prevW, learningRate);
                }

            }
            //Update LearningRate
            if(isUseAdeGrad()) {
                if (learningRate > getMinLearningRate())
                    learningRate = utils.updateLR(getLearningRate(), getLearningDecayRate(), epoch);
            }
        }
    }

    /**
     * Testing Data Method
     * @param input
     * @param output
     */
    public void reconstruct(int input[], double output[]) {
        double layerInput[] = new double[0];
        double prevLayerInput[] = new double[nIn];

        for (int i = 0; i < nIn; i++) prevLayerInput[i] = input[i];

        //Hidden Layer
        for (int i = 0; i < layerSize; i++) {
            layerInput = new double[hiddenSize[i]];
            hLayer[i].forwardCal(prevLayerInput, layerInput);

            if (i < layerSize) {
                prevLayerInput = new double[hiddenSize[i]];
                for (int j = 0; j < hiddenSize[i]; j++) prevLayerInput[j] = layerInput[j];
            }
        }

        //Output Layer
        logLayer.reconstruct(layerInput, output);
    }

    public static class Builder extends example.deeplearning.nn.util.HyperParameters.Builder<StackedAutoEncoder.Builder>{

        private int nOut;
        private int N;
        private int nIn;
        private int layerSize;
        private int nHidden[];
        private String activation;
        private long seed;

        public Builder(){
        }

        public StackedAutoEncoder.Builder N(int N){
            this.N = N;
            return this;
        }

        public StackedAutoEncoder.Builder nIn(int nIn){
            this.nIn = nIn;
            return this;
        }

        public StackedAutoEncoder.Builder layerSize(int layerSize){
            this.layerSize = layerSize;
            return this;
        }

        public StackedAutoEncoder.Builder nHidden(int nHidden[]){
            this.nHidden = nHidden;
            return this;
        }

        public StackedAutoEncoder.Builder nOut(int nOut){
            this.nOut = nOut;
            return this;
        }

        public StackedAutoEncoder.Builder activation(String activation){
            this.activation = activation;
            return this;
        }

        public StackedAutoEncoder.Builder seed(long seed){
            this.seed = seed;
            return this;
        }

        public StackedAutoEncoder build(){
            return new StackedAutoEncoder(this);
        }
    }
}