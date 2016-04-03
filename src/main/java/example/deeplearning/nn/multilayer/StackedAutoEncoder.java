package example.deeplearning.nn.multilayer;

import example.deeplearning.nn.layers.AutoEncoder;
import example.deeplearning.nn.layers.HiddenLayer;
import example.deeplearning.nn.layers.LogisticRegression;
import example.deeplearning.nn.util.Dropout;
import example.deeplearning.nn.util.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * StackedAutoEncoderの実現
 * Created by b1012059 on 2015/09/03.
 * @author Wataru Matsudate
 */
public class StackedAutoEncoder {

    private int layerSize;
    private int nIn;
    private int hiddenSize[];
    private int N;
    private HiddenLayer hLayer[];
    private AutoEncoder aeLayer[];
    private LogisticRegression logLayer;
    private Random rng;
    private String activation;

    public StackedAutoEncoder(int INPUT, int HIDDEN[], int OUTPUT, int N, Random rng, String activation) {

        int inputLayer;
        this.N = N;
        this.nIn = INPUT;
        this.layerSize = HIDDEN.length;
        this.hiddenSize = HIDDEN;
        this.aeLayer = new AutoEncoder[layerSize];
        this.hLayer = new HiddenLayer[layerSize];
        this.activation = activation;

        //randomの種
        if (rng == null) this.rng = new Random(1234);
        else this.rng = rng;

        //Hidden layerとAutoEncoder layerの初期化
        for (int i = 0; i < layerSize; i++) {
            if (i == 0) {
                inputLayer = INPUT;
            } else {
                inputLayer = this.hiddenSize[i - 1];
            }

            //AutoEncoder layer
            this.aeLayer[i] = new AutoEncoder(this.N, inputLayer, this.hiddenSize[i],
                    null, null, rng, activation);
            //hLayer[i].wIO, hLayer[i].bias, rng, activation);
        }
        this.logLayer = new LogisticRegression(this.hiddenSize[this.layerSize - 1], OUTPUT, this.N, rng);

    }

    /**
     * pre-trainingメソッド
     * AutoEncoderの学習を行う
     *
     * @param inputData
     * @param learningRate
     * @param epochs
     * @param corruptionLevel
     */
    public void preTraining(int inputData[][], double learningRate, int epochs, double corruptionLevel) {
        double[] inputLayer = new double[0];
        int prevInputSize;
        double[] prevInput;

        for (int i = 0; i < layerSize; i++) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int n = 0; n < N; n++) {
                    for (int j = 0; j <= i; j++) {
                        if (j == 0) {
                            inputLayer = new double[nIn];
                            for (int k = 0; k < nIn; k++) inputLayer[k] = inputData[n][k];
                        } else {
                            if (j == 1) prevInputSize = nIn;
                            else prevInputSize = hiddenSize[j - 2];

                            prevInput = new double[prevInputSize];
                            for (int k = 0; k < prevInputSize; k++) prevInput[k] = inputLayer[k];

                            inputLayer = new double[hiddenSize[j - 1]];
                            //hLayer[j-1].sampleHgive(prevInput, inputLayer);
                            aeLayer[j - 1].encoder(prevInput, inputLayer);
                        }
                    }
                    aeLayer[i].train(inputLayer, learningRate, corruptionLevel);
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
     * @param learningRate
     * @param epochs
     * @param decayRate
     */
    public void fineTuning(int inputData[][], int teach[][], double learningRate, int epochs,
                           double decayRate, boolean dropout, double pDropout) {
        int nLayer;
        double layerInput[] = new double[0];
        double prevLayerInput[] = new double[nIn];
        double defaultLR = learningRate;
        List<int[]> dropoutMask;
        List<double[]> layerOutput;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int n = 0; n < N; n++) {

                dropoutMask = new ArrayList<>(layerSize);
                layerOutput = new ArrayList<>(layerSize+1);

                for (int i = 0; i < layerSize; i++) {
                    if (i == 0)for(int j = 0; j < inputData[n].length; j++)prevLayerInput[j] = inputData[n][j];
                    else prevLayerInput = layerInput;

                    nLayer = prevLayerInput.length;
                    layerOutput.add(prevLayerInput);

                    //Hidden layer
                    hLayer[i] = new HiddenLayer(nLayer, hiddenSize[i],
                            aeLayer[i].getWightIO(), aeLayer[i].getEncodeBias(), N, rng, activation);
                    layerInput = new double[hiddenSize[i]];
                    hLayer[i].forwardCal(prevLayerInput, layerInput);

                    //Dropout
                    if(dropout){
                        int mask[];
                        mask = Dropout.dropout(hiddenSize[i], pDropout, rng);
                        for(int k = 0; k < hiddenSize[i]; k++) layerInput[k] *= mask[k];

                        dropoutMask.add(mask.clone());
                    }
                }

                //logistic layer forward and backward
                double prevW[][];
                double dLogLayer[] = logLayer.train(layerInput, teach[n], learningRate);
                layerOutput.add(layerInput);

                //hidden layers backward
                double prevDout[] = dLogLayer;
                double dOutput[] = new double[0];
                for (int k = layerSize-1; k >= 0; k--) {
                    if (k == layerSize - 1) {
                        prevW = logLayer.getW();
                        //hLayer[k].backwardCal(input.get(k), dhOutput, output.get(k), dOutput, logLayer.getW(), learningRate);
                    } else {
                        prevDout = dOutput;
                        prevW = hLayer[k+1].getW();
                        //hLayer[k].backwardCal(input.get(k), dhOutput, output.get(k), dhOutput, hLayer[k + 1].getW(), learningRate);
                    }

                    //Dropout
                    if(dropout){
                        for(int j = 0; j < prevDout.length; j++){
                            prevDout[j] *= dropoutMask.get(k)[j];
                        }
                    }

                    dOutput = new double[hiddenSize[k]];
                    hLayer[k].backwardCal(layerOutput.get(k), dOutput, layerOutput.get(k+1),
                            prevDout, prevW, learningRate);
                }

            }
            //Update LearningRate
            if (learningRate > 1E-3)
                learningRate = utils.updateLR(defaultLR, decayRate, epoch);
            //System.out.println(learningRate);
        }
    }

    /**
     * Testing Data Method
     *
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
}